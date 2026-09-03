import argparse
import logging
import math
import os
import pickle
import random
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import DatasetDict, load_from_disk
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from duration_model import SpeechLengthPredictor, calculate_remaining_lengths

logger = logging.getLogger(__name__)

CORRUPT_LATENT_ERRORS = (RuntimeError, EOFError, OSError, pickle.UnpicklingError)


def load_tensor(src):
    if isinstance(src, torch.Tensor):
        return src
    if isinstance(src, np.ndarray):
        return torch.from_numpy(src)
    if isinstance(src, (list, tuple)):
        return torch.tensor(src)
    if not isinstance(src, str):
        raise TypeError(f"unsupported latent payload type: {type(src)}")

    t = torch.load(src, map_location="cpu", weights_only=True)
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(t)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"expected tensor in {src}, got {type(t)}")
    return t


def load_tokenizer(tokenizer):
    if tokenizer is None:
        return None
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer, add_bos_token=True, add_eos_token=True)
    return tokenizer


def get_splits(ds, valid_split, valid_ratio, seed):
    def holdout(dataset, reason):
        if valid_ratio <= 0.0:
            raise ValueError(f"{reason}; set --valid_ratio to create a holdout split")
        split = dataset.train_test_split(test_size=valid_ratio, seed=seed)
        return split["train"], split["test"]

    if not isinstance(ds, DatasetDict):
        return holdout(ds, "single dataset provided but --valid_ratio <= 0")

    train_ds = ds["train"] if "train" in ds else ds[next(iter(ds.keys()))]

    if valid_split is not None and valid_split in ds:
        return train_ds, ds[valid_split]

    for name in ["validation", "valid", "dev", "test"]:
        if name in ds:
            return train_ds, ds[name]

    return holdout(train_ds, "dataset has no validation split and --valid_ratio <= 0")


def find_latest_checkpoint(exp_dir):
    ckpts = sorted(glob(os.path.join(exp_dir, "checkpoints", "step_*.pt")))
    return ckpts[-1] if ckpts else None


def save_checkpoint(accelerator, model, optimizer, scheduler, step, exp_dir, keep_last_k):
    ckpt_root = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)
    ckpt_path = os.path.join(ckpt_root, f"step_{step:07d}.pt")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(
            {
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
            },
            ckpt_path,
        )
        for stale in sorted(glob(os.path.join(ckpt_root, "step_*.pt")))[:-keep_last_k]:
            os.remove(stale)

    accelerator.wait_for_everyone()
    logger.info(f"Saved checkpoint: {ckpt_path}")


def load_checkpoint(accelerator, model, optimizer, scheduler, ckpt_path):
    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    accelerator.unwrap_model(model).load_state_dict(ckpt["model"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    accelerator.wait_for_everyone()
    return int(ckpt["step"])


def make_lr_scheduler(optimizer, warmup_steps, total_steps):
    warmup_steps = max(int(warmup_steps), 0)
    total_steps = max(int(total_steps), 1)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def load_titanet(device):
    import nemo.collections.asr as nemo_asr

    titanet = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large"
    )
    titanet = titanet.to(device).eval()
    for p in titanet.parameters():
        p.requires_grad = False
    return titanet


@torch.no_grad()
def extract_speaker_embs_from_batch(titanet, batch, device):
    audio = batch["speaker_audio"].to(device)
    lengths = batch["speaker_audio_lengths"].to(device)
    valid = batch["speaker_audio_valid"]
    B = audio.shape[0]

    speaker_embs = torch.zeros(B, 192, dtype=torch.float32, device=device)
    valid_idx = valid.nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        return speaker_embs

    _, embs = titanet.forward(
        input_signal=audio[valid_idx],
        input_signal_length=lengths[valid_idx],
    )
    speaker_embs[valid_idx] = embs
    return speaker_embs


class DurationDataset(Dataset):
    MAX_RESAMPLE_ATTEMPTS = 10

    def __init__(self, hf_dataset, need_audio_path=False, tokenizer=None, max_text_len=None):
        self.ds = hf_dataset
        self.need_audio_path = need_audio_path
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.ds)

    def _tokenize(self, text):
        if self.tokenizer is None:
            raise RuntimeError("row has no input_ids and no tokenizer was provided")
        if self.max_text_len is not None:
            enc = self.tokenizer(
                text,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
        else:
            enc = self.tokenizer(
                text,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=True,
            )
        return enc["input_ids"][0].long(), enc["attention_mask"][0].to(torch.bool)

    def _encode_text(self, row):
        if row.get("input_ids", None) is not None:
            input_ids = torch.tensor(row["input_ids"], dtype=torch.long)
            if row.get("attention_mask", None) is not None:
                attention_mask = torch.tensor(row["attention_mask"], dtype=torch.bool)
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            return input_ids, attention_mask
        return self._tokenize(row["text"])

    def __getitem__(self, idx):
        current_idx = idx
        last_error = None

        for _ in range(self.MAX_RESAMPLE_ATTEMPTS):
            row = self.ds[current_idx]
            try:
                latents = load_tensor(row["latents"]).float()
            except CORRUPT_LATENT_ERRORS as exc:
                last_error = exc
                logger.warning(
                    f"bad latents at idx {current_idx} ({row.get('filename')}): {exc}; resampling"
                )
                current_idx = random.randrange(len(self.ds))
                continue

            if latents.dim() == 1:
                latents = latents.unsqueeze(-1)

            input_ids, attention_mask = self._encode_text(row)
            duration = row.get("duration", None)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "latents": latents,
                "duration_sec": None if duration is None else float(duration),
                "audio_path": row.get("filename", None) if self.need_audio_path else None,
            }

        raise RuntimeError(f"too many corrupt latents near idx {idx}") from last_error


class DurationCollator:
    MIN_AUDIO_SAMPLES = 16000

    def __init__(self, vocab_size, max_text_len=None, max_latent_len=None, load_audio=False):
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.max_latent_len = max_latent_len
        self.load_audio = load_audio

    def _safe_load_audio(self, audio_path):
        import librosa

        if audio_path is None:
            return None
        try:
            wav, _ = librosa.load(audio_path, sr=16000)
        except Exception:
            return None
        return wav if len(wav) >= self.MIN_AUDIO_SAMPLES else None

    def _collate_audio(self, batch):
        wavs = [self._safe_load_audio(item["audio_path"]) for item in batch]
        lengths = [len(w) if w is not None else 0 for w in wavs]

        audio_batch = np.zeros((len(batch), max(lengths) or 1), dtype=np.float32)
        for i, wav in enumerate(wavs):
            if wav is not None:
                audio_batch[i, : len(wav)] = wav

        return {
            "speaker_audio": torch.from_numpy(audio_batch),
            "speaker_audio_lengths": torch.tensor(lengths, dtype=torch.long),
            "speaker_audio_valid": torch.tensor([w is not None for w in wavs], dtype=torch.bool),
        }

    def __call__(self, batch):
        batch_size = len(batch)
        latent_dim = batch[0]["latents"].shape[-1]
        for item in batch:
            if item["latents"].shape[-1] != latent_dim:
                raise ValueError(
                    f"latent dim mismatch inside batch: "
                    f"{item['latents'].shape[-1]} vs expected {latent_dim}"
                )

        text_lens = [int(item["attention_mask"].sum().item()) for item in batch]
        latent_lens = [int(item["latents"].shape[0]) for item in batch]

        pad_text_len = max(text_lens)
        if self.max_text_len is not None:
            pad_text_len = min(pad_text_len, self.max_text_len)
        pad_latent_len = max(latent_lens)
        if self.max_latent_len is not None:
            pad_latent_len = min(pad_latent_len, self.max_latent_len)

        text_ids = torch.full((batch_size, pad_text_len), self.vocab_size, dtype=torch.long)
        text_padding_mask = torch.ones((batch_size, pad_text_len), dtype=torch.bool)
        latents = torch.zeros((batch_size, pad_latent_len, latent_dim), dtype=torch.float32)
        latent_mask = torch.zeros((batch_size, pad_latent_len), dtype=torch.bool)

        duration_secs = []
        has_duration = True

        for i, item in enumerate(batch):
            valid_text = item["input_ids"][item["attention_mask"].to(torch.bool)]
            valid_text_len = min(valid_text.shape[0], pad_text_len)
            text_ids[i, :valid_text_len] = valid_text[:valid_text_len]
            text_padding_mask[i, :valid_text_len] = False

            valid_latent_len = min(item["latents"].shape[0], pad_latent_len)
            latents[i, :valid_latent_len] = item["latents"][:valid_latent_len]
            latent_mask[i, :valid_latent_len] = True

            if item["duration_sec"] is None:
                has_duration = False
                duration_secs.append(0.0)
            else:
                duration_secs.append(float(item["duration_sec"]))

        out = {
            "text_ids": text_ids,
            "text_padding_mask": text_padding_mask,
            "latents": latents,
            "latent_mask": latent_mask,
        }
        if has_duration:
            out["duration_sec"] = torch.tensor(duration_secs, dtype=torch.float32)
        if self.load_audio:
            out.update(self._collate_audio(batch))

        return out


def masked_l1(pred, target, mask):
    loss = F.l1_loss(pred.float(), target.float(), reduction="none") * mask.to(pred.dtype)
    return loss.sum() / mask.sum().clamp_min(1).to(loss.dtype)


def masked_ce(logits, labels, mask):
    b, t, c = logits.shape
    loss = F.cross_entropy(
        logits.reshape(b * t, c).float(),
        labels.reshape(b * t),
        reduction="none",
    ).reshape(b, t)
    loss = loss * mask.to(loss.dtype)
    return loss.sum() / mask.sum().clamp_min(1).to(loss.dtype)


def expected_frames_from_logits(logits, n_frame_per_class):
    bins = torch.arange(
        logits.shape[-1], device=logits.device, dtype=logits.dtype
    ) * float(n_frame_per_class)
    return (logits.softmax(dim=-1) * bins).sum(dim=-1)


def decode_frames_from_logits(logits, n_frame_per_class):
    return logits.argmax(dim=-1).float() * float(n_frame_per_class)


def total_lengths_from_batch(batch):
    return batch["latent_mask"].sum(dim=1).long()


def prediction_loss(out, targets, mask, loss_type, n_frame_per_class):
    if loss_type == "l1":
        loss = masked_l1(out, targets, mask)
        return loss, masked_l1(out.detach(), targets, mask)

    if loss_type not in ("ce", "hybrid"):
        raise ValueError(f"unknown loss_type: {loss_type}")

    labels = (targets // n_frame_per_class).clamp_min(0).clamp_max(out.shape[-1] - 1)
    loss = masked_ce(out, labels, mask)
    if loss_type == "hybrid":
        loss = loss + masked_l1(expected_frames_from_logits(out, n_frame_per_class), targets, mask)

    pred_frames = decode_frames_from_logits(out.detach(), n_frame_per_class)
    return loss, masked_l1(pred_frames, targets, mask)


def global_prediction_loss(out, total_lengths, loss_type, n_frame_per_class):
    mask = torch.ones(out.shape[0], 1, dtype=torch.bool, device=out.device)
    return prediction_loss(
        out[:, :1], total_lengths.unsqueeze(1), mask, loss_type, n_frame_per_class
    )


def compute_batch(
    model,
    batch,
    loss_type,
    n_frame_per_class,
    text_only_loss_weight,
    speaker_loss_weight,
    codec_rate_hz,
):
    text_ids = batch["text_ids"]
    text_padding_mask = batch["text_padding_mask"]
    latent_mask = batch["latent_mask"]
    device = text_ids.device

    total_lengths = total_lengths_from_batch(batch)
    remaining_targets = calculate_remaining_lengths(total_lengths)

    speaker_emb = batch.get("speaker_embs", None)

    prompted_out = model(
        text_ids=text_ids,
        latent_prompt=batch["latents"],
        text_padding_mask=text_padding_mask,
        latent_padding_mask=~latent_mask,
        speaker_emb=speaker_emb,
    )
    prompt_loss, prompt_frame_mae = prediction_loss(
        prompted_out, remaining_targets, latent_mask, loss_type, n_frame_per_class
    )

    text_only_out = model(
        text_ids=text_ids,
        latent_prompt=None,
        text_padding_mask=text_padding_mask,
    )
    text_only_loss, text_only_frame_mae = global_prediction_loss(
        text_only_out, total_lengths, loss_type, n_frame_per_class
    )

    total_loss = prompt_loss + float(text_only_loss_weight) * text_only_loss

    speaker_loss = torch.zeros((), device=device)
    speaker_frame_mae = torch.zeros((), device=device)
    if speaker_emb is not None and speaker_loss_weight > 0.0:
        speaker_out = model(
            text_ids=text_ids,
            latent_prompt=None,
            text_padding_mask=text_padding_mask,
            speaker_emb=speaker_emb,
        )
        speaker_loss, speaker_frame_mae = global_prediction_loss(
            speaker_out, total_lengths, loss_type, n_frame_per_class
        )
        total_loss = total_loss + float(speaker_loss_weight) * speaker_loss

    stats = {
        "loss": total_loss.detach(),
        "prompt_loss": prompt_loss.detach(),
        "text_only_loss": text_only_loss.detach(),
        "prompt_frame_mae": prompt_frame_mae.detach(),
        "prompt_sec_mae": prompt_frame_mae.detach() / float(codec_rate_hz),
        "text_only_frame_mae": text_only_frame_mae.detach(),
        "text_only_sec_mae": text_only_frame_mae.detach() / float(codec_rate_hz),
        "speaker_loss": speaker_loss.detach(),
        "speaker_frame_mae": speaker_frame_mae.detach(),
        "speaker_sec_mae": speaker_frame_mae.detach() / float(codec_rate_hz),
        "lr": torch.zeros((), device=device),
    }

    if "duration_sec" in batch:
        target_sec = total_lengths.float() / float(codec_rate_hz)
        stats["duration_col_sec_mae"] = F.l1_loss(
            target_sec, batch["duration_sec"].float()
        ).detach()

    return total_loss, stats


@torch.no_grad()
def run_validation(
    model,
    dataloader,
    accelerator,
    loss_type,
    n_frame_per_class,
    text_only_loss_weight,
    speaker_loss_weight,
    codec_rate_hz,
    step,
    titanet_model=None,
):
    model.eval()
    totals = {}
    count = 0

    for batch in dataloader:
        if titanet_model is not None and "speaker_audio" in batch:
            batch["speaker_embs"] = extract_speaker_embs_from_batch(
                titanet_model, batch, accelerator.device
            )

        _, stats = compute_batch(
            model=model,
            batch=batch,
            loss_type=loss_type,
            n_frame_per_class=n_frame_per_class,
            text_only_loss_weight=text_only_loss_weight,
            speaker_loss_weight=speaker_loss_weight,
            codec_rate_hz=codec_rate_hz,
        )

        for k, v in stats.items():
            gathered = accelerator.gather_for_metrics(v.detach().reshape(1))
            totals[k] = totals.get(k, 0.0) + gathered.mean().item()
        count += 1

    if count > 0:
        accelerator.log({f"valid/{k}": v / count for k, v in totals.items()}, step=step)

    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--valid_split", type=str, default=None)
    parser.add_argument("--valid_ratio", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_text_len", type=int, default=None)
    parser.add_argument("--max_latent_len", type=int, default=None)

    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_text_layer", type=int, default=8)
    parser.add_argument("--n_cross_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--output_dim", type=int, default=1)

    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "ce", "hybrid"])
    parser.add_argument("--n_frame_per_class", type=int, default=1)
    parser.add_argument("--text_only_loss_weight", type=float, default=1.0)

    parser.add_argument("--use_speaker_conditioning", action="store_true")
    parser.add_argument("--speaker_emb_dim", type=int, default=192)
    parser.add_argument("--speaker_loss_weight", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--codec_rate_hz", type=float, default=12.5)

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--keep_last_k", type=int, default=5)

    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def build_loaders(args, tokenizer):
    train_hf, valid_hf = get_splits(
        load_from_disk(args.dataset),
        valid_split=args.valid_split,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    def make(hf_split, shuffle, drop_last):
        dataset = DurationDataset(
            hf_split,
            need_audio_path=args.use_speaker_conditioning,
            tokenizer=tokenizer,
            max_text_len=args.max_text_len,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=DurationCollator(
                vocab_size=args.vocab_size,
                max_text_len=args.max_text_len,
                max_latent_len=args.max_latent_len,
                load_audio=args.use_speaker_conditioning,
            ),
            drop_last=drop_last,
        )
        return dataset, loader

    train_ds, train_loader = make(train_hf, shuffle=True, drop_last=True)
    valid_ds, valid_loader = make(valid_hf, shuffle=False, drop_last=False)
    return train_ds, train_loader, valid_ds, valid_loader


def main():
    args = parse_args()

    if args.loss_type != "l1" and args.output_dim <= 1:
        raise ValueError("--loss_type ce/hybrid requires --output_dim > 1")

    os.makedirs(args.exp_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.exp_dir, "log.txt"), mode="a"),
        ],
    )

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum_steps)
    set_seed(args.seed)

    logger.info(f"Experiment directory: {args.exp_dir}")
    logger.info(f"Speaker conditioning: {args.use_speaker_conditioning}")
    logger.info(f"Accelerator state: {accelerator.state}")

    tokenizer = load_tokenizer(args.tokenizer)
    train_ds, train_loader, valid_ds, valid_loader = build_loaders(args, tokenizer)

    model = SpeechLengthPredictor(
        vocab_size=args.vocab_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_text_layer=args.n_text_layer,
        n_cross_layer=args.n_cross_layer,
        n_head=args.n_head,
        output_dim=args.output_dim,
        use_speaker_conditioning=args.use_speaker_conditioning,
        speaker_emb_dim=args.speaker_emb_dim,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    total_train_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    scheduler = make_lr_scheduler(optimizer, args.warmup_steps, total_train_steps)

    model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, scheduler
    )

    titanet_model = None
    if args.use_speaker_conditioning:
        titanet_model = load_titanet(accelerator.device)
        logger.info("Loaded frozen TitaNet-Large for on-the-fly speaker embeddings")

    speaker_loss_weight = args.speaker_loss_weight if args.use_speaker_conditioning else 0.0

    global_step = 0
    if args.resume is not None:
        ckpt_path = find_latest_checkpoint(args.exp_dir) if args.resume == "latest" else args.resume
        if ckpt_path is None:
            logger.warning("No checkpoint found, starting from scratch.")
        else:
            global_step = load_checkpoint(accelerator, model, optimizer, scheduler, ckpt_path)

    logger.info(f"Train samples: {len(train_ds):,}")
    logger.info(f"Valid samples: {len(valid_ds):,}")
    logger.info(f"Training from step {global_step} to {total_train_steps}")

    model.train()
    running = {}
    running_count = 0

    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch}")

        for batch in train_loader:
            with accelerator.accumulate(model):
                if titanet_model is not None and "speaker_audio" in batch:
                    batch["speaker_embs"] = extract_speaker_embs_from_batch(
                        titanet_model, batch, accelerator.device
                    )

                loss, stats = compute_batch(
                    model=model,
                    batch=batch,
                    loss_type=args.loss_type,
                    n_frame_per_class=args.n_frame_per_class,
                    text_only_loss_weight=args.text_only_loss_weight,
                    speaker_loss_weight=speaker_loss_weight,
                    codec_rate_hz=args.codec_rate_hz,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            stats["lr"] = torch.tensor(scheduler.get_last_lr()[0], device=accelerator.device)

            for k, v in stats.items():
                gathered = accelerator.gather_for_metrics(v.detach().reshape(1))
                running[k] = running.get(k, 0.0) + gathered.mean().item()
            running_count += 1

            if global_step % args.log_every == 0:
                payload = {f"train/{k}": v / running_count for k, v in running.items()}
                accelerator.log(payload, step=global_step)

                msg = (
                    f"step={global_step:>7d} | "
                    f"loss={payload['train/loss']:.4f} | "
                    f"prompt_sec_mae={payload['train/prompt_sec_mae']:.4f} | "
                    f"text_only_sec_mae={payload['train/text_only_sec_mae']:.4f}"
                )
                if args.use_speaker_conditioning:
                    msg += f" | speaker_sec_mae={payload['train/speaker_sec_mae']:.4f}"
                msg += f" | lr={payload['train/lr']:.2e}"
                if "train/duration_col_sec_mae" in payload:
                    msg += f" | duration_col_sec_mae={payload['train/duration_col_sec_mae']:.4f}"
                logger.info(msg)

                running = {}
                running_count = 0

            if global_step % args.save_every == 0:
                save_checkpoint(
                    accelerator, model, optimizer, scheduler,
                    global_step, args.exp_dir, args.keep_last_k,
                )
                run_validation(
                    model=model,
                    dataloader=valid_loader,
                    accelerator=accelerator,
                    loss_type=args.loss_type,
                    n_frame_per_class=args.n_frame_per_class,
                    text_only_loss_weight=args.text_only_loss_weight,
                    speaker_loss_weight=speaker_loss_weight,
                    codec_rate_hz=args.codec_rate_hz,
                    step=global_step,
                    titanet_model=titanet_model,
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    save_checkpoint(
        accelerator, model, optimizer, scheduler,
        global_step, args.exp_dir, args.keep_last_k,
    )
    run_validation(
        model=model,
        dataloader=valid_loader,
        accelerator=accelerator,
        loss_type=args.loss_type,
        n_frame_per_class=args.n_frame_per_class,
        text_only_loss_weight=args.text_only_loss_weight,
        speaker_loss_weight=speaker_loss_weight,
        codec_rate_hz=args.codec_rate_hz,
        step=global_step,
        titanet_model=titanet_model,
    )
    logger.info("done")


if __name__ == "__main__":
    main()