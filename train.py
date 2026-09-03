import argparse
import json
import logging
import math
import os
import shutil
import time
from copy import deepcopy
from glob import glob

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_cosine_schedule_with_warmup
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

from dataset import build_multi_tts_dataloader, build_tts_dataloader, extract_speaker_embs_from_batch, load_titanet
from discriminator import ConformerDiscirminator
from model_transformer import load_config, model_from_config

logging.getLogger("tilelang.cache.kernel_cache").setLevel(logging.ERROR)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(__name__)

FP8_SKIP_EXACT = {
    "in_proj",
    "out_proj",
    "cond_proj",
    "span_mask_proj",
    "valid_audio_proj",
}

FP8_SKIP_PREFIXES = (
    "cond_module.",
    "speaker_film_adapter.",
    "speaker_reg_head.",
    "reference_proj.",
)


def module_filter_fn(mod, fqn):
    if not isinstance(mod, nn.Linear):
        return False
    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
        return False
    if mod.in_features < 256 or mod.out_features < 256:
        return False
    if fqn in FP8_SKIP_EXACT or fqn.startswith(FP8_SKIP_PREFIXES):
        return False
    return True


def get_num_params(model):
    n = sum(p.numel() for p in model.parameters())
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def format_eta(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m" if h > 0 else f"{m}m"


def maybe_len(obj):
    try:
        return len(obj)
    except TypeError:
        return None


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def sample_stratified_logit_normal(batch_size, device, dtype):
    offset = torch.rand((), device=device)
    shift = torch.randint(0, batch_size, (), device=device)
    idx = (torch.arange(batch_size, device=device) + shift) % batch_size
    q = (idx.float() + offset) / batch_size
    z = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
    return torch.sigmoid(z).to(dtype)


def sample_masked_span(
    latent_mask,
    frac_min=0.7,
    frac_max=1.0,
    min_mask_len=1,
    keep_at_least_one_unmasked=False,
):
    latent_mask = latent_mask.to(torch.bool)
    B, _ = latent_mask.shape
    device = latent_mask.device

    span_mask = torch.zeros_like(latent_mask, dtype=torch.bool)
    valid_lens = latent_mask.sum(dim=1).long()
    frac_lengths = torch.empty(B, device=device).uniform_(frac_min, frac_max)

    for i in range(B):
        L = int(valid_lens[i].item())
        if L <= 0:
            continue
        max_mask_len = max(L - 1, 0) if keep_at_least_one_unmasked else L
        if max_mask_len <= 0:
            continue
        mask_len = max(min(int(frac_lengths[i].item() * L), max_mask_len), min_mask_len)
        start = torch.randint(0, L - mask_len + 1, (1,), device=device).item()
        span_mask[i, start: start + mask_len] = True

    return span_mask & latent_mask


def extract_disc_features(
    model,
    latents,
    latent_lengths,
    text_ids,
    text_mask,
    detach_generator,
    speaker_emb=None,
    disc_t=None,
    disc_noise=None,
):

    max_len = int(latent_lengths.max().item())
    latents = latents[:, :max_len, :]
    valid_mask = torch.arange(max_len, device=latents.device)[None, :] < latent_lengths[:, None]
    B = latents.size(0)

    if disc_t is not None and disc_noise is not None:
        disc_noise = disc_noise[:, :max_len, :]
        t_exp = disc_t[:, None, None]
        noisy_latents = (1.0 - t_exp) * disc_noise + t_exp * latents
        noisy_latents = noisy_latents * valid_mask.unsqueeze(-1).to(noisy_latents.dtype)
        t_for_model = disc_t
    else:
        noisy_latents = latents
        t_for_model = torch.ones(B, device=latents.device, dtype=latents.dtype)

    kwargs = dict(
        x=noisy_latents,
        cond=noisy_latents,
        t=t_for_model,
        text_ids=text_ids,
        text_mask=text_mask,
        reference_latent=None,
        reference_mask=None,
        text_cond_drop=torch.zeros(B, device=latents.device, dtype=torch.bool),
        speaker_cond_drop=None,
        use_checkpoint=False,
        latent_mask=valid_mask,
        valid_audio_mask=valid_mask,
        span_mask=torch.zeros_like(valid_mask),
        return_hidden_states=True,
        speaker_emb=speaker_emb,
    )

    if detach_generator:
        with torch.no_grad():
            _, gen_layers = model(**kwargs)
        return [x.detach() for x in gen_layers]

    _, gen_layers = model(**kwargs)
    return gen_layers


def compute_discriminator_loss(model, discriminator, cache):
    B = cache["fake_latents"].size(0)
    device = cache["fake_latents"].device
    dtype = cache["fake_latents"].dtype

    disc_t = torch.rand(B, device=device, dtype=dtype)
    disc_noise = torch.randn_like(cache["fake_latents"])

    shared = dict(
        model=model,
        latent_lengths=cache["latent_lengths"],
        text_ids=cache["text_ids"],
        text_mask=cache["text_mask"],
        detach_generator=True,
        speaker_emb=cache.get("speaker_embs"),
        disc_t=disc_t,
        disc_noise=disc_noise,
    )

    fake_feats = extract_disc_features(latents=cache["fake_latents"], **shared)
    real_feats = extract_disc_features(latents=cache["real_latents"], **shared)

    fake_score = discriminator(fake_feats, cache["latent_lengths"])
    real_score = discriminator(real_feats, cache["latent_lengths"])

    return (fake_score ** 2).mean() + ((1.0 - real_score) ** 2).mean()


def compute_loss(
    model,
    batch,
    use_checkpoint,
    *,
    stage2=False,
    use_speaker=False,
    use_adversarial=False,
    discriminator=None,
    adv_lambda=0.0,
    speaker_aux_weight=0.1,
):
    latents = batch["latents"]
    valid_audio_mask = batch["latent_mask"].to(torch.bool)
    text_ids = batch["text_ids"]
    text_mask = batch["text_mask"].to(torch.bool)

    B = latents.size(0)
    device = latents.device
    dtype = latents.dtype

    x1 = latents * valid_audio_mask.unsqueeze(-1).to(latents.dtype)
    span_mask = sample_masked_span(valid_audio_mask, frac_min=0.7, frac_max=1.0)

    x0 = torch.randn_like(x1)
    t = sample_stratified_logit_normal(B, device=device, dtype=dtype)
    t_exp = t[:, None, None]

    x_t = (1.0 - t_exp) * x0 + t_exp * x1
    v_target = x1 - x0

    cond = torch.where(span_mask.unsqueeze(-1), torch.zeros_like(x1), x1)
    cond = cond * valid_audio_mask.unsqueeze(-1).to(cond.dtype)
    model_input = x_t * valid_audio_mask.unsqueeze(-1).to(x_t.dtype)


    p_drop_both = 0.1
    p_drop_text = 0.1
    p_drop_speaker = 0.0  # legacy, don't mind it 

    r = torch.rand(B, device=device)
    b0 = p_drop_both
    b1 = b0 + p_drop_text
    b2 = b1 + p_drop_speaker

    drop_both_bucket = r < b0
    drop_text_bucket = (r >= b0) & (r < b1)
    drop_speaker_bucket = (r >= b1) & (r < b2)

    text_cond_drop = drop_both_bucket | drop_text_bucket

    speaker_emb = batch.get("speaker_embs", None)
    if speaker_emb is not None:
        speaker_emb = speaker_emb.to(device)

    speaker_cond_drop = None
    if stage2 and use_speaker and speaker_emb is not None:
        speaker_cond_drop = drop_both_bucket | drop_speaker_bucket

    need_hidden = stage2 and use_speaker and speaker_emb is not None
    out = model(
        x=model_input,
        cond=cond,
        t=t,
        text_ids=text_ids,
        text_mask=text_mask,
        reference_latent=None,
        reference_mask=None,
        text_cond_drop=text_cond_drop,
        speaker_cond_drop=speaker_cond_drop,
        use_checkpoint=use_checkpoint,
        latent_mask=valid_audio_mask,
        valid_audio_mask=valid_audio_mask,
        span_mask=span_mask,
        speaker_emb=speaker_emb if (stage2 and use_speaker) else None,
        return_hidden_states=need_hidden,
    )
    v_pred, hidden_states = out if need_hidden else (out, None)

    per_pos_loss = F.mse_loss(v_pred.float(), v_target.float(), reduction="none").mean(dim=-1)
    masked_loss = per_pos_loss[span_mask]
    fm_loss = masked_loss.mean() if masked_loss.numel() > 0 else per_pos_loss.new_zeros(())

    zero = fm_loss.new_zeros(())
    if not stage2:
        stats = {"fm_loss": fm_loss.detach(), "gen_adv_loss": zero, "speaker_aux_loss": zero}
        return fm_loss, stats, None

    
    raw = unwrap_model(model)
    speaker_aux_loss = zero
    if need_hidden and hasattr(raw, "predict_speaker"):
        spk_pred = raw.predict_speaker(hidden_states[-1], valid_audio_mask)
        if spk_pred is not None:
            valid_spk = speaker_emb.abs().sum(dim=-1) > 0
            if speaker_cond_drop is not None:
                valid_spk = valid_spk & ~speaker_cond_drop.to(torch.bool)
            if valid_spk.any():
                speaker_aux_loss = F.mse_loss(spk_pred[valid_spk], speaker_emb[valid_spk])

    total_loss = fm_loss + speaker_aux_weight * speaker_aux_loss

    # --- adversarial ---
    gen_adv_loss = zero
    disc_cache = None
    if use_adversarial and discriminator is not None and adv_lambda > 0.0:
        latent_lengths = valid_audio_mask.sum(dim=1).long()

        # single-step fake generation
        x0_gen = torch.randn_like(x1)
        t_gen = torch.rand(B, device=device, dtype=dtype)
        t_gen_exp = t_gen[:, None, None]
        x_t_gen = (1.0 - t_gen_exp) * x0_gen + t_gen_exp * x1

        v_gen = model(
            x=x_t_gen * valid_audio_mask.unsqueeze(-1).to(x_t_gen.dtype),
            cond=torch.zeros_like(x1),
            t=t_gen,
            text_ids=text_ids,
            text_mask=text_mask,
            reference_latent=None,
            reference_mask=None,
            text_cond_drop=torch.zeros(B, device=device, dtype=torch.bool),
            use_checkpoint=use_checkpoint,
            latent_mask=valid_audio_mask,
            valid_audio_mask=valid_audio_mask,
            span_mask=valid_audio_mask,
            speaker_emb=speaker_emb if use_speaker else None,
        )
        x1_hat = (x0_gen + v_gen) * valid_audio_mask.unsqueeze(-1).to(v_gen.dtype)

        fake_feats = extract_disc_features(
            model=model,
            latents=x1_hat,
            latent_lengths=latent_lengths,
            text_ids=text_ids,
            text_mask=text_mask,
            detach_generator=False,
            speaker_emb=speaker_emb if use_speaker else None,
            disc_t=torch.rand(B, device=device, dtype=dtype),
            disc_noise=torch.randn_like(x1),
        )
        gen_adv_loss = ((1.0 - discriminator(fake_feats, latent_lengths)) ** 2).mean()

        disc_cache = {
            "fake_latents": x1_hat.detach(),
            "real_latents": x1.detach(),
            "latent_lengths": latent_lengths.detach(),
            "text_ids": text_ids.detach(),
            "text_mask": text_mask.detach(),
            "speaker_embs": speaker_emb.detach() if speaker_emb is not None else None,
        }

    total_loss = total_loss + adv_lambda * gen_adv_loss

    stats = {
        "fm_loss": fm_loss.detach(),
        "gen_adv_loss": gen_adv_loss.detach(),
        "speaker_aux_loss": speaker_aux_loss.detach(),
    }
    return total_loss, stats, disc_cache


@torch.no_grad()
def update_ema(ema_model, model, decay):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    ema_bufs = dict(ema_model.named_buffers())
    for name, buf in model.named_buffers():
        if name in ema_bufs:
            ema_bufs[name].copy_(buf)


def save_checkpoint(accelerator, model, ema, step, exp_dir, tag):
    ckpt_dir = os.path.join(exp_dir, "checkpoints", tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    accelerator.save_state(ckpt_dir)

    if accelerator.is_main_process:
        torch.save(unwrap_model(model).state_dict(), os.path.join(ckpt_dir, "model_state.pt"))
        torch.save({"step": step, "ema": ema.state_dict()},
                   os.path.join(ckpt_dir, "extra_state.pt"))

    accelerator.wait_for_everyone()
    logger.info(f"Saved checkpoint: {tag}")


def load_checkpoint(accelerator, model, ema, exp_dir, tag, drop_optimizer_state=False):
    ckpt_dir = os.path.join(exp_dir, "checkpoints", tag)

    if drop_optimizer_state:
        logger.info("drop_optimizer_state=True -> skipping optimizer/scheduler/RNG restore")
    else:
        accelerator.load_state(ckpt_dir)

    model_state_path = os.path.join(ckpt_dir, "model_state.pt")
    if os.path.exists(model_state_path):
        state = torch.load(model_state_path, map_location="cpu", weights_only=True)
        unwrap_model(model).load_state_dict(state, strict=False)
        logger.info("Loaded model state_dict from model_state.pt (strict=False)")

    extra = torch.load(os.path.join(ckpt_dir, "extra_state.pt"),
                       map_location="cpu", weights_only=True)

    # strict=False so a stage1 -> stage2 EMA shape change doesn't hard-fail
    ema.load_state_dict(extra["ema"], strict=False)
    ema.to(accelerator.device)

    if drop_optimizer_state:
        logger.info(f"Loaded weights from {tag} | optimizer dropped | step reset to 0")
        return 0

    logger.info(f"Resumed from {tag} (step {extra['step']})")
    return extra["step"]


def find_latest_checkpoint(exp_dir):
    dirs = sorted(glob(os.path.join(exp_dir, "checkpoints", "step_*")))
    return os.path.basename(dirs[-1]) if dirs else None


def cleanup_checkpoints(exp_dir, keep_last_k):
    dirs = sorted(glob(os.path.join(exp_dir, "checkpoints", "step_*")))
    for d in dirs[:-keep_last_k]:
        shutil.rmtree(d, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset", type=str)
    dataset_group.add_argument("--datasets", type=str)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--drop_optimizer_state", action="store_true")

    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--use_speaker", action="store_true")
    parser.add_argument("--speaker_aux_weight", type=float, default=0.1)
    parser.add_argument("--speaker_drop_prob", type=float, default=0.1)

    parser.add_argument("--use_adversarial", action="store_true")
    parser.add_argument("--adv_lambda", type=float, default=1.0)
    parser.add_argument("--disc_lr", type=float, default=5e-5)
    parser.add_argument("--disc_channels", type=int, default=512)
    parser.add_argument("--disc_num_layers", type=int, default=3)
    parser.add_argument("--disc_num_heads", type=int, default=8)
    parser.add_argument("--disc_kernel", type=int, default=15)
    return parser.parse_args()


def build_optimizer(args, model, train_cfg):
    if not args.stage2:
        return bnb.optim.AdamW8bit(
            model.parameters(),
            lr=train_cfg["lr"],
            betas=tuple(train_cfg["betas"]),
            weight_decay=train_cfg["weight_decay"],
        )

    base_lr, speaker_lr = 2e-5, 1e-4
    raw = unwrap_model(model)
    speaker_params, speaker_param_ids = [], set()

    if args.use_speaker:
        for module in (raw.speaker_film_adapter, raw.speaker_reg_head):
            if module is None:
                continue
            for p in module.parameters():
                if p.requires_grad:
                    speaker_params.append(p)
                    speaker_param_ids.add(id(p))

    base_params = [p for p in model.parameters()
                   if p.requires_grad and id(p) not in speaker_param_ids]

    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr,
                             "weight_decay": train_cfg["weight_decay"]})
    if speaker_params:
        param_groups.append({"params": speaker_params, "lr": speaker_lr,
                             "weight_decay": train_cfg["weight_decay"]})

    logger.info(f"Optimizer: base params={len(base_params)} @ lr={base_lr:.0e}, "
                f"speaker params={len(speaker_params)} @ lr={speaker_lr:.0e}")

    return bnb.optim.AdamW8bit(
        param_groups,
        betas=tuple(train_cfg["betas"]),
        weight_decay=train_cfg["weight_decay"],
    )


def build_data(args, cfg, need_audio):
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    raw_max = data_cfg["max_audio_seconds"] * data_cfg["codec_rate_hz"]
    common = dict(
        latent_dim=data_cfg["latent_dim"],
        need_audio=need_audio,
        max_text_len=data_cfg["max_text_length"],
        max_latent_len=int(data_cfg.get("max_latent_length", int(raw_max))),
        text_pad_id=data_cfg["text_pad_id"],
        train_cfg=train_cfg,
        tokenizer=(
            args.tokenizer
            or data_cfg.get("tokenizer")
            or data_cfg.get("tokenizer_name")
            or data_cfg.get("text_tokenizer")
        ),
    )

    if args.datasets is None:
        logger.info(f"Loading dataset from {args.dataset}")
        return build_tts_dataloader(
            dataset_path=args.dataset,
            streaming=args.streaming,
            dataset_split=args.dataset_split,
            **common,
        )

    if args.streaming:
        raise ValueError("--streaming is not supported with --datasets")

    with open(args.datasets, "r", encoding="utf-8") as f:
        dataset_configs = json.load(f).get("datasets")
    if not isinstance(dataset_configs, list) or not dataset_configs:
        raise ValueError("--datasets JSON must contain a non-empty top-level 'datasets' list")

    logger.info(f"Loading {len(dataset_configs)} configured datasets from {args.datasets}")
    return build_multi_tts_dataloader(dataset_configs=dataset_configs, **common)


def main():
    args = parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["grad_accum_steps"],
        log_with=None,
    )

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

    if accelerator.is_main_process:
        shutil.copy2(args.config, os.path.join(args.exp_dir, "config.json"))

    set_seed(train_cfg["seed"])

    logger.info(f"Experiment directory: {args.exp_dir}")
    logger.info(f"Stage: {'2' if args.stage2 else '1 (vanilla)'}"
                f"{' + speaker conditioning' if args.use_speaker else ''}"
                f"{' + adversarial' if args.use_adversarial else ''}")

    model = model_from_config(
        cfg,
        use_speaker_conditioning=args.use_speaker,
        speaker_emb_dim=192,  # TitaNet-Large
    )
    logger.info(f"Model parameters: {get_num_params(model)}")

    ema = deepcopy(model)
    ema.requires_grad_(False)
    ema.eval()

    convert_to_float8_training(
        model,
        config=Float8LinearConfig.from_recipe_name("tensorwise"),
        module_filter_fn=module_filter_fn,
    )

    if train_cfg.get("compile", False):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    optimizer = build_optimizer(args, model, train_cfg)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_cfg["warmup_steps"],
        num_training_steps=train_cfg["max_steps"],
    )

    titanet_model = None
    if args.stage2 and args.use_speaker:
        titanet_model = load_titanet(accelerator.device)
        logger.info("Loaded frozen TitaNet-Large for on-the-fly speaker embeddings")

    dataset, dataloader = build_data(args, cfg, need_audio=args.stage2 and args.use_speaker)
    n_samples = maybe_len(dataset)
    logger.info(f"Dataset size: {n_samples:,} samples" if n_samples else "Dataset size: streaming")

    adversarial = args.stage2 and args.use_adversarial
    discriminator = None
    disc_optimizer = None

    if adversarial:
        base_model = unwrap_model(model)
        discriminator = ConformerDiscirminator(
            input_dim=base_model.num_layers * base_model.model_size,
            channels=args.disc_channels,
            num_layers=args.disc_num_layers,
            num_heads=args.disc_num_heads,
            depthwise_conv_kernel_size=args.disc_kernel,
            use_group_norm=True,
        )
        disc_optimizer = bnb.optim.AdamW8bit(
            discriminator.parameters(),
            lr=args.disc_lr,
            betas=(0.9, 0.95),
            weight_decay=train_cfg["weight_decay"],
        )

        model, optimizer, dataloader, scheduler, discriminator, disc_optimizer = accelerator.prepare(
            model, optimizer, dataloader, scheduler, discriminator, disc_optimizer
        )
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )

    ema = ema.to(accelerator.device)

    global_step = 0
    if args.resume is None:
        update_ema(ema, unwrap_model(model), decay=0)
    else:
        tag = find_latest_checkpoint(args.exp_dir) if args.resume == "latest" else args.resume
        if tag is None:
            logger.warning("No checkpoint found, training from scratch")
        else:
            global_step = load_checkpoint(
                accelerator, model, ema, args.exp_dir, tag,
                drop_optimizer_state=args.drop_optimizer_state,
            )

    max_steps = train_cfg["max_steps"]
    grad_accum = train_cfg["grad_accum_steps"]
    log_every = train_cfg["log_every"]
    save_every = train_cfg["save_every"]
    keep_last_k = train_cfg["keep_last_k"]
    grad_clip = train_cfg["grad_clip"]
    ema_decay = train_cfg["ema_decay"]
    use_ckpt = train_cfg["use_checkpoint"]

    num_batches = maybe_len(dataloader)
    steps_per_epoch = (num_batches // grad_accum) if num_batches else max(max_steps - global_step, 1)
    steps_per_epoch = max(steps_per_epoch, 1)

    logger.info(f"Training from step {global_step} to {max_steps}")
    logger.info(f"Batch size per device: {train_cfg['batch_size']}")
    logger.info(f"Gradient accumulation: {grad_accum}")
    logger.info(
        f"Effective batch size: "
        f"{train_cfg['batch_size'] * grad_accum * accelerator.num_processes}"
    )
    if args.stage2:
        logger.info(
            f"Stage 2 | use_speaker={args.use_speaker} | adversarial={args.use_adversarial} | "
            f"adv_lambda={args.adv_lambda if args.use_adversarial else 0.0} | "
            f"speaker_aux_weight={args.speaker_aux_weight} | "
            f"speaker_drop_prob={args.speaker_drop_prob}"
        )

    model.train()
    if adversarial:
        discriminator.train()

    running = {"loss": 0.0, "fm": 0.0, "adv": 0.0, "spk_aux": 0.0, "disc": 0.0}
    log_steps = 0
    start_time = time.time()
    epoch = global_step // steps_per_epoch

    while global_step < max_steps:
        epoch_step = 0
        logger.info(f"Starting epoch {epoch}  ({steps_per_epoch} steps)")

        for batch in dataloader:
            accumulate_ctx = (
                accelerator.accumulate(model, discriminator)
                if adversarial
                else accelerator.accumulate(model)
            )

            with accumulate_ctx:
                if titanet_model is not None:
                    batch["speaker_embs"] = extract_speaker_embs_from_batch(
                        titanet_model, batch, accelerator.device,
                    )

                if adversarial:
                    set_requires_grad(discriminator, False)

                with accelerator.autocast():
                    loss, stats, disc_cache = compute_loss(
                        model=model,
                        batch=batch,
                        use_checkpoint=use_ckpt,
                        stage2=args.stage2,
                        use_speaker=args.use_speaker,
                        use_adversarial=adversarial,
                        discriminator=discriminator,
                        adv_lambda=args.adv_lambda if adversarial else 0.0,
                        speaker_aux_weight=args.speaker_aux_weight,
                    )

                accelerator.backward(loss)

                if adversarial:
                    set_requires_grad(discriminator, True)
                    disc_loss = compute_discriminator_loss(
                        model=model, discriminator=discriminator, cache=disc_cache,
                    )
                    accelerator.backward(disc_loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                        accelerator.clip_grad_norm_(discriminator.parameters(), grad_clip)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    disc_optimizer.step()
                    disc_optimizer.zero_grad()
                else:
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    disc_loss = loss.new_zeros(())

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            epoch_step += 1
            update_ema(ema, unwrap_model(model), ema_decay)

            running["loss"] += loss.detach().item()
            running["fm"] += stats["fm_loss"].item()
            running["adv"] += stats["gen_adv_loss"].item()
            running["spk_aux"] += stats["speaker_aux_loss"].item()
            running["disc"] += disc_loss.detach().item()
            log_steps += 1

            if global_step % log_every == 0:
                n = log_steps
                elapsed = time.time() - start_time
                sps = n / elapsed
                speed = f"{sps:.2f} it/s" if sps >= 1.0 else f"{elapsed / n:.2f} s/it"
                head = (
                    f"step={global_step:>7d} | "
                    f"epoch={epoch} [{epoch_step}/{steps_per_epoch}] | "
                    f"loss={running['loss'] / n:.4f}"
                )
                if args.stage2:
                    head += (
                        f" | fm={running['fm'] / n:.4f}"
                        f" | spk_aux={running['spk_aux'] / n:.4f}"
                        f" | g_adv={running['adv'] / n:.4f}"
                        f" | d={running['disc'] / n:.4f}"
                    )
                logger.info(
                    f"{head} | lr={scheduler.get_last_lr()[0]:.2e} | {speed} | "
                    f"mem={torch.cuda.max_memory_allocated() / 1e9:.1f}GB | "
                    f"eta={format_eta((max_steps - global_step) / sps)}"
                )

                running = {k: 0.0 for k in running}
                log_steps = 0
                start_time = time.time()

            if global_step % save_every == 0:
                save_checkpoint(accelerator, model, ema, global_step,
                                args.exp_dir, f"step_{global_step:07d}")
                if accelerator.is_main_process:
                    cleanup_checkpoints(args.exp_dir, keep_last_k)

            if global_step >= max_steps:
                break

        epoch += 1

    save_checkpoint(accelerator, model, ema, global_step,
                    args.exp_dir, f"step_{global_step:07d}")
    logger.info("Training complete.")


if __name__ == "__main__":
    main()