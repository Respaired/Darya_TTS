import hashlib
import io
import logging
import os
import pickle
from glob import glob

import numpy as np
import torch
from datasets import Audio, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_tensor(src):
    if isinstance(src, torch.Tensor):
        return src
    if isinstance(src, np.ndarray):
        return torch.from_numpy(src)
    if isinstance(src, (list, tuple)):
        return torch.tensor(src)

    if isinstance(src, (bytes, bytearray)):
        x = torch.load(io.BytesIO(src), map_location="cpu", weights_only=True)
    else:
        x = torch.load(src, map_location="cpu", weights_only=True)

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, (list, tuple)):
        x = torch.tensor(x)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor in {src}, got {type(x)}")
    return x


def get_first_split(ds):
    if isinstance(ds, DatasetDict):
        for name in ["train", "validation", "valid", "dev", "test"]:
            if name in ds:
                return ds[name]
        raise ValueError(f"No usable split. Available: {list(ds.keys())}")
    return ds


def _load_tokenizer(tokenizer):
    if tokenizer is None:
        return None
    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer, add_bos_token=True, add_eos_token=True)
    return tokenizer


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


class _TTSDataMixin:
    def _tokenize(self, text):
        if self.tokenizer is None:
            raise RuntimeError("requires a tokenizer for on-the-fly text.")
        enc = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return enc["input_ids"][0], enc["attention_mask"][0].to(torch.bool)

    def _fix_latent_shape(self, latent):
        latent = latent.float()
        if latent.dim() == 1:
            latent = latent.unsqueeze(-1)
        if latent.shape[-1] != self.latent_dim and latent.shape[0] == self.latent_dim:
            latent = latent.T
        if latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent dim mismatch: expected {self.latent_dim}, got {tuple(latent.shape)}"
            )
        return latent

    def _load_latent(self, row):
        return self._fix_latent_shape(load_tensor(row["latents"]))

    @staticmethod
    def _merged_keys(row):
        merged = row.get("merged", "❌")
        if merged is None or merged == "❌":
            return None
        if not isinstance(merged, str):
            merged = str(merged)
        keys = [key for key in merged.split("➕") if key]
        return keys or None

    def _is_overlength(self, latent):
        return self.max_speech_frames is not None and int(latent.shape[0]) > self.max_speech_frames

    def _audio_ref(self, row):
        return row.get("filename", row.get("audio", None))


class TTSDataset(_TTSDataMixin, Dataset):
    MAX_SKIP_ATTEMPTS = 100

    def __init__(
        self,
        hf_dataset,
        latent_dim,
        need_audio_path=False,
        tokenizer=None,
        max_text_len=None,
        max_speech_frames=None,
        key_index_cache_dir=None,
    ):
        self.ds = hf_dataset
        self.latent_dim = latent_dim
        self.need_audio_path = need_audio_path
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_speech_frames = None if max_speech_frames is None else int(max_speech_frames)
        self.key_index_cache_dir = os.path.expanduser(
            key_index_cache_dir
            or os.environ.get("ECHO_TTS_KEY_INDEX_CACHE", "~/.cache/echo_tts/key_indices")
        )

        columns = getattr(hf_dataset, "column_names", None) or []
        self.key_to_index = (
            self._build_key_index() if ("merged" in columns and "key" in columns) else {}
        )

    def _key_index_cache_path(self):
        fingerprint = getattr(self.ds, "_fingerprint", None)
        if not fingerprint:
            return None
        signature = f"hf:{fingerprint}:rows={len(self.ds)}:column=key"
        digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:24]
        return os.path.join(self.key_index_cache_dir, f"metadata_{digest}.pkl")

    def _build_key_index(self):
        cache_path = self._key_index_cache_path()

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    payload = pickle.load(f)
            except Exception as exc:
                logger.warning("Ignoring unreadable key-index cache %s (%s)", cache_path, exc)
                payload = None
            if isinstance(payload, dict) and payload.get("num_rows") == len(self.ds):
                index = payload.get("key_to_index")
                if isinstance(index, dict):
                    logger.info("Loaded cached key index | entries=%s", f"{len(index):,}")
                    return index

        logger.info("Building key index | rows=%s", f"{len(self.ds):,}")
        index = {key: i for i, key in enumerate(self.ds["key"])}

        if cache_path:
            os.makedirs(self.key_index_cache_dir, exist_ok=True)
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            with open(tmp_path, "wb") as f:
                pickle.dump(
                    {"num_rows": len(self.ds), "key_to_index": index},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(tmp_path, cache_path)
            logger.info("Saved key index cache: %s", cache_path)

        return index

    def _load_merged_latent(self, row):
        keys = self._merged_keys(row)
        if not keys:
            return self._load_latent(row)
        if not self.key_to_index:
            raise ValueError("merged rows require a 'key' column in the dataset")

        latents = []
        for key in keys:
            if key not in self.key_to_index:
                raise KeyError(f"merged key not found in dataset: {key}")
            latents.append(self._load_latent(self.ds[self.key_to_index[key]]))
        return torch.concatenate(latents)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        n = len(self.ds)
        if n == 0:
            raise IndexError("cannot fetch from an empty dataset")

        last_error = None
        for offset in range(min(n, self.MAX_SKIP_ATTEMPTS)):
            current_idx = (idx + offset) % n
            row = self.ds[current_idx]

            try:
                latent = self._load_merged_latent(row)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Skipping invalid latent | index=%s | %s: %s",
                    current_idx, type(exc).__name__, exc,
                )
                continue

            if self._is_overlength(latent):
                logger.warning(
                    "Skipping overlength latent | index=%s | frames=%s | max=%s",
                    current_idx, int(latent.shape[0]), self.max_speech_frames,
                )
                continue

            text = row["text"]
            input_ids, attention_mask = self._tokenize(text)
            audio_path = self._audio_ref(row) if self.need_audio_path else None
            return input_ids, attention_mask, latent, text, audio_path

        raise RuntimeError(
            f"no valid sample within {self.MAX_SKIP_ATTEMPTS} rows of index {idx}"
        ) from last_error


class StreamingTTSDataset(_TTSDataMixin, IterableDataset):
    def __init__(
        self,
        hf_dataset,
        latent_dim,
        need_audio_path=False,
        tokenizer=None,
        max_text_len=None,
        max_speech_frames=None,
    ):
        self.ds = hf_dataset
        self.latent_dim = latent_dim
        self.need_audio_path = need_audio_path
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_speech_frames = None if max_speech_frames is None else int(max_speech_frames)

    def _row_to_item(self, row, key_to_latent):
        keys = self._merged_keys(row)
        if keys:
            missing = [key for key in keys if key not in key_to_latent]
            if missing:
                raise KeyError(f"merged key not found in streamed dataset cache: {missing}")
            latent = torch.concatenate([key_to_latent[key] for key in keys])
        else:
            latent = self._load_latent(row)

        if self._is_overlength(latent):
            logger.warning(
                "Skipping overlength latent | key=%s | frames=%s | max=%s",
                row.get("key", None), int(latent.shape[0]), self.max_speech_frames,
            )
            return None

        text = row["text"]
        input_ids, attention_mask = self._tokenize(text)
        audio_path = self._audio_ref(row) if self.need_audio_path else None
        return input_ids, attention_mask, latent, text, audio_path

    def _can_emit(self, row, key_to_latent):
        keys = self._merged_keys(row)
        if not keys:
            return True
        return all(key in key_to_latent for key in keys)

    def __iter__(self):
        key_to_latent = {}
        pending = []
        merges_possible = None

        for row in self.ds:
            if merges_possible is None:
                merges_possible = "merged" in row

            # only merged datasets need the by-key latent cache; caching
            # unconditionally would hold the whole stream in memory
            if merges_possible:
                key = row.get("key", None)
                if key is not None:
                    key_to_latent[key] = self._load_latent(row)

            if self._can_emit(row, key_to_latent):
                item = self._row_to_item(row, key_to_latent)
                if item is not None:
                    yield item
            else:
                pending.append(row)
                continue

            if pending:
                still_pending = []
                for pending_row in pending:
                    if self._can_emit(pending_row, key_to_latent):
                        item = self._row_to_item(pending_row, key_to_latent)
                        if item is not None:
                            yield item
                    else:
                        still_pending.append(pending_row)
                pending = still_pending

        if pending:
            missing = {
                pending_row.get("key", "<unknown>"): [
                    key for key in (self._merged_keys(pending_row) or [])
                    if key not in key_to_latent
                ]
                for pending_row in pending
            }
            raise KeyError(f"merged keys not found in streamed dataset cache: {missing}")


class TTSCollator:
    MIN_AUDIO_SAMPLES = 16000

    def __init__(
        self,
        max_text_len,
        max_latent_len,
        text_pad_id,
        latent_dim,
        load_audio=False,
    ):
        self.max_text_len = max_text_len
        self.max_latent_len = int(max_latent_len)
        self.text_pad_id = text_pad_id
        self.latent_dim = latent_dim
        self.load_audio = load_audio

    def _load_audio_ref(self, audio_path):
        import librosa

        if audio_path is None:
            return None

        if isinstance(audio_path, dict):
            if audio_path.get("array", None) is not None:
                wav = np.asarray(audio_path["array"], dtype=np.float32)
                sr = audio_path.get("sampling_rate", 16000)
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                return wav
            if audio_path.get("bytes", None) is not None:
                wav, _ = librosa.load(io.BytesIO(audio_path["bytes"]), sr=16000)
                return wav
            audio_path = audio_path.get("path", None)

        if isinstance(audio_path, (bytes, bytearray)):
            wav, _ = librosa.load(io.BytesIO(audio_path), sr=16000)
            return wav
        if isinstance(audio_path, os.PathLike):
            audio_path = os.fspath(audio_path)
        if isinstance(audio_path, str):
            wav, _ = librosa.load(audio_path, sr=16000)
            return wav
        return None

    def _safe_load_audio(self, audio_path):
        if audio_path is None:
            return None
        # a corrupt clip drops to a zero placeholder rather than killing the run
        try:
            wav = self._load_audio_ref(audio_path)
        except Exception:
            return None
        if wav is None or len(wav) < self.MIN_AUDIO_SAMPLES:
            return None
        return wav

    def _collate_audio(self, audio_paths, B):
        wavs = [self._safe_load_audio(p) for p in audio_paths]
        lengths = [len(w) if w is not None else 0 for w in wavs]

        audio_batch = np.zeros((B, max(lengths) or 1), dtype=np.float32)
        for i, wav in enumerate(wavs):
            if wav is not None:
                audio_batch[i, : len(wav)] = wav

        return {
            "speaker_audio": torch.from_numpy(audio_batch),
            "speaker_audio_lengths": torch.tensor(lengths, dtype=torch.long),
            "speaker_audio_valid": torch.tensor([w is not None for w in wavs], dtype=torch.bool),
        }

    def __call__(self, batch):
        text_list, attn_list, latent_list, raw_texts, audio_paths = zip(*batch)
        B = len(text_list)

        text_ids = torch.full((B, self.max_text_len), self.text_pad_id, dtype=torch.long)
        text_mask = torch.zeros(B, self.max_text_len, dtype=torch.bool)
        for i, (ids, attn) in enumerate(zip(text_list, attn_list)):
            L = min(len(ids), self.max_text_len)
            text_ids[i, :L] = ids[:L]
            text_mask[i, :L] = attn[:L]

        # padding past each sample's true length is masked out of the loss
        speech_lengths = [int(lat.shape[0]) for lat in latent_list]
        batch_pad_len = max(speech_lengths)
        if batch_pad_len > self.max_latent_len:
            raise ValueError(
                f"speech longer than max_latent_len: "
                f"frames={batch_pad_len}, max_latent_len={self.max_latent_len}"
            )

        latents = torch.zeros(B, batch_pad_len, self.latent_dim, dtype=torch.float32)
        latent_mask = torch.zeros(B, batch_pad_len, dtype=torch.bool)
        for i, (lat, speech_len) in enumerate(zip(latent_list, speech_lengths)):
            latents[i, :speech_len] = lat.float()
            latent_mask[i, :speech_len] = True

        out = {
            "text_ids": text_ids,
            "text_mask": text_mask,
            "texts": list(raw_texts),
            "latents": latents,
            "latent_mask": latent_mask,
            "speech_lengths": torch.tensor(speech_lengths, dtype=torch.long),
        }

        if self.load_audio:
            out.update(self._collate_audio(audio_paths, B))

        return out


def _s3_storage_options(train_cfg):
    options = {
        "default_block_size": train_cfg.get("s3_block_size", 32 * 1024 * 1024),
        "default_cache_type": "readahead",
    }
    for env_name, opt_name in [
        ("AWS_ACCESS_KEY_ID", "key"),
        ("AWS_SECRET_ACCESS_KEY", "secret"),
        ("AWS_SESSION_TOKEN", "token"),
        ("AWS_DEFAULT_REGION", "client_kwargs"),
    ]:
        value = os.environ.get(env_name)
        if not value:
            continue
        if opt_name == "client_kwargs":
            options["client_kwargs"] = {"region_name": value}
        else:
            options[opt_name] = value
    return options


def _load_streaming_dataset(dataset_path, dataset_split, train_cfg):
    storage_options = _s3_storage_options(train_cfg)
    is_s3 = dataset_path.startswith("s3://")

    if not (is_s3 or os.path.isdir(dataset_path)):
        return load_dataset(dataset_path, split=dataset_split, streaming=True)

    if is_s3:
        import s3fs

        fs = s3fs.S3FileSystem(**{
            k: v for k, v in storage_options.items()
            if k in ("key", "secret", "token", "client_kwargs")
        })
        base = dataset_path.rstrip("/")[5:]
        parquet_files = sorted("s3://" + p for p in fs.glob(f"{base}/**/*.parquet"))
        arrow_files = sorted("s3://" + p for p in fs.glob(f"{base}/**/*.arrow"))
    else:
        parquet_files = sorted(glob(os.path.join(dataset_path, "**", "*.parquet"), recursive=True))
        arrow_files = sorted(glob(os.path.join(dataset_path, "**", "*.arrow"), recursive=True))

    if parquet_files:
        builder, files = "parquet", parquet_files
    elif arrow_files:
        builder, files = "arrow", arrow_files
    else:
        raise FileNotFoundError(f"No parquet or arrow shards found under {dataset_path}")

    return load_dataset(
        builder,
        data_files={dataset_split: files},
        split=dataset_split,
        streaming=True,
        storage_options=storage_options if is_s3 else None,
    )


def _is_saved_to_disk(path):
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "dataset_dict.json"))
        or os.path.exists(os.path.join(path, "state.json"))
    )


def _load_nonstreaming_dataset(dataset_path):
    mode = "load_from_disk" if _is_saved_to_disk(dataset_path) else "load_dataset"
    loader = load_from_disk if mode == "load_from_disk" else load_dataset
    ds = get_first_split(loader(dataset_path))
    logger.info(f"Dataset loaded: {dataset_path} | mode={mode} | rows={len(ds):,}")
    return ds


def build_tts_dataloader(
    dataset_path,
    latent_dim,
    need_audio,
    max_text_len,
    max_latent_len,
    text_pad_id,
    train_cfg,
    tokenizer=None,
    streaming=False,
    dataset_split="train",
):
    tokenizer = _load_tokenizer(tokenizer)
    max_speech_frames = int(max_latent_len)

    if streaming:
        hf_ds = _load_streaming_dataset(dataset_path, dataset_split, train_cfg)

        if need_audio and "filename" in (getattr(hf_ds, "column_names", None) or []):
            hf_ds = hf_ds.cast_column("filename", Audio(sampling_rate=16000))

        shuffle_buffer = train_cfg.get("streaming_shuffle_buffer", 10_000)
        if shuffle_buffer:
            hf_ds = hf_ds.shuffle(
                buffer_size=shuffle_buffer,
                seed=train_cfg.get("seed", 42),
            )

        dataset = StreamingTTSDataset(
            hf_ds,
            latent_dim=latent_dim,
            need_audio_path=need_audio,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            max_speech_frames=max_speech_frames,
        )
    else:
        dataset = TTSDataset(
            _load_nonstreaming_dataset(dataset_path),
            latent_dim=latent_dim,
            need_audio_path=need_audio,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            max_speech_frames=max_speech_frames,
            key_index_cache_dir=train_cfg.get("key_index_cache_dir"),
        )

    collator = TTSCollator(
        max_text_len=max_text_len,
        max_latent_len=max_latent_len,
        text_pad_id=text_pad_id,
        latent_dim=latent_dim,
        load_audio=need_audio,
    )

    num_workers = train_cfg.get("num_workers", 0 if streaming else 12)

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=not streaming,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator,
        persistent_workers=num_workers > 0,
        prefetch_factor=train_cfg.get("prefetch_factor", 4) if num_workers > 0 else None,
    )

    return dataset, dataloader