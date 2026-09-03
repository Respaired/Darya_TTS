import logging
import re

import torch

from duration_predictor.duration_model import SpeechLengthPredictor
from model_transformer import load_config, model_from_config


logger = logging.getLogger(__name__)


def _as_batch_list(x, batch_size=None, name="input"):
    if isinstance(x, tuple):
        x = list(x)
    elif not isinstance(x, list):
        x = [x]

    if batch_size is not None:
        if len(x) == 1 and batch_size > 1:
            x = x * batch_size
        elif len(x) != batch_size:
            raise ValueError(f"{name} batch {len(x)} != expected batch {batch_size}")

    return x


def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def _ensure_batch_tensor(x, single_ndim, name):
    if isinstance(x, (list, tuple)):
        x = torch.stack([_to_tensor(v) for v in x], dim=0)
    else:
        x = _to_tensor(x)

    if x.ndim == single_ndim:
        x = x.unsqueeze(0)

    return x


def _match_batch_tensor(x, batch_size, name):
    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, *([-1] * (x.ndim - 1)))
    elif x.shape[0] != batch_size:
        raise ValueError(f"{name} batch {x.shape[0]} != text batch {batch_size}")

    return x


def _tokenize_text_batch(tokenizer, max_length, device, text, duration_text=None):
    text_batch = _as_batch_list(text, name="text")
    duration_text_batch = _as_batch_list(
        text if duration_text is None else duration_text,
        batch_size=len(text_batch),
        name="duration_text",
    )
    def tok(x):
        t = tokenizer(
            x,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return (
            t["input_ids"].to(device).long(),
            t["attention_mask"].to(device).bool(),
        )
    text_ids, text_mask = tok(text_batch)
    duration_text_ids, duration_text_mask = tok(duration_text_batch)

    return text_ids, text_mask, duration_text_ids, duration_text_mask


def _apg_norm(x) :
    return torch.sqrt((x * x).sum(dim=tuple(range(1, x.dim())), keepdim=True) + 1e-12)


def _apg_project(diff, cond_pred, eta, norm) :
    if norm is not None:
        n = _apg_norm(diff)
        scale = torch.minimum(
            torch.ones_like(n),
            torch.as_tensor(norm, device=diff.device, dtype=diff.dtype) / n,
        )
        diff = diff * scale

    c_hat = cond_pred / _apg_norm(cond_pred)
    parallel = (diff * c_hat).sum(dim=tuple(range(1, diff.dim())), keepdim=True) * c_hat
    orthogonal = diff - parallel

    return orthogonal + float(eta) * parallel


def _apg_remaining_time(t, x, eps = 1e-5) :
    t = t.to(device=x.device, dtype=torch.float32)

    while t.ndim < x.ndim:
        t = t[:, None]

    return (1.0 - t).clamp_min(eps)


def _apg_guidance_single(
    *,
    x,
    t,
    v_cond,
    v_uncond,
    scale,
    eta,
    momentum,
    norm,
    buffer,
) :
    out_dtype = v_cond.dtype
    x_f = x.float()
    v_cond_f = v_cond.float()
    v_uncond_f = v_uncond.float()
    remaining = _apg_remaining_time(t, x_f)
    pred_cond = x_f + remaining * v_cond_f
    pred_uncond = x_f + remaining * v_uncond_f
    diff = pred_cond - pred_uncond

    if momentum is None:
        momentum = 0.0

    if buffer is not None and float(momentum) != 0.0:
        diff = diff + float(momentum) * buffer

    buffer = diff
    update = _apg_project(diff, pred_cond, eta, norm)
    pred = pred_cond + float(scale) * update
    v = (pred - x_f) / remaining

    return v.to(out_dtype), buffer


def _apg_guidance_dual(
    *,
    x,
    t,
    v_full,
    v_text_uncond,
    v_speaker_uncond,
    text_scale,
    speaker_scale,
    eta_text,
    eta_speaker,
    momentum_text,
    momentum_speaker,
    norm_text,
    norm_speaker,
    buffer_text,
    buffer_speaker,
) :
    out_dtype = v_full.dtype
    x_f = x.float()
    v_full_f = v_full.float()
    v_text_uncond_f = v_text_uncond.float()
    v_speaker_uncond_f = v_speaker_uncond.float()
    remaining = _apg_remaining_time(t, x_f)
    pred_full = x_f + remaining * v_full_f
    pred_text_uncond = x_f + remaining * v_text_uncond_f
    pred_speaker_uncond = x_f + remaining * v_speaker_uncond_f
    diff_text = pred_full - pred_text_uncond

    if momentum_text is None:
        momentum_text = 0.0

    if buffer_text is not None and float(momentum_text) != 0.0:
        diff_text = diff_text + float(momentum_text) * buffer_text

    buffer_text = diff_text
    diff_speaker = pred_full - pred_speaker_uncond

    if momentum_speaker is None:
        momentum_speaker = 0.0

    if buffer_speaker is not None and float(momentum_speaker) != 0.0:
        diff_speaker = diff_speaker + float(momentum_speaker) * buffer_speaker

    buffer_speaker = diff_speaker
    update_text = _apg_project(diff_text, pred_full, eta_text, norm_text)
    update_speaker = _apg_project(diff_speaker, pred_full, eta_speaker, norm_speaker)
    pred = pred_full + float(text_scale) * update_text + float(speaker_scale) * update_speaker
    v = (pred - x_f) / remaining

    return v.to(out_dtype), buffer_text, buffer_speaker


def load_model_from_checkpoint(
    config_path,
    ckpt_dir,
    device="cpu",
    use_speaker_conditioning=False,
    dtype=torch.bfloat16,
):
    cfg = load_config(config_path)

    with torch.device("cpu"):
        model = model_from_config(cfg, use_speaker_conditioning=use_speaker_conditioning)

    extra = torch.load(f"{ckpt_dir}/extra_state.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(extra["ema"])
    model = model.to(device=device, dtype=dtype).eval()
    logger.info("Loaded EMA weights from step %s", extra["step"])
    logger.info("Parameters: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)

    return model, cfg


@torch.inference_mode()
def predict_total_duration(
    duration_model,
    text_ids,
    text_mask,
    *,
    cond_latents=None,
    cond_latent_mask=None,
    n_frame_per_class=1,
    min_total_frames=1,
    max_total_frames=None,
):
    device = next(duration_model.parameters()).device
    model_dtype = next(duration_model.parameters()).dtype
    text_ids = _ensure_batch_tensor(text_ids, 1, "text_ids")
    text_mask = _ensure_batch_tensor(text_mask, 1, "text_mask")

    if text_ids.shape[0] != text_mask.shape[0]:
        raise ValueError(f"text_ids batch {text_ids.shape[0]} != text_mask batch {text_mask.shape[0]}")

    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device, dtype=torch.bool)
    B = text_ids.shape[0]
    text_padding_mask = ~text_mask

    if cond_latents is None:
        out = duration_model(
            text_ids=text_ids,
            latent_prompt=None,
            text_padding_mask=text_padding_mask,
        )

        if out.ndim == 3:
            total = out[:, 0, :].argmax(dim=-1) * n_frame_per_class
        else:
            total = out[:, 0].round().long()

        total = total.long().clamp_min(min_total_frames)

        if max_total_frames is not None:
            total = total.clamp_max(max_total_frames)

        return total

    cond_latents = _ensure_batch_tensor(cond_latents, 2, "cond_latents")
    cond_latents = _match_batch_tensor(cond_latents, B, "cond_latents")
    cond_latents = cond_latents.to(device=device, dtype=model_dtype)

    if cond_latent_mask is None:
        cond_latent_mask = torch.ones(
            cond_latents.shape[:2], device=device, dtype=torch.bool
        )
    else:
        cond_latent_mask = _ensure_batch_tensor(cond_latent_mask, 1, "cond_latent_mask")
        cond_latent_mask = _match_batch_tensor(cond_latent_mask, B, "cond_latent_mask")
        cond_latent_mask = cond_latent_mask.to(device=device, dtype=torch.bool)

    out = duration_model(
        text_ids=text_ids,
        latent_prompt=cond_latents,
        text_padding_mask=text_padding_mask,
        latent_padding_mask=~cond_latent_mask,
    )
    prompt_len = cond_latent_mask.sum(dim=1).long()
    last_idx = (prompt_len - 1).clamp_min(0)

    if out.ndim == 3:
        remain_all = out.argmax(dim=-1) * n_frame_per_class
    else:
        remain_all = out.round().long()

    remaining = remain_all.gather(1, last_idx[:, None]).squeeze(1)
    total = prompt_len + remaining
    total = total.clamp_min(min_total_frames)

    if max_total_frames is not None:
        total = total.clamp_max(max_total_frames)

    return total


def _prepare_sampling_inputs(
    model,
    text_ids,
    text_mask,
    *,
    duration=None,
    cond_latents=None,
    cond_latent_mask=None,
    duration_model=None,
    n_frame_per_class=1,
    max_duration=None,
):
    text_ids = _ensure_batch_tensor(text_ids, 1, "text_ids")
    text_mask = _ensure_batch_tensor(text_mask, 1, "text_mask")

    if text_ids.shape[0] != text_mask.shape[0]:
        raise ValueError(f"text_ids batch {text_ids.shape[0]} != text_mask batch {text_mask.shape[0]}")

    B, D, device, dtype = text_ids.shape[0], model.latent_size, model.device, model.dtype
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device, dtype=torch.bool)

    if cond_latents is not None:
        cond_latents = _ensure_batch_tensor(cond_latents, 2, "cond_latents")
        cond_latents = _match_batch_tensor(cond_latents, B, "cond_latents")

    if cond_latent_mask is not None:
        cond_latent_mask = _ensure_batch_tensor(cond_latent_mask, 1, "cond_latent_mask")
        cond_latent_mask = _match_batch_tensor(cond_latent_mask, B, "cond_latent_mask")

    if duration is None:
        if duration_model is None:
            raise ValueError("duration is None but duration_model was not provided")

        duration = predict_total_duration(
            duration_model,
            text_ids=text_ids,
            text_mask=text_mask,
            cond_latents=cond_latents,
            cond_latent_mask=cond_latent_mask,
            n_frame_per_class=n_frame_per_class,
            min_total_frames=1 if cond_latents is None else cond_latents.shape[1] + 1,
            max_total_frames=max_duration,
        )

    if isinstance(duration, (int, float)):
        T = int(duration)
        dur = torch.full((B,), T, device=device, dtype=torch.long)
    else:
        dur = torch.as_tensor(duration, device=device, dtype=torch.long)

        if dur.ndim == 0:
            dur = dur.expand(B)
        elif dur.ndim == 1 and dur.numel() == 1 and B > 1:
            dur = dur.expand(B)
        elif dur.ndim != 1 or dur.numel() != B:
            raise ValueError(f"duration batch {dur.numel()} != text batch {B}")

        T = int(dur.max().item())

    valid = torch.arange(T, device=device)[None] < dur[:, None]
    cond = torch.zeros(B, T, D, device=device, dtype=dtype)
    cond_mask = torch.zeros(B, T, device=device, dtype=torch.bool)

    if cond_latents is not None:
        cond_latents = cond_latents.to(device=device, dtype=dtype)
        Lp = min(cond_latents.shape[1], T)
        cond[:, :Lp] = cond_latents[:, :Lp]

        if cond_latent_mask is None:
            cond_latent_mask = torch.ones(B, Lp, device=device, dtype=torch.bool)
        else:
            cond_latent_mask = cond_latent_mask.to(device=device, dtype=torch.bool)[:, :Lp]

        cond_mask[:, :Lp] = cond_latent_mask & valid[:, :Lp]

    span_mask = valid & ~cond_mask
    common = dict(
        text_ids=text_ids,
        text_mask=text_mask,
        reference_latent=None,
        reference_mask=None,
        speaker_cond_drop=None,
        use_checkpoint=False,
        latent_mask=valid,
        valid_audio_mask=valid,
        span_mask=span_mask,
        audio_cond_drop=None,
    )

    return B, D, device, dtype, T, dur, valid, cond, cond_mask, span_mask, common


@torch.inference_mode()
def sample_euler(
    model, text_ids, text_mask, *,
    duration=None,
    cond_latents=None,
    cond_latent_mask=None,
    duration_model=None,
    n_frame_per_class=1,
    max_duration=None,
    steps=32, cfg=2.0, seed=None,
):
    B, D, device, dtype, T, dur, valid, cond, cond_mask, span_mask, common = _prepare_sampling_inputs(
        model, text_ids, text_mask,
        duration=duration,
        cond_latents=cond_latents,
        cond_latent_mask=cond_latent_mask,
        duration_model=duration_model,
        n_frame_per_class=n_frame_per_class,
        max_duration=max_duration,
    )
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    x = torch.randn(B, T, D, device=device, dtype=dtype, generator=g) * valid.unsqueeze(-1).to(dtype)
    times = torch.linspace(0, 1, steps + 1, device=device)

    for i in range(steps):
        t = times[i].expand(B).to(dtype)
        dt = (times[i + 1] - times[i]).to(dtype)
        v = model(x=x, cond=cond, t=t, text_cond_drop=None, **common).to(dtype)

        if cfg > 0:
            v_null = model(
                x=x, cond=cond, t=t,
                text_cond_drop=torch.ones(B, device=device, dtype=torch.bool),
                **common
            ).to(dtype)
            v = v + cfg * (v - v_null)

        x = ((x + dt * v) * valid.unsqueeze(-1).to(dtype)).to(dtype)
        x = torch.where(cond_mask.unsqueeze(-1), cond, x)

    return torch.where(cond_mask.unsqueeze(-1), cond, x).float()


@torch.inference_mode()
def sample_midpoint(
    model, text_ids, text_mask, *,
    duration=None,
    cond_latents=None,
    cond_latent_mask=None,
    duration_model=None,
    n_frame_per_class=1,
    max_duration=None,
    steps=16, cfg=2.0, seed=None,
):
    B, D, device, dtype, T, dur, valid, cond, cond_mask, span_mask, common = _prepare_sampling_inputs(
        model, text_ids, text_mask,
        duration=duration,
        cond_latents=cond_latents,
        cond_latent_mask=cond_latent_mask,
        duration_model=duration_model,
        n_frame_per_class=n_frame_per_class,
        max_duration=max_duration,
    )
    def vel(x, t):
        v = model(x=x, cond=cond, t=t, text_cond_drop=None, **common).to(dtype)

        if cfg > 0:
            v_null = model(
                x=x, cond=cond, t=t,
                text_cond_drop=torch.ones(B, device=device, dtype=torch.bool),
                **common
            ).to(dtype)
            v = v + cfg * (v - v_null)

        return v
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    x = torch.randn(B, T, D, device=device, dtype=dtype, generator=g) * valid.unsqueeze(-1).to(dtype)
    times = torch.linspace(0, 1, steps + 1, device=device)

    for i in range(steps):
        dt = (times[i + 1] - times[i]).to(dtype)
        v1 = vel(x, times[i].expand(B).to(dtype))
        x_mid = (x + dt * v1) * valid.unsqueeze(-1).to(dtype)
        x_mid = torch.where(cond_mask.unsqueeze(-1), cond, x_mid).to(dtype)
        v2 = vel(x_mid, times[i + 1].expand(B).to(dtype))
        x = (x + dt * 0.5 * (v1 + v2)) * valid.unsqueeze(-1).to(dtype)
        x = torch.where(cond_mask.unsqueeze(-1), cond, x).to(dtype)

    return torch.where(cond_mask.unsqueeze(-1), cond, x).float()


@torch.inference_mode()
def sample_euler_reducio(
    model, text_ids, text_mask, *,
    duration=None,
    cond_latents=None,
    cond_latent_mask=None,
    duration_model=None,
    n_frame_per_class=1,
    max_duration=None,
    steps=32, cfg=2.0, seed=None,
    ts_fraction=0.2, bs_fraction=0.2,
    apg_eta = None,
    apg_momentum = None,
    apg_norm = None,
):
    B, D, device, dtype, T, dur, valid, cond, cond_mask, span_mask, common = _prepare_sampling_inputs(
        model, text_ids, text_mask,
        duration=duration,
        cond_latents=cond_latents,
        cond_latent_mask=cond_latent_mask,
        duration_model=duration_model,
        n_frame_per_class=n_frame_per_class,
        max_duration=max_duration,
    )
    n_ts = int(steps * ts_fraction)
    n_bs = int(steps * bs_fraction)
    ts_start = steps - n_ts
    bs_start = ts_start - n_bs
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    x = torch.randn(B, T, D, device=device, dtype=dtype, generator=g) * valid.unsqueeze(-1).to(dtype)
    prev_v = None
    prev_v_cond = None
    prev_v_null = None
    apg_buffer = None
    use_apg = apg_eta is not None
    times = torch.linspace(0, 1, steps + 1, device=device)

    for i in range(steps):
        t = times[i].expand(B).to(dtype)
        dt = (times[i + 1] - times[i]).to(dtype)

        if i >= ts_start and prev_v is not None:
            v = prev_v
        else:
            v_cond = model(x=x, cond=cond, t=t, text_cond_drop=None, **common).to(dtype)

            if cfg > 0:
                if i >= bs_start and i < ts_start and prev_v_cond is not None:
                    v_null = prev_v_null - prev_v_cond + v_cond
                else:
                    v_null = model(
                        x=x, cond=cond, t=t,
                        text_cond_drop=torch.ones(B, device=device, dtype=torch.bool),
                        **common
                    ).to(dtype)

                if use_apg:
                    v, apg_buffer = _apg_guidance_single(
                        x=x,
                        t=t,
                        v_cond=v_cond,
                        v_uncond=v_null,
                        scale=cfg,
                        eta=apg_eta,
                        momentum=apg_momentum,
                        norm=apg_norm,
                        buffer=apg_buffer,
                    )
                else:
                    v = v_cond + cfg * (v_cond - v_null)

                prev_v_cond = v_cond
                prev_v_null = v_null
            else:
                v = v_cond

            prev_v = v

        x = ((x + dt * v) * valid.unsqueeze(-1).to(dtype)).to(dtype)
        x = torch.where(cond_mask.unsqueeze(-1), cond, x)

    return torch.where(cond_mask.unsqueeze(-1), cond, x).float()


@torch.inference_mode()
def sample_midpoint_reducio(
    model, text_ids, text_mask, *,
    duration=None,
    cond_latents=None,
    cond_latent_mask=None,
    duration_model=None,
    n_frame_per_class=1,
    max_duration=None,
    steps=16, cfg=2.0, seed=None,
    ts_fraction=0.2, bs_fraction=0.2,
):
    B, D, device, dtype, T, dur, valid, cond, cond_mask, span_mask, common = _prepare_sampling_inputs(
        model, text_ids, text_mask,
        duration=duration,
        cond_latents=cond_latents,
        cond_latent_mask=cond_latent_mask,
        duration_model=duration_model,
        n_frame_per_class=n_frame_per_class,
        max_duration=max_duration,
    )
    n_ts = int(steps * ts_fraction)
    n_bs = int(steps * bs_fraction)
    ts_start = steps - n_ts
    bs_start = ts_start - n_bs
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    x = torch.randn(B, T, D, device=device, dtype=dtype, generator=g) * valid.unsqueeze(-1).to(dtype)
    prev_v = None
    prev_v_cond = None
    prev_v_null = None
    times = torch.linspace(0, 1, steps + 1, device=device)

    for i in range(steps):
        t0, t1 = times[i], times[i + 1]
        dt = (t1 - t0).to(dtype)

        if i >= ts_start and prev_v is not None:
            v = prev_v
        else:
            v1_cond = model(
                x=x, cond=cond, t=t0.expand(B).to(dtype),
                text_cond_drop=None, **common
            ).to(dtype)

            if cfg > 0:
                if i >= bs_start and i < ts_start and prev_v_cond is not None:
                    v1_null = prev_v_null - prev_v_cond + v1_cond
                else:
                    v1_null = model(
                        x=x, cond=cond, t=t0.expand(B).to(dtype),
                        text_cond_drop=torch.ones(B, device=device, dtype=torch.bool),
                        **common
                    ).to(dtype)

                v1 = v1_cond + cfg * (v1_cond - v1_null)
                prev_v_cond = v1_cond
                prev_v_null = v1_null
            else:
                v1 = v1_cond

            x_mid = (x + dt * v1) * valid.unsqueeze(-1).to(dtype)
            x_mid = torch.where(cond_mask.unsqueeze(-1), cond, x_mid).to(dtype)
            v2 = model(
                x=x_mid, cond=cond, t=t1.expand(B).to(dtype),
                text_cond_drop=None, **common
            ).to(dtype)

            if cfg > 0:
                v2_null = model(
                    x=x_mid, cond=cond, t=t1.expand(B).to(dtype),
                    text_cond_drop=torch.ones(B, device=device, dtype=torch.bool),
                    **common
                ).to(dtype)
                v2 = v2 + cfg * (v2 - v2_null)

            v = 0.5 * (v1 + v2)
            prev_v = v

        x = (x + dt * v) * valid.unsqueeze(-1).to(dtype)
        x = torch.where(cond_mask.unsqueeze(-1), cond, x).to(dtype)

    return torch.where(cond_mask.unsqueeze(-1), cond, x).float()


@torch.inference_mode()
def sample_euler_reducio_spk(
    model,
    text_ids,
    text_mask,
    *,
    duration=None,
    cond_latents=None,
    cond_latent_mask=None,
    duration_model=None,
    n_frame_per_class=1,
    max_duration=None,
    reference_latent=None,
    reference_mask=None,
    speaker_emb=None,
    speaker_adaln_scale = 1.0,
    steps=32,
    cfg=2.0,
    speaker_cfg = None,
    seed=None,
    ts_fraction=0.2,
    bs_fraction=0.2,
    apg_eta_text = None,
    apg_eta_speaker = None,
    apg_momentum_text = None,
    apg_momentum_speaker = None,
    apg_norm_text = None,
    apg_norm_speaker = None,
):
    if speaker_cfg is None:
        speaker_cfg = 0.0

    use_dual_cfg = speaker_cfg > 0.0 and speaker_emb is not None
    use_dual_apg = use_dual_cfg and (apg_eta_text is not None or apg_eta_speaker is not None)
    use_single_apg = (not use_dual_cfg) and apg_eta_text is not None

    if apg_eta_text is None:
        apg_eta_text = 1.0

    if apg_eta_speaker is None:
        apg_eta_speaker = 1.0

    B, D, device, dtype, T, dur, valid, cond, cond_mask, span_mask, common = _prepare_sampling_inputs(
        model, text_ids, text_mask,
        duration=duration,
        cond_latents=cond_latents,
        cond_latent_mask=cond_latent_mask,
        duration_model=duration_model,
        n_frame_per_class=n_frame_per_class,
        max_duration=max_duration,
    )

    if reference_latent is not None:
        reference_latent = _ensure_batch_tensor(reference_latent, 1, "reference_latent")
        reference_latent = reference_latent.to(device=device, dtype=dtype)

        if reference_latent.ndim == 2:
            if reference_latent.shape[0] == 1 and B > 1:
                reference_latent = reference_latent.expand(B, -1)
            elif reference_latent.shape[0] != B:
                raise ValueError(
                    f"reference_latent batch {reference_latent.shape[0]} != text batch {B}"
                )

            reference_latent = reference_latent.unsqueeze(1)

            if reference_mask is None:
                reference_mask = torch.ones(B, 1, device=device, dtype=torch.bool)
            else:
                reference_mask = _ensure_batch_tensor(reference_mask, 0, "reference_mask")
                reference_mask = _match_batch_tensor(reference_mask, B, "reference_mask")
                reference_mask = reference_mask.to(device=device, dtype=torch.bool)

                if reference_mask.shape == (B,):
                    reference_mask = reference_mask.unsqueeze(1)
                elif reference_mask.shape != (B, 1):
                    raise ValueError(
                        f"for global reference_latent, expected reference_mask {(B,)} or {(B, 1)}, "
                        f"got {tuple(reference_mask.shape)}"
                    )
        elif reference_latent.ndim == 3:
            if reference_latent.shape[0] == 1 and B > 1:
                reference_latent = reference_latent.expand(B, -1, -1)
            elif reference_latent.shape[0] != B:
                raise ValueError(
                    f"reference_latent batch {reference_latent.shape[0]} != text batch {B}"
                )

            if reference_mask is None:
                reference_mask = torch.ones(
                    B, reference_latent.shape[1], device=device, dtype=torch.bool
                )
            else:
                reference_mask = _ensure_batch_tensor(reference_mask, 1, "reference_mask")
                reference_mask = _match_batch_tensor(reference_mask, B, "reference_mask")
                reference_mask = reference_mask.to(device=device, dtype=torch.bool)

                if reference_mask.shape != (B, reference_latent.shape[1]):
                    raise ValueError(
                        f"for sequence reference_latent, expected reference_mask "
                        f"{(B, reference_latent.shape[1])}, got {tuple(reference_mask.shape)}"
                    )
        else:
            raise ValueError(
                f"expected reference_latent [B, D_ref] or [B, T_ref, D_ref], "
                f"got {tuple(reference_latent.shape)}"
            )
    else:
        reference_mask = None

    if speaker_emb is not None:
        speaker_emb = _ensure_batch_tensor(speaker_emb, 1, "speaker_emb")
        speaker_emb = speaker_emb.to(device=device, dtype=dtype)

        if speaker_emb.ndim == 1:
            speaker_emb = speaker_emb.unsqueeze(0)

        if speaker_emb.ndim != 2:
            raise ValueError(
                f"expected speaker_emb [B, D_spk] or [D_spk], got {tuple(speaker_emb.shape)}"
            )

        if speaker_emb.shape[0] == 1 and B > 1:
            speaker_emb = speaker_emb.expand(B, -1)
        elif speaker_emb.shape[0] != B:
            raise ValueError(
                f"speaker_emb batch {speaker_emb.shape[0]} != text batch {B}"
            )

    common["reference_latent"] = reference_latent
    common["reference_mask"] = reference_mask
    common["speaker_emb"] = speaker_emb
    common["speaker_adaln_scale"] = speaker_adaln_scale
    text_drop_mask = torch.ones(B, device=device, dtype=torch.bool)
    speaker_drop_mask = torch.ones(B, device=device, dtype=torch.bool)
    n_ts = int(steps * ts_fraction)
    n_bs = int(steps * bs_fraction)
    ts_start = steps - n_ts
    bs_start = ts_start - n_bs
    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    x = torch.randn(B, T, D, device=device, dtype=dtype, generator=g)
    x = x * valid.unsqueeze(-1).to(dtype)
    prev_v = None
    prev_v_full = None
    prev_v_text = None
    prev_v_null = None
    prev_v_speaker = None
    apg_buffer_text = None
    apg_buffer_speaker = None
    times = torch.linspace(0, 1, steps + 1, device=device)

    for i in range(steps):
        t = times[i].expand(B).to(dtype)
        dt = (times[i + 1] - times[i]).to(dtype)

        if i >= ts_start and prev_v is not None:
            v = prev_v
        else:
            if use_dual_cfg:
                common["speaker_cond_drop"] = None
                v_full = model(
                    x=x, cond=cond, t=t,
                    text_cond_drop=None,
                    **common,
                ).to(dtype)

                if i >= bs_start and i < ts_start and prev_v_full is not None:
                    v_text = prev_v_text - prev_v_full + v_full
                else:
                    common["speaker_cond_drop"] = speaker_drop_mask
                    v_text = model(
                        x=x, cond=cond, t=t,
                        text_cond_drop=None,
                        **common,
                    ).to(dtype)

                if use_dual_apg:
                    if i >= bs_start and i < ts_start and prev_v_full is not None:
                        v_speaker = prev_v_speaker - prev_v_full + v_full
                    else:
                        common["speaker_cond_drop"] = None
                        v_speaker = model(
                            x=x, cond=cond, t=t,
                            text_cond_drop=text_drop_mask,
                            **common,
                        ).to(dtype)

                    v, apg_buffer_text, apg_buffer_speaker = _apg_guidance_dual(
                        x=x,
                        t=t,
                        v_full=v_full,
                        v_text_uncond=v_speaker,
                        v_speaker_uncond=v_text,
                        text_scale=cfg,
                        speaker_scale=speaker_cfg,
                        eta_text=apg_eta_text,
                        eta_speaker=apg_eta_speaker,
                        momentum_text=apg_momentum_text,
                        momentum_speaker=apg_momentum_speaker,
                        norm_text=apg_norm_text,
                        norm_speaker=apg_norm_speaker,
                        buffer_text=apg_buffer_text,
                        buffer_speaker=apg_buffer_speaker,
                    )
                    prev_v_speaker = v_speaker
                else:
                    if i >= bs_start and i < ts_start and prev_v_full is not None:
                        v_null = prev_v_null - prev_v_full + v_full
                    else:
                        common["speaker_cond_drop"] = speaker_drop_mask
                        v_null = model(
                            x=x, cond=cond, t=t,
                            text_cond_drop=text_drop_mask,
                            **common,
                        ).to(dtype)

                    v = v_full + cfg * (v_text - v_null) + speaker_cfg * (v_full - v_text)
                    prev_v_null = v_null

                prev_v_full = v_full
                prev_v_text = v_text
            else:
                common["speaker_cond_drop"] = None
                v_cond = model(
                    x=x, cond=cond, t=t,
                    text_cond_drop=None,
                    **common,
                ).to(dtype)

                if cfg > 0:
                    if i >= bs_start and i < ts_start and prev_v_full is not None:
                        v_null = prev_v_null - prev_v_full + v_cond
                    else:
                        common["speaker_cond_drop"] = speaker_drop_mask
                        v_null = model(
                            x=x, cond=cond, t=t,
                            text_cond_drop=text_drop_mask,
                            **common,
                        ).to(dtype)

                    if use_single_apg:
                        v, apg_buffer_text = _apg_guidance_single(
                            x=x,
                            t=t,
                            v_cond=v_cond,
                            v_uncond=v_null,
                            scale=cfg,
                            eta=apg_eta_text,
                            momentum=apg_momentum_text,
                            norm=apg_norm_text,
                            buffer=apg_buffer_text,
                        )
                    else:
                        v = v_cond + cfg * (v_cond - v_null)

                    prev_v_full = v_cond
                    prev_v_null = v_null
                else:
                    v = v_cond

            prev_v = v

        x = (x + dt * v).to(dtype)
        x = x * valid.unsqueeze(-1).to(dtype)
        x = torch.where(cond_mask.unsqueeze(-1), cond, x)

    return torch.where(cond_mask.unsqueeze(-1), cond, x).float()


def load_duration_model(
    ckpt_path,
    *,
    vocab_size,
    latent_dim,
    hidden_dim=256,
    n_text_layer=4,
    n_cross_layer=4,
    n_head=8,
    output_dim=512,
    device="cuda",
    use_speaker_conditioning=False,
):
    model = SpeechLengthPredictor(
        vocab_size=vocab_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        n_text_layer=n_text_layer,
        n_cross_layer=n_cross_layer,
        n_head=n_head,
        output_dim=output_dim,
        use_speaker_conditioning=use_speaker_conditioning,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    return model


@torch.no_grad()
def decode_audio(dac_model, z):
    z = z.transpose(1, 2)
    z_len = torch.full((z.shape[0],), z.shape[-1], device=z.device, dtype=torch.long)
    codes = dac_model.codec.vector_quantizer.encode(inputs=z, input_len=z_len)
    codes = codes.permute(1, 0, 2).contiguous()
    audio = dac_model.decode(codes)[0]

    return audio


def build_latent_edit_condition(
    original_lats,
    wav_num_samples,
    wav_sr,
    parts_to_edit,
    *,
    extra_duration=None,
    duration_scale=1.0,
    padding_sec=0.25,
    min_edit_sec=0.10,
    codec_rate_hz=None,
    device="cuda",
    dtype=None,
):
    if not isinstance(original_lats, torch.Tensor):
        original_lats = torch.as_tensor(original_lats)

    if original_lats.ndim != 2:
        raise ValueError(
            f"expected original_lats [T, D], got {tuple(original_lats.shape)}"
        )

    if dtype is None:
        dtype = original_lats.dtype

    original_lats = original_lats.to(device=device, dtype=dtype)
    T_src, D = original_lats.shape
    audio_dur_sec = float(wav_num_samples) / float(wav_sr)

    if codec_rate_hz is None:
        codec_rate_hz = T_src / max(audio_dur_sec, 1e-8)

    if not isinstance(parts_to_edit, (list, tuple)) or len(parts_to_edit) == 0:
        raise ValueError("parts_to_edit must be a non-empty list of [start_sec, end_sec]")

    num_parts = len(parts_to_edit)
    def _expand_param(value, name, default):
        if value is None:
            return [default] * num_parts

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()

        if isinstance(value, (int, float)):
            return [float(value)] * num_parts

        if isinstance(value, (list, tuple)):
            if len(value) == 1 and num_parts > 1:
                return [float(value[0])] * num_parts

            if len(value) != num_parts:
                raise ValueError(
                    f"{name} has length {len(value)}, but parts_to_edit has length {num_parts}"
                )

            return [float(v) for v in value]

        raise TypeError(f"{name} must be None, a number, or a list/tuple of numbers")
    extra_durations = _expand_param(extra_duration, "extra_duration", 0.0)
    duration_scales = _expand_param(duration_scale, "duration_scale", 1.0)
    normalized_parts = []

    for i, part in enumerate(parts_to_edit):
        if not isinstance(part, (list, tuple)) or len(part) != 2:
            raise ValueError(
                f"each edit part must be [start_sec, end_sec], got {part!r}"
            )

        raw_start_sec, raw_end_sec = part

        if raw_start_sec is None:
            raw_start_sec = 0.0

        if raw_end_sec is None:
            raw_end_sec = audio_dur_sec

        raw_start_sec = float(raw_start_sec)
        raw_end_sec = float(raw_end_sec)

        if raw_start_sec < 0:
            raw_start_sec = 0.0

        if raw_end_sec > audio_dur_sec:
            raw_end_sec = audio_dur_sec

        if raw_end_sec < raw_start_sec:
            raise ValueError(
                f"edit part {i} has end before start: {part!r}"
            )

        padded_start_sec = max(0.0, raw_start_sec - float(padding_sec))
        padded_end_sec = min(audio_dur_sec, raw_end_sec + float(padding_sec))
        start_frame = int(round(padded_start_sec * codec_rate_hz))
        end_frame = int(round(padded_end_sec * codec_rate_hz))
        start_frame = max(0, min(start_frame, T_src))
        end_frame = max(start_frame, min(end_frame, T_src))
        original_region_sec = max(0.0, padded_end_sec - padded_start_sec)
        new_region_sec = (
            original_region_sec * duration_scales[i]
            + extra_durations[i]
        )
        new_region_sec = max(float(min_edit_sec), new_region_sec)
        new_region_frames = max(1, int(round(new_region_sec * codec_rate_hz)))
        normalized_parts.append(
            {
                "index": i,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "new_region_frames": new_region_frames,
            }
        )

    normalized_parts.sort(key=lambda x: x["start_frame"])

    for prev, cur in zip(normalized_parts[:-1], normalized_parts[1:]):
        if cur["start_frame"] < prev["end_frame"]:
            raise ValueError(
                "edit regions overlap after padding_sec was applied. "
                "Use one wider edit region instead of overlapping regions."
            )

    cond_chunks = []
    keep_chunks = []
    edit_ranges_out = []
    src_offset = 0
    out_offset = 0

    for part in normalized_parts:
        start_frame = part["start_frame"]
        end_frame = part["end_frame"]
        new_region_frames = part["new_region_frames"]
        keep_len = start_frame - src_offset

        if keep_len > 0:
            cond_chunks.append(original_lats[src_offset:start_frame])
            keep_chunks.append(
                torch.ones(keep_len, device=device, dtype=torch.bool)
            )
            out_offset += keep_len

        cond_chunks.append(
            torch.zeros(new_region_frames, D, device=device, dtype=dtype)
        )
        keep_chunks.append(
            torch.zeros(new_region_frames, device=device, dtype=torch.bool)
        )
        edit_ranges_out.append([out_offset, out_offset + new_region_frames])
        out_offset += new_region_frames
        src_offset = end_frame

    if src_offset < T_src:
        tail_len = T_src - src_offset
        cond_chunks.append(original_lats[src_offset:])
        keep_chunks.append(
            torch.ones(tail_len, device=device, dtype=torch.bool)
        )

    cond = torch.cat(cond_chunks, dim=0).unsqueeze(0)
    keep_mask = torch.cat(keep_chunks, dim=0).unsqueeze(0)
    valid = torch.ones_like(keep_mask, dtype=torch.bool)
    span_mask = valid & ~keep_mask

    return cond, keep_mask, span_mask, valid, edit_ranges_out


@torch.inference_mode()
def sample_euler_reducio_edit(
    model,
    text_ids,
    text_mask,
    *,
    cond,
    keep_mask,
    valid_mask=None,
    steps=32,
    cfg=2.0,
    seed=None,
    ts_fraction=0.2,
    bs_fraction=0.2,
    apg_eta = None,
    apg_momentum = None,
    apg_norm = None,
):
    device = model.device
    dtype = model.dtype

    if not isinstance(text_ids, torch.Tensor):
        text_ids = torch.as_tensor(text_ids)

    if not isinstance(text_mask, torch.Tensor):
        text_mask = torch.as_tensor(text_mask)

    if text_ids.ndim == 1:
        text_ids = text_ids.unsqueeze(0)

    if text_mask.ndim == 1:
        text_mask = text_mask.unsqueeze(0)

    text_ids = text_ids.to(device=device, dtype=torch.long)
    text_mask = text_mask.to(device=device, dtype=torch.bool)

    if not isinstance(cond, torch.Tensor):
        cond = torch.as_tensor(cond)

    if cond.ndim == 2:
        cond = cond.unsqueeze(0)

    if cond.ndim != 3:
        raise ValueError(f"expected cond [B, T, D] or [T, D], got {tuple(cond.shape)}")

    cond = cond.to(device=device, dtype=dtype)

    if not isinstance(keep_mask, torch.Tensor):
        keep_mask = torch.as_tensor(keep_mask)

    if keep_mask.ndim == 1:
        keep_mask = keep_mask.unsqueeze(0)

    keep_mask = keep_mask.to(device=device, dtype=torch.bool)
    B, T, D = cond.shape

    if keep_mask.shape != (B, T):
        if keep_mask.shape[0] == 1 and B > 1 and keep_mask.shape[1] == T:
            keep_mask = keep_mask.expand(B, T)
        else:
            raise ValueError(
                f"keep_mask shape {tuple(keep_mask.shape)} does not match cond shape {(B, T, D)}"
            )

    if text_ids.shape[0] == 1 and B > 1:
        text_ids = text_ids.expand(B, -1)

    if text_mask.shape[0] == 1 and B > 1:
        text_mask = text_mask.expand(B, -1)

    if text_ids.shape[0] != B:
        raise ValueError(
            f"text batch {text_ids.shape[0]} does not match cond batch {B}"
        )

    if valid_mask is None:
        valid = torch.ones(B, T, device=device, dtype=torch.bool)
    else:
        if not isinstance(valid_mask, torch.Tensor):
            valid_mask = torch.as_tensor(valid_mask)

        if valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)

        valid = valid_mask.to(device=device, dtype=torch.bool)

        if valid.shape != (B, T):
            if valid.shape[0] == 1 and B > 1 and valid.shape[1] == T:
                valid = valid.expand(B, T)
            else:
                raise ValueError(
                    f"valid_mask shape {tuple(valid.shape)} does not match cond shape {(B, T, D)}"
                )

    cond_mask = keep_mask & valid
    span_mask = valid & ~cond_mask
    common = dict(
        text_ids=text_ids,
        text_mask=text_mask,
        reference_latent=None,
        reference_mask=None,
        speaker_cond_drop=None,
        use_checkpoint=False,
        latent_mask=valid,
        valid_audio_mask=valid,
        span_mask=span_mask,
        audio_cond_drop=None,
    )
    n_ts = int(steps * ts_fraction)
    n_bs = int(steps * bs_fraction)
    ts_start = steps - n_ts
    bs_start = ts_start - n_bs

    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    x = torch.randn(
        B,
        T,
        D,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    x = x * valid.unsqueeze(-1).to(dtype)
    x = torch.where(cond_mask.unsqueeze(-1), cond, x)
    prev_v = None
    prev_v_cond = None
    prev_v_null = None
    apg_buffer = None
    use_apg = apg_eta is not None
    times = torch.linspace(0.0, 1.0, steps + 1, device=device)
    text_drop_mask = torch.ones(B, device=device, dtype=torch.bool)

    for i in range(steps):
        t = times[i].expand(B).to(dtype)
        dt = (times[i + 1] - times[i]).to(dtype)

        if i >= ts_start and prev_v is not None:
            v = prev_v
        else:
            v_cond = model(
                x=x,
                cond=cond,
                t=t,
                text_cond_drop=None,
                **common,
            ).to(dtype)

            if cfg > 0:
                if i >= bs_start and i < ts_start and prev_v_cond is not None:
                    v_null = prev_v_null - prev_v_cond + v_cond
                else:
                    v_null = model(
                        x=x,
                        cond=cond,
                        t=t,
                        text_cond_drop=text_drop_mask,
                        **common,
                    ).to(dtype)

                if use_apg:
                    v, apg_buffer = _apg_guidance_single(
                        x=x,
                        t=t,
                        v_cond=v_cond,
                        v_uncond=v_null,
                        scale=cfg,
                        eta=apg_eta,
                        momentum=apg_momentum,
                        norm=apg_norm,
                        buffer=apg_buffer,
                    )
                else:
                    v = v_cond + float(cfg) * (v_cond - v_null)

                prev_v_cond = v_cond
                prev_v_null = v_null
            else:
                v = v_cond

            prev_v = v

        x = x + dt * v
        x = x * valid.unsqueeze(-1).to(dtype)
        x = torch.where(cond_mask.unsqueeze(-1), cond, x).to(dtype)

    x = torch.where(cond_mask.unsqueeze(-1), cond, x)
    x = x * valid.unsqueeze(-1).to(dtype)

    return x.float()


def strip_tags(text) :
    return re.sub(r'^(?:<[^>]+>\s)+', '', text).strip()


def get_silence_slice(
    silence_fsq,
    seconds,
    *,
    codec_rate_hz=12.5,
    device="cuda",
    dtype=torch.float32,
):
    n_frames = int(round(seconds * codec_rate_hz))
    n_frames = max(1, n_frames)

    if n_frames > silence_fsq.shape[0]:
        raise ValueError(
            f"Requested {n_frames} silence frames, but silence_fsq only has "
            f"{silence_fsq.shape[0]} frames."
        )

    return silence_fsq[:n_frames].to(device=device, dtype=dtype)


class Extractor:
    def __init__(self, tokenizer, cfg, device, duration_model, speaker_model):
        self.tokenizer = tokenizer
        self.max_length = cfg["data"]["max_text_length"]
        self.max_duration = int(cfg["data"]["max_audio_seconds"] * cfg["data"]["codec_rate_hz"])
        self.device = device
        self.duration_model = duration_model
        self.speaker_model = speaker_model
    def get_tokens(self, text, duration_text=None):
        return _tokenize_text_batch(
            self.tokenizer,
            self.max_length,
            self.device,
            text,
            duration_text,
            )
    def get_duration(
        self,
        duration_text_ids,
        duration_text_mask,
        speed=1.0,
        *,
        cond_latents=None,
        cond_latent_mask=None,
        n_frame_per_class=1,
        min_total_frames=None,
        max_total_frames=None,
        duration_breathing=0.0,
    ):
        if max_total_frames is None:
            max_total_frames = self.max_duration

        if cond_latents is not None:
            if cond_latents.ndim == 2:
                cond_latents = cond_latents.unsqueeze(0)

            if cond_latents.ndim != 3:
                raise ValueError(
                    f"expected cond_latents [T, D] or [1, T, D], got {tuple(cond_latents.shape)}"
                )

            if cond_latents.shape[0] != 1:
                raise ValueError(
                    f"non-batched Extractor expects cond_latents batch 1, got {cond_latents.shape[0]}"
                )

            if cond_latent_mask is not None:
                if cond_latent_mask.ndim == 1:
                    cond_latent_mask = cond_latent_mask.unsqueeze(0)

                if cond_latent_mask.shape[0] != 1:
                    raise ValueError(
                        "non-batched Extractor expects cond_latent_mask batch 1, "
                        f"got {cond_latent_mask.shape[0]}"
                    )

                prompt_len = int(cond_latent_mask.to(torch.bool).sum().item())
            else:
                prompt_len = int(cond_latents.shape[1])

            pred_min_total_frames = prompt_len + 1
        else:
            prompt_len = 0
            pred_min_total_frames = 1

        if min_total_frames is not None:
            pred_min_total_frames = min_total_frames

        pred = predict_total_duration(
            self.duration_model,
            text_ids=duration_text_ids,
            text_mask=duration_text_mask,
            cond_latents=cond_latents,
            cond_latent_mask=cond_latent_mask,
            n_frame_per_class=n_frame_per_class,
            min_total_frames=pred_min_total_frames,
            max_total_frames=max_total_frames,
        ).long()
        prompt_len_tensor = torch.full_like(pred, int(prompt_len))
        remaining = (pred - prompt_len_tensor).clamp_min(1)
        remaining = (remaining.float() / float(speed))

        if duration_breathing is not None and float(duration_breathing) > 0.0:
            remaining = remaining * (1.0 + float(duration_breathing))

        remaining = torch.ceil(remaining).long().clamp_min(1)
        duration = prompt_len_tensor + remaining
        duration = duration.clamp(
            min=pred_min_total_frames,
            max=max_total_frames,
        )

        return duration
