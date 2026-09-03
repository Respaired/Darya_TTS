import numpy as np
import torch
import onnxruntime as ort
import librosa

def f32_to_bf16_u16(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + r) >> 16).astype(np.uint16)


def bf16_u16_to_f32(x):
    x = np.asarray(x, dtype=np.uint16)
    u = x.astype(np.uint32) << 16
    return u.view(np.float32)


def as_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _require_batch_size_one(b, fn_name, allow_batch=False):
    if not allow_batch and int(b) != 1:
        raise ValueError(
            f"{fn_name} expects batch size 1 for this ONNX export, got B={b}. "
            "Pass allow_batch=True only if your encoder/decoder ONNX graphs were "
            "exported and validated for dynamic batch > 1."
        )


def _validate_cfg_window(cfg_min_t, cfg_max_t):
    cfg_min_t = float(cfg_min_t)
    cfg_max_t = float(cfg_max_t)
    if cfg_min_t > cfg_max_t:
        raise ValueError(
            f"cfg_min_t must be <= cfg_max_t, got {cfg_min_t} > {cfg_max_t}"
        )
    return cfg_min_t, cfg_max_t


def _cfg_timestep_enabled(t_value, cfg_min_t, cfg_max_t):
    t_value = float(t_value)
    return cfg_min_t <= t_value <= cfg_max_t


def _apg_norm_np(x):
    x = np.asarray(x, dtype=np.float32)
    axes = tuple(range(1, x.ndim))
    return np.sqrt(np.sum(x * x, axis=axes, keepdims=True) + np.float32(1e-12))


def _apg_project_np(diff, cond_pred, eta, norm):
    diff = np.asarray(diff, dtype=np.float32)
    cond_pred = np.asarray(cond_pred, dtype=np.float32)

    if norm is not None:
        n = _apg_norm_np(diff)
        scale = np.minimum(
            np.ones_like(n, dtype=np.float32),
            np.float32(norm) / n,
        )
        diff = diff * scale

    c_hat = cond_pred / _apg_norm_np(cond_pred)
    axes = tuple(range(1, diff.ndim))
    parallel = np.sum(diff * c_hat, axis=axes, keepdims=True) * c_hat
    orthogonal = diff - parallel
    return (orthogonal + np.float32(eta) * parallel).astype(np.float32)


def _apg_remaining_time_np(t, x, eps=1e-5):
    t = np.asarray(t, dtype=np.float32)
    while t.ndim < x.ndim:
        t = t[..., None]
    return np.maximum(np.float32(1.0) - t, np.float32(eps)).astype(np.float32)


def _apg_guidance_single_np(
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
):
    x_f = np.asarray(x, dtype=np.float32)
    v_cond_f = np.asarray(v_cond, dtype=np.float32)
    v_uncond_f = np.asarray(v_uncond, dtype=np.float32)
    remaining = _apg_remaining_time_np(t, x_f)

    pred_cond = x_f + remaining * v_cond_f
    pred_uncond = x_f + remaining * v_uncond_f

    diff = pred_cond - pred_uncond
    momentum = 0.0 if momentum is None else float(momentum)
    if buffer is not None and momentum != 0.0:
        diff = diff + np.float32(momentum) * buffer

    buffer = diff.astype(np.float32)
    update = _apg_project_np(diff, pred_cond, eta, norm)
    pred = pred_cond + np.float32(scale) * update
    v = (pred - x_f) / remaining
    return v.astype(np.float32), buffer


def _apg_guidance_dual_np(
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
):
    x_f = np.asarray(x, dtype=np.float32)
    v_full_f = np.asarray(v_full, dtype=np.float32)
    v_text_uncond_f = np.asarray(v_text_uncond, dtype=np.float32)
    v_speaker_uncond_f = np.asarray(v_speaker_uncond, dtype=np.float32)
    remaining = _apg_remaining_time_np(t, x_f)

    pred_full = x_f + remaining * v_full_f
    pred_text_uncond = x_f + remaining * v_text_uncond_f
    pred_speaker_uncond = x_f + remaining * v_speaker_uncond_f

    diff_text = pred_full - pred_text_uncond
    momentum_text = 0.0 if momentum_text is None else float(momentum_text)
    if buffer_text is not None and momentum_text != 0.0:
        diff_text = diff_text + np.float32(momentum_text) * buffer_text
    buffer_text = diff_text.astype(np.float32)

    diff_speaker = pred_full - pred_speaker_uncond
    momentum_speaker = 0.0 if momentum_speaker is None else float(momentum_speaker)
    if buffer_speaker is not None and momentum_speaker != 0.0:
        diff_speaker = diff_speaker + np.float32(momentum_speaker) * buffer_speaker
    buffer_speaker = diff_speaker.astype(np.float32)

    update_text = _apg_project_np(diff_text, pred_full, eta_text, norm_text)
    update_speaker = _apg_project_np(diff_speaker, pred_full, eta_speaker, norm_speaker)

    pred = (
        pred_full
        + np.float32(text_scale) * update_text
        + np.float32(speaker_scale) * update_speaker
    )
    v = (pred - x_f) / remaining
    return v.astype(np.float32), buffer_text, buffer_speaker


class DaryaONNXCore:
    def __init__(
        self,
        encoder_path="onnx_out/tts_encoder_bf16.onnx",
        decoder_path="onnx_out/tts_decoder_denoiser_bf16.onnx",
        provider="CPUExecutionProvider",
    ):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.encoder = ort.InferenceSession(encoder_path, sess_options=so, providers=[provider])
        self.decoder = ort.InferenceSession(decoder_path, sess_options=so, providers=[provider])
        self.encoder_input_types = {x.name: x.type for x in self.encoder.get_inputs()}
        self.encoder_output_types = {x.name: x.type for x in self.encoder.get_outputs()}
        self.decoder_input_types = {x.name: x.type for x in self.decoder.get_inputs()}
        self.decoder_output_types = {x.name: x.type for x in self.decoder.get_outputs()}

    def encode(self, text_ids, text_mask, text_cond_drop=None):
        text_ids = as_numpy(text_ids).astype(np.int64)
        text_mask = as_numpy(text_mask).astype(np.bool_)
        if text_cond_drop is None:
            text_cond_drop = np.zeros((text_ids.shape[0],), dtype=np.bool_)
        else:
            text_cond_drop = as_numpy(text_cond_drop).astype(np.bool_)
        feed = self._make_feed(
            self.encoder,
            {
                "text_ids": text_ids,
                "text_mask": text_mask,
                "text_cond_drop": text_cond_drop,
            },
            self.encoder_input_types,
        )
        out = self.encoder.run(None, feed)
        names = [x.name for x in self.encoder.get_outputs()]
        out = {k: self._decode_output(v, self.encoder_output_types[k]) for k, v in zip(names, out)}
        return out["context"], out["context_mask"].astype(np.bool_)


    def decode_velocity(
        self,
        x,
        t,
        context,
        context_mask,
        latent_mask,
        span_mask,
        valid_audio_mask,
        cond,
        *,
        reference_latent=None,
        reference_mask=None,
        speaker_emb=None,
        speaker_cond_drop=None,
        speaker_adaln_scale=1.0,
    ):
        x_np = as_numpy(x)

        if x_np.ndim != 3 or x_np.shape[0] != 1:
            raise ValueError(
                f"ONNX decoder is single-sample only; expected x [1, T, D], got {x_np.shape}"
            )

        values = {
            "x": x,
            "t": t,
            "context": context,
            "context_mask": context_mask,
            "latent_mask": latent_mask,
            "span_mask": span_mask,
            "valid_audio_mask": valid_audio_mask,
            "cond": cond,
        }

        if "reference_latent" in self.decoder_input_types:
            if reference_latent is None:
                raise ValueError("decoder expects reference_latent, but got None")
            values["reference_latent"] = reference_latent

        if "reference_mask" in self.decoder_input_types:
            if reference_mask is None:
                raise ValueError("decoder expects reference_mask, but got None")
            values["reference_mask"] = reference_mask

        if "speaker_emb" in self.decoder_input_types:
            if speaker_emb is None:
                raise ValueError("decoder expects speaker_emb, but got None")
            values["speaker_emb"] = speaker_emb

        if "speaker_cond_drop" in self.decoder_input_types:
            if speaker_cond_drop is None:
                speaker_cond_drop = np.zeros((1,), dtype=np.bool_)
            values["speaker_cond_drop"] = speaker_cond_drop

        if "speaker_adaln_scale" in self.decoder_input_types:
            values["speaker_adaln_scale"] = np.asarray(
                speaker_adaln_scale,
                dtype=np.float32,
            )

        feed = self._make_feed(
            self.decoder,
            values,
            self.decoder_input_types,
        )

        out = self.decoder.run(None, feed)
        name = self.decoder.get_outputs()[0].name
        return self._decode_output(out[0], self.decoder_output_types[name]).astype(np.float32)


    def _make_feed(self, session, values, types):
        feed = {}

        for inp in session.get_inputs():
            name = inp.name

            if name not in values:
                raise KeyError(
                    f"missing required ONNX input {name!r}; "
                    f"provided inputs: {sorted(values.keys())}"
                )

            typ = types[name]
            value = values[name]

            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            else:
                value = np.asarray(value)

            if typ == "tensor(int64)":
                value = value.astype(np.int64)
            elif typ == "tensor(int32)":
                value = value.astype(np.int32)
            elif typ == "tensor(bool)":
                value = value.astype(np.bool_)
            elif typ == "tensor(float16)":
                value = value.astype(np.float16)
            elif typ == "tensor(float)":
                value = value.astype(np.float32)
            elif typ == "tensor(bfloat16)":
                value = f32_to_bf16_u16(value)
            else:
                raise TypeError(f"{name}: unsupported ONNX input type {typ}")

            feed[name] = value

        return feed

    def _decode_output(self, value, typ):
        if typ == "tensor(bfloat16)":
            if value.dtype == np.uint16:
                return bf16_u16_to_f32(value)
            return value.astype(np.float32)
        if typ == "tensor(float16)":
            return value.astype(np.float32)
        if typ == "tensor(float)":
            return value.astype(np.float32)
        if typ == "tensor(bool)":
            return value.astype(np.bool_)
        return value


def prepare_onnx_sampling_inputs(
    text_ids,
    text_mask,
    latent_size,
    duration,
    cond_latents=None,
    cond_latent_mask=None,
):
    text_ids_np = as_numpy(text_ids).astype(np.int64)
    text_mask_np = as_numpy(text_mask).astype(np.bool_)

    b = text_ids_np.shape[0]
    d = int(latent_size)

    if isinstance(duration, int):
        dur = np.full((b,), duration, dtype=np.int64)
    else:
        dur = as_numpy(duration).astype(np.int64).reshape(-1)

    t_total = int(dur.max())
    valid = np.arange(t_total)[None, :] < dur[:, None]

    cond = np.zeros((b, t_total, d), dtype=np.float32)
    cond_mask = np.zeros((b, t_total), dtype=np.bool_)

    if cond_latents is not None:
        cl = as_numpy(cond_latents).astype(np.float32)
        lp = min(cl.shape[1], t_total)
        cond[:, :lp, :] = cl[:, :lp, :]

        if cond_latent_mask is None:
            cm = np.ones((b, lp), dtype=np.bool_)
        else:
            cm = as_numpy(cond_latent_mask).astype(np.bool_)[:, :lp]

        cond_mask[:, :lp] = cm & valid[:, :lp]

    span_mask = valid & (~cond_mask)

    return {
        "text_ids": text_ids_np,
        "text_mask": text_mask_np,
        "duration": dur,
        "valid": valid.astype(np.bool_),
        "cond": cond,
        "cond_mask": cond_mask.astype(np.bool_),
        "span_mask": span_mask.astype(np.bool_),
    }


def sample_midpoint_reducio_onnx(
    core,
    text_ids,
    text_mask,
    latent_size,
    duration,
    cond_latents=None,
    cond_latent_mask=None,
    steps=16,
    cfg=2.0,
    cfg_min_t=0.0,
    cfg_max_t=1.0,
    seed=None,
    ts_fraction=0.2,
    bs_fraction=0.2,
    torch_output_device="cpu",
    *,
    init_cond=False,
    allow_batch=False,
):
    data = prepare_onnx_sampling_inputs(
        text_ids=text_ids,
        text_mask=text_mask,
        latent_size=latent_size,
        duration=duration,
        cond_latents=cond_latents,
        cond_latent_mask=cond_latent_mask,
    )

    text_ids_np = data["text_ids"]
    text_mask_np = data["text_mask"]
    valid = data["valid"]
    cond = data["cond"]
    cond_mask = data["cond_mask"]
    span_mask = data["span_mask"]

    b, t_total, d = cond.shape
    _require_batch_size_one(b, "sample_midpoint_reducio_onnx", allow_batch=allow_batch)
    cfg_min_t, cfg_max_t = _validate_cfg_window(cfg_min_t, cfg_max_t)

    context, context_mask = core.encode(
        text_ids_np,
        text_mask_np,
        text_cond_drop=np.zeros((b,), dtype=np.bool_),
    )
    null_context_mask = np.zeros_like(context_mask, dtype=np.bool_)

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))

    x = torch.randn((b, t_total, d), generator=gen, dtype=torch.float32).numpy()
    x = x * valid[:, :, None].astype(np.float32)

    if init_cond:
        x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)

    n_ts = int(steps * ts_fraction)
    n_bs = int(steps * bs_fraction)
    ts_start = steps - n_ts
    bs_start = ts_start - n_bs

    prev_v = None
    prev_v_window_state = None
    prev_v_cond = None
    prev_v_null = None

    times = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)

    for i in range(int(steps)):
        t0 = times[i]
        t1 = times[i + 1]
        dt = np.float32(t1 - t0)
        cfg_t0 = float(cfg) > 0.0 and _cfg_timestep_enabled(t0, cfg_min_t, cfg_max_t)
        cfg_t1 = float(cfg) > 0.0 and _cfg_timestep_enabled(t1, cfg_min_t, cfg_max_t)
        window_state = (cfg_t0, cfg_t1)

        if (
            i >= ts_start
            and prev_v is not None
            and prev_v_window_state == window_state
        ):
            v = prev_v
        else:
            t0_arr = np.full((b,), float(t0), dtype=np.float32)
            v1_cond = core.decode_velocity(
                x=x,
                t=t0_arr,
                context=context,
                context_mask=context_mask,
                latent_mask=valid,
                span_mask=span_mask,
                valid_audio_mask=valid,
                cond=cond,
            )

            if cfg_t0:
                if (
                    i >= bs_start
                    and i < ts_start
                    and prev_v_cond is not None
                    and prev_v_null is not None
                ):
                    v1_null = prev_v_null - prev_v_cond + v1_cond
                else:
                    v1_null = core.decode_velocity(
                        x=x,
                        t=t0_arr,
                        context=context,
                        context_mask=null_context_mask,
                        latent_mask=valid,
                        span_mask=span_mask,
                        valid_audio_mask=valid,
                        cond=cond,
                    )

                v1 = v1_cond + np.float32(cfg) * (v1_cond - v1_null)
                prev_v_cond = v1_cond
                prev_v_null = v1_null
            else:
                v1 = v1_cond

            x_mid = (x + dt * v1) * valid[:, :, None].astype(np.float32)
            x_mid = np.where(cond_mask[:, :, None], cond, x_mid).astype(np.float32)

            t1_arr = np.full((b,), float(t1), dtype=np.float32)
            v2_cond = core.decode_velocity(
                x=x_mid,
                t=t1_arr,
                context=context,
                context_mask=context_mask,
                latent_mask=valid,
                span_mask=span_mask,
                valid_audio_mask=valid,
                cond=cond,
            )

            if cfg_t1:
                v2_null = core.decode_velocity(
                    x=x_mid,
                    t=t1_arr,
                    context=context,
                    context_mask=null_context_mask,
                    latent_mask=valid,
                    span_mask=span_mask,
                    valid_audio_mask=valid,
                    cond=cond,
                )
                v2 = v2_cond + np.float32(cfg) * (v2_cond - v2_null)
            else:
                v2 = v2_cond

            v = np.float32(0.5) * (v1 + v2)
            prev_v = v
            prev_v_window_state = window_state

        x = (x + dt * v) * valid[:, :, None].astype(np.float32)
        x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)

    x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)
    return torch.from_numpy(x).to(torch_output_device)


def sample_euler_reducio_onnx(
    core,
    text_ids,
    text_mask,
    latent_size,
    duration,
    cond_latents=None,
    cond_latent_mask=None,
    steps=32,
    cfg=2.0,
    cfg_min_t=0.0,
    cfg_max_t=1.0,
    seed=None,
    ts_fraction=0.2,
    bs_fraction=0.2,
    torch_output_device="cpu",
    *,
    init_cond=False,
    allow_batch=False,
    apg_eta=None,
    apg_momentum=None,
    apg_norm=None,
):
    data = prepare_onnx_sampling_inputs(
        text_ids=text_ids,
        text_mask=text_mask,
        latent_size=latent_size,
        duration=duration,
        cond_latents=cond_latents,
        cond_latent_mask=cond_latent_mask,
    )

    text_ids_np = data["text_ids"]
    text_mask_np = data["text_mask"]
    valid = data["valid"]
    cond = data["cond"]
    cond_mask = data["cond_mask"]
    span_mask = data["span_mask"]

    b, t_total, d = cond.shape
    _require_batch_size_one(b, "sample_euler_reducio_onnx", allow_batch=allow_batch)
    cfg_min_t, cfg_max_t = _validate_cfg_window(cfg_min_t, cfg_max_t)

    context, context_mask = core.encode(
        text_ids_np,
        text_mask_np,
        text_cond_drop=np.zeros((b,), dtype=np.bool_),
    )
    null_context_mask = np.zeros_like(context_mask, dtype=np.bool_)

    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
    else:
        gen = None

    x = torch.randn((b, t_total, d), generator=gen, dtype=torch.float32).numpy()
    x = x * valid[:, :, None].astype(np.float32)

    if init_cond:
        x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)

    n_ts = int(steps * ts_fraction)
    n_bs = int(steps * bs_fraction)
    ts_start = steps - n_ts
    bs_start = ts_start - n_bs

    prev_v = None
    prev_v_has_cfg = None
    prev_v_cond = None
    prev_v_null = None
    apg_buffer = None
    use_apg = apg_eta is not None

    times = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)

    for i in range(int(steps)):
        t_value = times[i]
        t = np.full((b,), float(t_value), dtype=np.float32)
        dt = np.float32(times[i + 1] - times[i])
        has_cfg = float(cfg) > 0.0 and _cfg_timestep_enabled(
            t_value, cfg_min_t, cfg_max_t
        )

        if i >= ts_start and prev_v is not None and prev_v_has_cfg == has_cfg:
            v = prev_v
        else:
            v_cond = core.decode_velocity(
                x=x,
                t=t,
                context=context,
                context_mask=context_mask,
                latent_mask=valid,
                span_mask=span_mask,
                valid_audio_mask=valid,
                cond=cond,
            )

            if has_cfg:
                if (
                    i >= bs_start
                    and i < ts_start
                    and prev_v_cond is not None
                    and prev_v_null is not None
                ):
                    v_null = prev_v_null - prev_v_cond + v_cond
                else:
                    v_null = core.decode_velocity(
                        x=x,
                        t=t,
                        context=context,
                        context_mask=null_context_mask,
                        latent_mask=valid,
                        span_mask=span_mask,
                        valid_audio_mask=valid,
                        cond=cond,
                    )

                if use_apg:
                    v, apg_buffer = _apg_guidance_single_np(
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
                    v = v_cond + np.float32(cfg) * (v_cond - v_null)

                prev_v_cond = v_cond
                prev_v_null = v_null
            else:
                v = v_cond

            prev_v = v.astype(np.float32)
            prev_v_has_cfg = has_cfg

        x = (x + dt * v) * valid[:, :, None].astype(np.float32)
        x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)

    x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)
    return torch.from_numpy(x).to(torch_output_device)


def sample_euler_reducio_onnx_spk(
    core,
    text_ids,
    text_mask,
    latent_size,
    duration,
    cond_latents=None,
    cond_latent_mask=None,
    *,
    reference_latent=None,
    reference_mask=None,
    speaker_emb=None,
    speaker_adaln_scale=1.0,
    steps=32,
    cfg=2.0,
    speaker_cfg=None,
    cfg_min_t=0.0,
    cfg_max_t=1.0,
    seed=None,
    ts_fraction=0.2,
    bs_fraction=0.2,
    torch_output_device="cpu",
    init_cond=False,
    apg_eta_text=None,
    apg_eta_speaker=None,
    apg_momentum_text=None,
    apg_momentum_speaker=None,
    apg_norm_text=None,
    apg_norm_speaker=None,
):
    if speaker_cfg is None:
        speaker_cfg = 0.0

    def _as_np(x, dtype=None):
        if x is None:
            return None
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        if dtype is not None:
            x = x.astype(dtype)
        return x

    data = prepare_onnx_sampling_inputs(
        text_ids=text_ids,
        text_mask=text_mask,
        latent_size=latent_size,
        duration=duration,
        cond_latents=cond_latents,
        cond_latent_mask=cond_latent_mask,
    )

    text_ids_np = data["text_ids"]
    text_mask_np = data["text_mask"]
    valid = data["valid"]
    cond = data["cond"]
    cond_mask = data["cond_mask"]
    span_mask = data["span_mask"]

    b, t_total, d = cond.shape
    if b != 1:
        raise ValueError(
            f"ONNX sampler is single-sample only; expected batch size 1, got {b}"
        )
    cfg_min_t, cfg_max_t = _validate_cfg_window(cfg_min_t, cfg_max_t)

    reference_latent_np = None
    reference_mask_np = None

    if reference_latent is not None:
        reference_latent_np = _as_np(reference_latent, np.float32)

        if reference_latent_np.ndim == 1:
            reference_latent_np = reference_latent_np[None, None, :]
        elif reference_latent_np.ndim == 2:
            if reference_latent_np.shape[0] == 1:
                reference_latent_np = reference_latent_np[:, None, :]
            else:
                reference_latent_np = reference_latent_np[None, :, :]
        elif reference_latent_np.ndim == 3:
            if reference_latent_np.shape[0] != 1:
                raise ValueError(
                    "ONNX sampler is single-sample only; "
                    f"expected reference_latent batch 1, got {reference_latent_np.shape[0]}"
                )
        else:
            raise ValueError(
                "expected reference_latent [D_ref], [1, D_ref], "
                "[T_ref, D_ref], or [1, T_ref, D_ref], "
                f"got {tuple(reference_latent_np.shape)}"
            )

        if reference_mask is None:
            reference_mask_np = np.ones(
                (1, reference_latent_np.shape[1]),
                dtype=np.bool_,
            )
        else:
            reference_mask_np = _as_np(reference_mask, np.bool_)

            if reference_mask_np.ndim == 0:
                reference_mask_np = reference_mask_np.reshape(1, 1)
            elif reference_mask_np.ndim == 1:
                if reference_mask_np.shape[0] == 1:
                    reference_mask_np = reference_mask_np.reshape(1, 1)
                elif reference_mask_np.shape[0] == reference_latent_np.shape[1]:
                    reference_mask_np = reference_mask_np[None, :]
                else:
                    raise ValueError(
                        f"reference_mask length {reference_mask_np.shape[0]} "
                        f"does not match reference length {reference_latent_np.shape[1]}"
                    )
            elif reference_mask_np.ndim == 2:
                if reference_mask_np.shape[0] != 1:
                    raise ValueError(
                        "ONNX sampler is single-sample only; "
                        f"expected reference_mask batch 1, got {reference_mask_np.shape[0]}"
                    )
            else:
                raise ValueError(
                    f"expected reference_mask [T_ref] or [1, T_ref], "
                    f"got {tuple(reference_mask_np.shape)}"
                )

            if reference_mask_np.shape != (1, reference_latent_np.shape[1]):
                raise ValueError(
                    f"expected reference_mask shape {(1, reference_latent_np.shape[1])}, "
                    f"got {tuple(reference_mask_np.shape)}"
                )

    speaker_emb_np = None
    if speaker_emb is not None:
        speaker_emb_np = _as_np(speaker_emb, np.float32)

        if speaker_emb_np.ndim == 1:
            speaker_emb_np = speaker_emb_np[None, :]

        if speaker_emb_np.ndim != 2:
            raise ValueError(
                f"expected speaker_emb [D_spk] or [1, D_spk], got {tuple(speaker_emb_np.shape)}"
            )

        if speaker_emb_np.shape[0] != 1:
            raise ValueError(
                "ONNX sampler is single-sample only; "
                f"expected speaker_emb batch 1, got {speaker_emb_np.shape[0]}"
            )

    use_dual_cfg = float(speaker_cfg) > 0.0 and speaker_emb_np is not None
    use_dual_apg = use_dual_cfg and (
        apg_eta_text is not None or apg_eta_speaker is not None
    )
    use_single_apg = (not use_dual_cfg) and apg_eta_text is not None

    if apg_eta_text is None:
        apg_eta_text = 1.0
    if apg_eta_speaker is None:
        apg_eta_speaker = 1.0

    context, context_mask = core.encode(
        text_ids_np,
        text_mask_np,
        text_cond_drop=np.zeros((1,), dtype=np.bool_),
    )
    null_context_mask = np.zeros_like(context_mask, dtype=np.bool_)

    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
    else:
        gen = None

    x = torch.randn(
        (1, t_total, d),
        generator=gen,
        dtype=torch.float32,
    ).numpy()
    x = x * valid[:, :, None].astype(np.float32)

    if init_cond:
        x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)

    speaker_drop_mask = np.ones((1,), dtype=np.bool_)

    n_ts = int(steps * ts_fraction)
    n_bs = int(steps * bs_fraction)
    ts_start = steps - n_ts
    bs_start = ts_start - n_bs

    prev_v = None
    prev_v_has_guidance = None
    prev_v_full = None
    prev_v_text = None
    prev_v_null = None
    prev_v_speaker = None
    apg_buffer_text = None
    apg_buffer_speaker = None

    times = np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)

    def decode_branch(x, t, branch_context_mask, speaker_cond_drop):
        return core.decode_velocity(
            x=x,
            t=t,
            context=context,
            context_mask=branch_context_mask,
            latent_mask=valid,
            span_mask=span_mask,
            valid_audio_mask=valid,
            cond=cond,
            reference_latent=reference_latent_np,
            reference_mask=reference_mask_np,
            speaker_emb=speaker_emb_np,
            speaker_cond_drop=speaker_cond_drop,
            speaker_adaln_scale=np.asarray(speaker_adaln_scale, dtype=np.float32),
        ).astype(np.float32)

    for i in range(int(steps)):
        t_value = times[i]
        t = np.full((1,), float(t_value), dtype=np.float32)
        dt = np.float32(times[i + 1] - times[i])
        in_cfg_window = _cfg_timestep_enabled(t_value, cfg_min_t, cfg_max_t)
        has_guidance = in_cfg_window and (
            float(cfg) > 0.0 or use_dual_cfg
        )

        if (
            i >= ts_start
            and prev_v is not None
            and prev_v_has_guidance == has_guidance
        ):
            v = prev_v
        else:
            v_full = decode_branch(
                x=x,
                t=t,
                branch_context_mask=context_mask,
                speaker_cond_drop=None,
            )

            if has_guidance and use_dual_cfg:
                if (
                    i >= bs_start
                    and i < ts_start
                    and prev_v_full is not None
                    and prev_v_text is not None
                ):
                    v_text = prev_v_text - prev_v_full + v_full
                else:
                    v_text = decode_branch(
                        x=x,
                        t=t,
                        branch_context_mask=context_mask,
                        speaker_cond_drop=speaker_drop_mask,
                    )

                if use_dual_apg:
                    if (
                        i >= bs_start
                        and i < ts_start
                        and prev_v_full is not None
                        and prev_v_speaker is not None
                    ):
                        v_speaker = prev_v_speaker - prev_v_full + v_full
                    else:
                        v_speaker = decode_branch(
                            x=x,
                            t=t,
                            branch_context_mask=null_context_mask,
                            speaker_cond_drop=None,
                        )

                    v, apg_buffer_text, apg_buffer_speaker = _apg_guidance_dual_np(
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
                    if (
                        i >= bs_start
                        and i < ts_start
                        and prev_v_full is not None
                        and prev_v_null is not None
                    ):
                        v_null = prev_v_null - prev_v_full + v_full
                    else:
                        v_null = decode_branch(
                            x=x,
                            t=t,
                            branch_context_mask=null_context_mask,
                            speaker_cond_drop=speaker_drop_mask,
                        )

                    v = (
                        v_full
                        + np.float32(cfg) * (v_text - v_null)
                        + np.float32(speaker_cfg) * (v_full - v_text)
                    ).astype(np.float32)
                    prev_v_null = v_null

                prev_v_full = v_full
                prev_v_text = v_text

            elif has_guidance and float(cfg) > 0.0:
                if (
                    i >= bs_start
                    and i < ts_start
                    and prev_v_full is not None
                    and prev_v_null is not None
                ):
                    v_null = prev_v_null - prev_v_full + v_full
                else:
                    v_null = decode_branch(
                        x=x,
                        t=t,
                        branch_context_mask=null_context_mask,
                        speaker_cond_drop=speaker_drop_mask,
                    )

                if use_single_apg:
                    v, apg_buffer_text = _apg_guidance_single_np(
                        x=x,
                        t=t,
                        v_cond=v_full,
                        v_uncond=v_null,
                        scale=cfg,
                        eta=apg_eta_text,
                        momentum=apg_momentum_text,
                        norm=apg_norm_text,
                        buffer=apg_buffer_text,
                    )
                else:
                    v = v_full + np.float32(cfg) * (v_full - v_null)

                prev_v_full = v_full
                prev_v_null = v_null
            else:
                v = v_full
                prev_v_full = v_full

            prev_v = v.astype(np.float32)
            prev_v_has_guidance = has_guidance

        x = (x + dt * v) * valid[:, :, None].astype(np.float32)
        x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)

    x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)
    return torch.from_numpy(x).to(torch_output_device)


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

    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device, dtype=torch.bool)
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


    cond_latents = cond_latents.to(device=device, dtype=model_dtype)

    if cond_latent_mask is None:
        cond_latent_mask = torch.ones(
            cond_latents.shape[:2], device=device, dtype=torch.bool
        )
    else:
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


def build_latent_edit_condition_onnx(
    original_lats,
    wav_num_samples,
    wav_sr,
    parts_to_edit,
    *,
    extra_duration=None,
    duration_scale=1.0,
    padding_sec=0.5,
    min_edit_sec=0.10,
    codec_rate_hz=None,
    dtype=np.float32,
):
    original_lats = as_numpy(original_lats).astype(dtype, copy=False)

    if original_lats.ndim == 3:
        if original_lats.shape[0] != 1:
            raise ValueError(
                f"expected original_lats [T, D] or [1, T, D], got {tuple(original_lats.shape)}"
            )
        original_lats = original_lats[0]

    if original_lats.ndim != 2:
        raise ValueError(
            f"expected original_lats [T, D], got {tuple(original_lats.shape)}"
        )

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

        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()

        if isinstance(value, np.ndarray):
            value = value.tolist()

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

        raw_start_sec = max(0.0, raw_start_sec)
        raw_end_sec = min(audio_dur_sec, raw_end_sec)

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
            keep_chunks.append(np.ones((keep_len,), dtype=np.bool_))
            out_offset += keep_len

        cond_chunks.append(
            np.zeros((new_region_frames, D), dtype=dtype)
        )
        keep_chunks.append(
            np.zeros((new_region_frames,), dtype=np.bool_)
        )

        edit_ranges_out.append([out_offset, out_offset + new_region_frames])
        out_offset += new_region_frames

        src_offset = end_frame

    if src_offset < T_src:
        tail_len = T_src - src_offset
        cond_chunks.append(original_lats[src_offset:])
        keep_chunks.append(np.ones((tail_len,), dtype=np.bool_))

    cond = np.concatenate(cond_chunks, axis=0)[None, :, :].astype(np.float32)
    keep_mask = np.concatenate(keep_chunks, axis=0)[None, :].astype(np.bool_)

    valid = np.ones_like(keep_mask, dtype=np.bool_)
    span_mask = valid & (~keep_mask)

    return cond, keep_mask, span_mask, valid, edit_ranges_out


def prepare_onnx_edit_sampling_inputs(
    text_ids,
    text_mask,
    cond,
    keep_mask,
    valid_mask=None,
):
    text_ids_np = as_numpy(text_ids).astype(np.int64)
    text_mask_np = as_numpy(text_mask).astype(np.bool_)

    if text_ids_np.ndim == 1:
        text_ids_np = text_ids_np[None, :]

    if text_mask_np.ndim == 1:
        text_mask_np = text_mask_np[None, :]

    cond_np = as_numpy(cond).astype(np.float32)

    if cond_np.ndim == 2:
        cond_np = cond_np[None, :, :]

    if cond_np.ndim != 3:
        raise ValueError(f"expected cond [B, T, D] or [T, D], got {tuple(cond_np.shape)}")

    keep_mask_np = as_numpy(keep_mask).astype(np.bool_)

    if keep_mask_np.ndim == 1:
        keep_mask_np = keep_mask_np[None, :]

    b, t_total, d = cond_np.shape

    if keep_mask_np.shape != (b, t_total):
        if keep_mask_np.shape[0] == 1 and b > 1 and keep_mask_np.shape[1] == t_total:
            keep_mask_np = np.broadcast_to(keep_mask_np, (b, t_total)).copy()
        else:
            raise ValueError(
                f"keep_mask shape {tuple(keep_mask_np.shape)} does not match cond shape {(b, t_total, d)}"
            )

    if text_ids_np.shape[0] == 1 and b > 1:
        text_ids_np = np.broadcast_to(text_ids_np, (b, text_ids_np.shape[1])).copy()

    if text_mask_np.shape[0] == 1 and b > 1:
        text_mask_np = np.broadcast_to(text_mask_np, (b, text_mask_np.shape[1])).copy()

    if text_ids_np.shape[0] != b:
        raise ValueError(
            f"text batch {text_ids_np.shape[0]} does not match cond batch {b}"
        )

    if valid_mask is None:
        valid_np = np.ones((b, t_total), dtype=np.bool_)
    else:
        valid_np = as_numpy(valid_mask).astype(np.bool_)

        if valid_np.ndim == 1:
            valid_np = valid_np[None, :]

        if valid_np.shape != (b, t_total):
            if valid_np.shape[0] == 1 and b > 1 and valid_np.shape[1] == t_total:
                valid_np = np.broadcast_to(valid_np, (b, t_total)).copy()
            else:
                raise ValueError(
                    f"valid_mask shape {tuple(valid_np.shape)} does not match cond shape {(b, t_total, d)}"
                )

    cond_mask_np = keep_mask_np & valid_np
    span_mask_np = valid_np & (~cond_mask_np)

    return {
        "text_ids": text_ids_np,
        "text_mask": text_mask_np,
        "valid": valid_np.astype(np.bool_),
        "cond": cond_np.astype(np.float32),
        "cond_mask": cond_mask_np.astype(np.bool_),
        "span_mask": span_mask_np.astype(np.bool_),
    }


def sample_euler_reducio_edit_onnx(
    core,
    text_ids,
    text_mask,
    *,
    cond,
    keep_mask,
    valid_mask=None,
    steps=32,
    cfg=2.0,
    cfg_min_t=0.0,
    cfg_max_t=1.0,
    seed=None,
    ts_fraction=0.2,
    bs_fraction=0.2,
    torch_output_device="cpu",
    init_cond=False,
    allow_batch=False,
    apg_eta=None,
    apg_momentum=None,
    apg_norm=None,
):
    data = prepare_onnx_edit_sampling_inputs(
        text_ids=text_ids,
        text_mask=text_mask,
        cond=cond,
        keep_mask=keep_mask,
        valid_mask=valid_mask,
    )

    text_ids_np = data["text_ids"]
    text_mask_np = data["text_mask"]
    valid = data["valid"]
    cond = data["cond"]
    cond_mask = data["cond_mask"]
    span_mask = data["span_mask"]

    b, t_total, d = cond.shape
    _require_batch_size_one(b, "sample_euler_reducio_edit_onnx", allow_batch=allow_batch)
    cfg_min_t, cfg_max_t = _validate_cfg_window(cfg_min_t, cfg_max_t)

    context, context_mask = core.encode(
        text_ids_np,
        text_mask_np,
        text_cond_drop=np.zeros((b,), dtype=np.bool_),
    )
    null_context_mask = np.zeros_like(context_mask, dtype=np.bool_)

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(int(seed))

    x = torch.randn(
        (b, t_total, d),
        generator=gen,
        dtype=torch.float32,
    ).numpy()
    x = x * valid[:, :, None].astype(np.float32)

    if init_cond:
        x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)

    n_ts = int(steps * ts_fraction)
    n_bs = int(steps * bs_fraction)
    ts_start = int(steps) - n_ts
    bs_start = ts_start - n_bs

    prev_v = None
    prev_v_has_cfg = None
    prev_v_cond = None
    prev_v_null = None
    apg_buffer = None
    use_apg = apg_eta is not None

    times = np.linspace(0.0, 1.0, int(steps) + 1, dtype=np.float32)

    for i in range(int(steps)):
        t_value = times[i]
        t = np.full((b,), float(t_value), dtype=np.float32)
        dt = np.float32(times[i + 1] - times[i])
        has_cfg = float(cfg) > 0.0 and _cfg_timestep_enabled(
            t_value, cfg_min_t, cfg_max_t
        )

        if i >= ts_start and prev_v is not None and prev_v_has_cfg == has_cfg:
            v = prev_v
        else:
            v_cond = core.decode_velocity(
                x=x,
                t=t,
                context=context,
                context_mask=context_mask,
                latent_mask=valid,
                span_mask=span_mask,
                valid_audio_mask=valid,
                cond=cond,
            )

            if has_cfg:
                if (
                    i >= bs_start
                    and i < ts_start
                    and prev_v_cond is not None
                    and prev_v_null is not None
                ):
                    v_null = prev_v_null - prev_v_cond + v_cond
                else:
                    v_null = core.decode_velocity(
                        x=x,
                        t=t,
                        context=context,
                        context_mask=null_context_mask,
                        latent_mask=valid,
                        span_mask=span_mask,
                        valid_audio_mask=valid,
                        cond=cond,
                    )

                if use_apg:
                    v, apg_buffer = _apg_guidance_single_np(
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
                    v = v_cond + np.float32(cfg) * (v_cond - v_null)

                prev_v_cond = v_cond
                prev_v_null = v_null
            else:
                v = v_cond

            prev_v = v.astype(np.float32)
            prev_v_has_cfg = has_cfg

        x = (x + dt * v) * valid[:, :, None].astype(np.float32)
        x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)

    x = np.where(cond_mask[:, :, None], cond, x).astype(np.float32)
    x = x * valid[:, :, None].astype(np.float32)

    return torch.from_numpy(x).to(torch_output_device)


class Extractor:
    def __init__(self, tokenizer, device, duration_model, speaker_model):
        self.tokenizer = tokenizer
        self.max_length = 512
        self.max_duration = int(30 * 12.5)
        self.device = device
        self.duration_model = duration_model
        self.speaker_model = speaker_model

    def get_tokens(self, text, duration_text=None):
        if duration_text is None:
            duration_text = text

        t = self.tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        dt = self.tokenizer(
            [duration_text],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        return (
            t["input_ids"].to(self.device).long(),
            t["attention_mask"].to(self.device).bool(),
            dt["input_ids"].to(self.device).long(),
            dt["attention_mask"].to(self.device).bool(),
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
                        f"non-batched Extractor expects cond_latent_mask batch 1, got {cond_latent_mask.shape[0]}"
                    )

                prompt_len = int(cond_latent_mask.to(torch.bool).sum().item())
            else:
                prompt_len = int(cond_latents.shape[1])

            pred_min_total_frames = prompt_len + 1
        else:
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
        )

        duration = (pred.float() / float(speed)).round().long()
        duration = duration.clamp(
            min=pred_min_total_frames,
            max=max_total_frames,
        )

        return duration

    def get_speaker_embedding(self, audio):
        if self.speaker_model is None:
            raise ValueError("speaker_model is None")

        if isinstance(audio, np.ndarray):
            audio = librosa.resample(audio, orig_sr=44100, target_sr=16000)
            emb, _ = self.speaker_model.infer_segment(audio)
        else:
            emb = self.speaker_model.get_embedding(audio)

        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        elif not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)

        emb = emb.to(self.device, dtype=torch.float32)

        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        elif emb.ndim == 2 and emb.shape[0] == 1:
            pass
        else:
            raise ValueError(f"Unexpected speaker embedding shape: {tuple(emb.shape)}")

        return emb