import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


logger = logging.getLogger(__name__)


class ExportSettings:
    def __init__(self, **values):
        self.__dict__.update(values)


def _ensure_repo(repo_root):
    root = Path(repo_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(str(root))

    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)

    return root


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _checkpoint_path(ckpt_dir):
    path = Path(ckpt_dir).expanduser().resolve()
    if path.is_file():
        return path

    extra_state = path / "extra_state.pt"
    if extra_state.exists():
        return extra_state

    raise FileNotFoundError(str(extra_state))


def _load_state_dict(checkpoint):
    for key in ("ema", "model", "model_state_dict"):
        if key in checkpoint:
            return checkpoint[key], key

    keys = ", ".join(sorted(checkpoint)) if isinstance(checkpoint, dict) else type(checkpoint).__name__
    raise KeyError(f"checkpoint has no model weights; available keys: {keys}")


def _clone_export_cfg(cfg, text_attn_implementation):
    export_cfg = copy.deepcopy(cfg)
    export_cfg["model"]["decoder"]["use_flash_attn"] = False
    export_cfg["model"]["text_encoder"]["attn_implementation"] = text_attn_implementation
    return export_cfg


def _disable_flash_attention(model):
    for module in model.modules():
        if hasattr(module, "use_flash_attn"):
            module.use_flash_attn = False


def _load_model(repo_root, config_path, ckpt_dir, settings):
    _ensure_repo(repo_root)
    from model_transformer import model_from_config

    cfg = _clone_export_cfg(
        _load_json(config_path),
        settings.text_attn_implementation,
    )
    model = model_from_config(
        cfg,
        use_speaker_conditioning=settings.use_speaker_conditioning,
        speaker_emb_dim=settings.speaker_emb_dim,
    )

    ckpt_path = _checkpoint_path(ckpt_dir)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state, state_key = _load_state_dict(checkpoint)

    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device=settings.device, dtype=settings.dtype)
    _disable_flash_attention(model)

    return model, cfg, checkpoint, ckpt_path, state_key


def _make_encoder_decoder(model, use_speaker_conditioning):
    from nb_tts_core_full_wrappers import (
        TTSFullDecoderDenoiserONNX,
        TTSFullDecoderDenoiserWithSpeakerONNX,
        TTSFullEncoderNoRefONNX,
    )

    encoder = TTSFullEncoderNoRefONNX(model).eval()
    decoder_cls = (
        TTSFullDecoderDenoiserWithSpeakerONNX
        if use_speaker_conditioning
        else TTSFullDecoderDenoiserONNX
    )

    return encoder, decoder_cls(model).eval()


def _vocab_size(cfg, model):
    text_encoder = getattr(model, "text_encoder", None)
    bert = getattr(text_encoder, "bert", None)
    config = getattr(bert, "config", None)
    vocab_size = getattr(config, "vocab_size", None)

    if vocab_size is not None:
        return int(vocab_size)

    return int(cfg["model"]["text_encoder"]["vocab_size"])


def _make_dummy_inputs(model, cfg, settings):
    device = settings.device
    dtype = settings.dtype
    batch_size = int(settings.dummy_batch)
    text_len = int(settings.dummy_text_len)
    audio_len = int(settings.dummy_audio_len)
    speaker_dim = int(settings.speaker_emb_dim)

    latent_dim = int(model.latent_size)
    prompt_len = max(1, min(audio_len - 1, audio_len // 4))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)

    text_ids = torch.randint(
        0,
        _vocab_size(cfg, model),
        (batch_size, text_len),
        generator=generator,
        dtype=torch.long,
    ).to(device)

    text_mask = torch.ones((batch_size, text_len), dtype=torch.bool, device=device)
    text_cond_drop = torch.zeros((batch_size,), dtype=torch.bool, device=device)

    x = torch.randn(
        (batch_size, audio_len, latent_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)

    cond = torch.zeros(
        (batch_size, audio_len, latent_dim),
        device=device,
        dtype=dtype,
    )
    prompt = torch.randn(
        (batch_size, prompt_len, latent_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    cond[:, :prompt_len] = prompt

    latent_mask = torch.ones(
        (batch_size, audio_len),
        dtype=torch.bool,
        device=device,
    )
    valid_audio_mask = torch.ones_like(latent_mask)
    cond_mask = torch.zeros_like(latent_mask)
    cond_mask[:, :prompt_len] = True
    span_mask = valid_audio_mask & ~cond_mask

    inputs = {
        "text_ids": text_ids,
        "text_mask": text_mask,
        "text_cond_drop": text_cond_drop,
        "x": x,
        "t": torch.full(
            (batch_size,),
            0.5,
            device=device,
            dtype=dtype,
        ),
        "cond": cond,
        "latent_mask": latent_mask,
        "span_mask": span_mask,
        "valid_audio_mask": valid_audio_mask,
    }

    if settings.use_speaker_conditioning:
        inputs["speaker_emb"] = torch.randn(
            (batch_size, speaker_dim),
            generator=generator,
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        inputs["speaker_cond_drop"] = torch.zeros(
            (batch_size,), dtype=torch.bool, device=device
        )

    return inputs


def _float_compare(name, actual, expected, precision, fail):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    diff = (actual - expected).abs()

    if precision == "bf16":
        atol = 0.40
        rtol = 0.40
    elif precision == "fp16":
        atol = 0.30
        rtol = 0.30
    else:
        atol = 1e-3
        rtol = 1e-3

    tolerance = atol + rtol * expected.abs()
    ok = bool(torch.all(diff <= tolerance))

    denom = torch.maximum(
        expected.abs(),
        torch.full_like(expected, 1e-8),
    )
    rel = diff / denom

    result = {
        "name": name,
        "ok": ok,
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel": float(rel.mean().item()) if rel.numel() else 0.0,
        "atol": atol,
        "rtol": rtol,
    }

    if fail and not ok:
        raise AssertionError(f"validation failed for {name}")

    return result


def _bool_compare(name, actual, expected, fail):
    ok = bool(
        torch.equal(
            actual.detach().cpu().bool(),
            expected.detach().cpu().bool(),
        )
    )
    result = {
        "name": name,
        "ok": ok,
    }

    if fail and not ok:
        raise AssertionError(f"validation failed for {name}")

    return result


def _as_numpy(tensor):
    if tensor.dtype == torch.bfloat16:
        raise TypeError("bfloat16 tensors cannot be passed through numpy")

    return tensor.detach().cpu().numpy()


def _ort_session(path, provider):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=[provider],
    )


def _feed_for_session(session, tensors):
    feed = {}

    for item in session.get_inputs():
        if item.name not in tensors:
            raise KeyError(f"missing ONNX input: {item.name}")
        feed[item.name] = _as_numpy(tensors[item.name])

    return feed


def _ort_to_torch(value):
    arr = np.asarray(value)

    if arr.dtype == np.bool_:
        return torch.from_numpy(arr.astype(np.bool_))
    if np.issubdtype(arr.dtype, np.integer):
        return torch.from_numpy(arr)

    return torch.from_numpy(arr.astype(np.float32))


def _decoder_tensors(inputs, context, context_mask, use_speaker_conditioning):
    tensors = {
        "x": inputs["x"],
        "t": inputs["t"],
        "cond": inputs["cond"],
        "latent_mask": inputs["latent_mask"],
        "span_mask": inputs["span_mask"],
        "valid_audio_mask": inputs["valid_audio_mask"],
        "context": context,
        "context_mask": context_mask,
    }

    if use_speaker_conditioning:
        tensors["speaker_emb"] = inputs["speaker_emb"]
        tensors["speaker_cond_drop"] = inputs["speaker_cond_drop"]

    return tensors


def _decoder_spec(inputs, context, context_mask, use_speaker_conditioning):
    tensors = _decoder_tensors(
        inputs,
        context,
        context_mask,
        use_speaker_conditioning,
    )
    names = list(tensors)
    args = tuple(tensors[name] for name in names)

    axes = {
        "x": {0: "batch", 1: "audio_len"},
        "t": {0: "batch"},
        "cond": {0: "batch", 1: "audio_len"},
        "latent_mask": {0: "batch", 1: "audio_len"},
        "span_mask": {0: "batch", 1: "audio_len"},
        "valid_audio_mask": {0: "batch", 1: "audio_len"},
        "context": {0: "batch", 1: "context_len"},
        "context_mask": {0: "batch", 1: "context_len"},
        "velocity": {0: "batch", 1: "audio_len"},
    }

    if use_speaker_conditioning:
        axes["speaker_emb"] = {0: "batch"}
        axes["speaker_cond_drop"] = {0: "batch"}

    return tensors, names, args, axes


def _encoder_spec(inputs):
    tensors = {
        "text_ids": inputs["text_ids"],
        "text_mask": inputs["text_mask"],
        "text_cond_drop": inputs["text_cond_drop"],
    }
    names = list(tensors)
    args = tuple(tensors[name] for name in names)
    axes = {
        "text_ids": {0: "batch", 1: "text_len"},
        "text_mask": {0: "batch", 1: "text_len"},
        "text_cond_drop": {0: "batch"},
        "context": {0: "batch", 1: "text_len"},
        "context_mask": {0: "batch", 1: "text_len"},
    }

    return tensors, names, args, axes


def _validate_encoder_ort(path, encoder, inputs, settings):
    if settings.precision == "bf16":
        return [{
            "name": "encoder_ort",
            "ok": None,
            "skipped": True,
            "reason": "bf16 numpy feed/output handling skipped",
        }]

    session = _ort_session(path, settings.ort_provider)

    with torch.no_grad():
        pt_context, pt_mask = encoder(
            inputs["text_ids"],
            inputs["text_mask"],
            inputs["text_cond_drop"],
        )

    out = session.run(
        None,
        _feed_for_session(
            session,
            _encoder_spec(inputs)[0],
        ),
    )

    return [
        _float_compare(
            "encoder_context_ort",
            pt_context,
            _ort_to_torch(out[0]),
            settings.precision,
            settings.fail_on_ort_mismatch,
        ),
        _bool_compare(
            "encoder_context_mask_ort",
            pt_mask,
            _ort_to_torch(out[1]),
            settings.fail_on_ort_mismatch,
        ),
    ]


def _validate_decoder_ort(
    path,
    decoder,
    inputs,
    context,
    context_mask,
    settings,
):
    if settings.precision == "bf16":
        return [{
            "name": "decoder_ort",
            "ok": None,
            "skipped": True,
            "reason": "bf16 numpy feed/output handling skipped",
        }]

    session = _ort_session(path, settings.ort_provider)
    tensors, _, args, _ = _decoder_spec(
        inputs,
        context,
        context_mask,
        settings.use_speaker_conditioning,
    )

    with torch.no_grad():
        pt = decoder(*args)

    out = session.run(
        None,
        _feed_for_session(session, tensors),
    )

    return [
        _float_compare(
            "decoder_velocity_ort",
            pt,
            _ort_to_torch(out[0]),
            settings.precision,
            settings.fail_on_ort_mismatch,
        )
    ]


def _eager_parity(model, encoder, decoder, inputs, settings):
    with torch.no_grad():
        context, context_mask = encoder(
            inputs["text_ids"],
            inputs["text_mask"],
            inputs["text_cond_drop"],
        )

        model_context = model.encode_text(
            inputs["text_ids"],
            inputs["text_mask"],
        )
        model_context_mask = (
            inputs["text_mask"]
            & ~inputs["text_cond_drop"][:, None]
        )

        encoder_result = _float_compare(
            "eager_encoder_context",
            context,
            model_context,
            settings.precision,
            settings.fail_on_eager_mismatch,
        )
        mask_result = _bool_compare(
            "eager_encoder_context_mask",
            context_mask,
            model_context_mask,
            settings.fail_on_eager_mismatch,
        )

        _, _, decoder_args, _ = _decoder_spec(
            inputs,
            context,
            context_mask,
            settings.use_speaker_conditioning,
        )
        decoder_velocity = decoder(*decoder_args)

        full_velocity = model(
            x=inputs["x"],
            t=inputs["t"],
            text_ids=inputs["text_ids"],
            text_mask=inputs["text_mask"],
            reference_latent=None,
            reference_mask=None,
            text_cond_drop=inputs["text_cond_drop"],
            speaker_cond_drop=(
                inputs["speaker_cond_drop"]
                if settings.use_speaker_conditioning
                else None
            ),
            start_pos=0,
            use_checkpoint=False,
            latent_mask=inputs["latent_mask"],
            span_mask=inputs["span_mask"],
            valid_audio_mask=inputs["valid_audio_mask"],
            cond=inputs["cond"],
            audio_cond_drop=None,
            return_hidden_states=False,
            speaker_emb=(
                inputs["speaker_emb"]
                if settings.use_speaker_conditioning
                else None
            ),
            speaker_adaln_scale=1.0,
        )

        decoder_result = _float_compare(
            "eager_decoder_velocity_vs_model_forward",
            decoder_velocity,
            full_velocity,
            settings.precision,
            settings.fail_on_eager_mismatch,
        )

    return [
        encoder_result,
        mask_result,
        decoder_result,
    ], context, context_mask


def _export_onnx(
    module,
    args,
    path,
    input_names,
    output_names,
    dynamic_axes,
    settings,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    exporter = settings.exporter
    if exporter not in ("dynamo", "auto", "legacy"):
        raise ValueError("exporter must be 'dynamo', 'auto', or 'legacy'")

    torch.onnx.export(
        module,
        args,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=settings.opset,
        do_constant_folding=True,
        dynamo=exporter != "legacy",
        external_data=exporter != "legacy",
    )


def _export_pair(
    encoder,
    decoder,
    inputs,
    context,
    context_mask,
    out_dir,
    settings,
):
    precision = settings.precision
    encoder_path = out_dir / f"tts_encoder_{precision}.onnx"
    decoder_path = out_dir / f"tts_decoder_denoiser_{precision}.onnx"

    _, encoder_names, encoder_args, encoder_axes = _encoder_spec(inputs)
    _, decoder_names, decoder_args, decoder_axes = _decoder_spec(
        inputs,
        context,
        context_mask,
        settings.use_speaker_conditioning,
    )

    _export_onnx(
        encoder,
        encoder_args,
        encoder_path,
        encoder_names,
        ["context", "context_mask"],
        encoder_axes,
        settings,
    )
    _export_onnx(
        decoder,
        decoder_args,
        decoder_path,
        decoder_names,
        ["velocity"],
        decoder_axes,
        settings,
    )

    return encoder_path, decoder_path


def _run_ort_validation(
    encoder_path,
    decoder_path,
    encoder,
    decoder,
    inputs,
    context,
    context_mask,
    settings,
):
    results = []

    checks = [
        (
            "encoder_ort",
            lambda: _validate_encoder_ort(
                encoder_path,
                encoder,
                inputs,
                settings,
            ),
        ),
        (
            "decoder_ort",
            lambda: _validate_decoder_ort(
                decoder_path,
                decoder,
                inputs,
                context,
                context_mask,
                settings,
            ),
        ),
    ]

    for name, check in checks:
        try:
            results.extend(check())
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            if settings.fail_on_ort_mismatch:
                raise

            logger.exception("%s validation failed", name)
            results.append({
                "name": name,
                "ok": False,
                "exception": repr(exc),
            })

    return results


def _prepare_export(model, cfg, out_dir, settings):
    encoder, decoder = _make_encoder_decoder(
        model,
        settings.use_speaker_conditioning,
    )
    inputs = _make_dummy_inputs(
        model,
        cfg,
        settings,
    )

    eager_results = []

    if settings.run_eager_parity:
        eager_results, context, context_mask = _eager_parity(
            model,
            encoder,
            decoder,
            inputs,
            settings,
        )
    else:
        with torch.no_grad():
            context, context_mask = encoder(
                inputs["text_ids"],
                inputs["text_mask"],
                inputs["text_cond_drop"],
            )

    encoder_path, decoder_path = _export_pair(
        encoder,
        decoder,
        inputs,
        context,
        context_mask,
        out_dir,
        settings,
    )

    checker_results = []
    if settings.run_onnx_checker:
        onnx.checker.check_model(str(encoder_path))
        onnx.checker.check_model(str(decoder_path))
        checker_results = [
            {"name": "encoder_checker", "ok": True},
            {"name": "decoder_checker", "ok": True},
        ]

    ort_results = []
    if settings.validate_ort:
        ort_results = _run_ort_validation(
            encoder_path,
            decoder_path,
            encoder,
            decoder,
            inputs,
            context,
            context_mask,
            settings,
        )

    return {
        "encoder_path": encoder_path,
        "decoder_path": decoder_path,
        "eager_validation": eager_results,
        "onnx_checker": checker_results,
        "ort_validation": ort_results,
    }


def _resolve_precision(precision):
    plans = {
        "auto": ("fp32", torch.float32),
        "fp32": ("fp32", torch.float32),
        "bf16": ("bf16", torch.bfloat16),
        "fp16": ("fp16", torch.float16),
    }

    if precision not in plans:
        raise ValueError("precision must be 'auto', 'fp32', 'bf16', or 'fp16'")

    return plans[precision]


def _build_settings(
    *,
    device,
    precision,
    use_speaker_conditioning,
    speaker_emb_dim,
    text_attn_implementation,
    dummy_batch,
    dummy_text_len,
    dummy_audio_len,
    opset,
    exporter,
    run_eager_parity,
    run_onnx_checker,
    validate_ort,
    ort_provider,
    fail_on_eager_mismatch,
    fail_on_ort_mismatch,
):
    precision_name, dtype = _resolve_precision(precision)

    return ExportSettings(
        device=device,
        precision=precision_name,
        dtype=dtype,
        use_speaker_conditioning=use_speaker_conditioning,
        speaker_emb_dim=speaker_emb_dim,
        text_attn_implementation=text_attn_implementation,
        dummy_batch=dummy_batch,
        dummy_text_len=dummy_text_len,
        dummy_audio_len=dummy_audio_len,
        opset=opset,
        exporter=exporter,
        run_eager_parity=run_eager_parity,
        run_onnx_checker=run_onnx_checker,
        validate_ort=validate_ort,
        ort_provider=ort_provider,
        fail_on_eager_mismatch=fail_on_eager_mismatch,
        fail_on_ort_mismatch=fail_on_ort_mismatch,
    )



def _build_manifest(settings, result, extra=None):
    manifest = {
        "precision": settings.precision,
        "dtype": str(settings.dtype),
        "device": settings.device,
        "encoder_path": str(result["encoder_path"]),
        "decoder_path": str(result["decoder_path"]),
        "opset": int(settings.opset),
        "exporter": settings.exporter,
        "use_speaker_conditioning": bool(settings.use_speaker_conditioning),
        "text_attn_implementation": settings.text_attn_implementation,
        "dummy_batch": int(settings.dummy_batch),
        "dummy_text_len": int(settings.dummy_text_len),
        "dummy_audio_len": int(settings.dummy_audio_len),
        "eager_validation": result["eager_validation"],
        "onnx_checker": result["onnx_checker"],
        "ort_validation": result["ort_validation"],
    }

    if extra:
        manifest.update(extra)

    return manifest



def export_tts_core_full_onnx(
    repo_root,
    config_path,
    ckpt_dir,
    out_dir,
    device="cpu",
    precision="auto",
    use_speaker_conditioning=False,
    speaker_emb_dim=192,
    text_attn_implementation="sdpa",
    dummy_batch=1,
    dummy_text_len=64,
    dummy_audio_len=256,
    opset=18,
    exporter="dynamo",
    run_eager_parity=True,
    run_onnx_checker=True,
    validate_ort=True,
    ort_provider="CPUExecutionProvider",
    fail_on_eager_mismatch=True,
    fail_on_ort_mismatch=False,
):
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = _build_settings(
        device=device,
        precision=precision,
        use_speaker_conditioning=use_speaker_conditioning,
        speaker_emb_dim=speaker_emb_dim,
        text_attn_implementation=text_attn_implementation,
        dummy_batch=dummy_batch,
        dummy_text_len=dummy_text_len,
        dummy_audio_len=dummy_audio_len,
        opset=opset,
        exporter=exporter,
        run_eager_parity=run_eager_parity,
        run_onnx_checker=run_onnx_checker,
        validate_ort=validate_ort,
        ort_provider=ort_provider,
        fail_on_eager_mismatch=fail_on_eager_mismatch,
        fail_on_ort_mismatch=fail_on_ort_mismatch,
    )
    model, cfg, checkpoint, ckpt_path, state_key = _load_model(
        repo_root,
        config_path,
        ckpt_dir,
        settings,
    )
    result = _prepare_export(
        model,
        cfg,
        out_dir,
        settings,
    )

    manifest = _build_manifest(
        settings,
        result,
        {
            "repo_root": str(Path(repo_root).expanduser().resolve()),
            "config_path": str(Path(config_path).expanduser().resolve()),
            "ckpt_path": str(ckpt_path),
            "checkpoint_state_key": state_key,
            "checkpoint_step": (
                checkpoint.get("step")
                if isinstance(checkpoint, dict)
                else None
            ),
        },
    )


    manifest_path = out_dir / f"manifest_{settings.precision}.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)

    return manifest


def export_tts_core_full_onnx_from_loaded_model(
    model,
    cfg,
    out_dir,
    *,
    device="cpu",
    use_speaker_conditioning=False,
    speaker_emb_dim=192,
    text_attn_implementation="sdpa",
    dummy_batch=1,
    dummy_text_len=64,
    dummy_audio_len=256,
    opset=18,
    exporter="legacy",
    run_eager_parity=False,
    run_onnx_checker=True,
    validate_ort=False,
    ort_provider="CPUExecutionProvider",
    fail_on_eager_mismatch=True,
    fail_on_ort_mismatch=False,
):
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = _build_settings(
        device=device,
        precision="fp32",
        use_speaker_conditioning=use_speaker_conditioning,
        speaker_emb_dim=speaker_emb_dim,
        text_attn_implementation=text_attn_implementation,
        dummy_batch=dummy_batch,
        dummy_text_len=dummy_text_len,
        dummy_audio_len=dummy_audio_len,
        opset=opset,
        exporter=exporter,
        run_eager_parity=run_eager_parity,
        run_onnx_checker=run_onnx_checker,
        validate_ort=validate_ort,
        ort_provider=ort_provider,
        fail_on_eager_mismatch=fail_on_eager_mismatch,
        fail_on_ort_mismatch=fail_on_ort_mismatch,
    )

    export_model = copy.deepcopy(model).eval()
    export_model.to(
        device=device,
        dtype=torch.float32,
    )
    _disable_flash_attention(export_model)

    result = _prepare_export(
        export_model,
        cfg,
        out_dir,
        settings,
    )

    manifest = _build_manifest(
        settings,
        result,
        {
            "kind": "core_from_loaded_model",
            "source": "already_loaded_model_object",
        },
    )


    manifest_path = out_dir / "manifest_from_loaded_model_fp32.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)

    return manifest
