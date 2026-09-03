import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from transformers import AutoModel, ModernBertConfig, ModernBertModel


_FA_VARLEN = None
FLASH_ATTN_VERSION = 0

try:
    from flash_attn_interface import flash_attn_varlen_func as _FA_VARLEN
    FLASH_ATTN_VERSION = 3
except ImportError:
    try:
        from flash_attn import flash_attn_varlen_func as _FA_VARLEN
        FLASH_ATTN_VERSION = 2
    except ImportError:
        _FA_VARLEN = None
        FLASH_ATTN_VERSION = 0


def flash_attn_available():
    return _FA_VARLEN is not None


def _unpad(x, mask):
    seqlens = mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    x_unpad = x.reshape(-1, *x.shape[2:])[indices]
    return x_unpad, indices, cu_seqlens, int(seqlens.max().item())


def _pad(x_unpad, indices, batch, seqlen):
    out = torch.zeros(
        batch * seqlen,
        *x_unpad.shape[1:],
        dtype=x_unpad.dtype,
        device=x_unpad.device,
    )
    out[indices] = x_unpad
    return out.view(batch, seqlen, *x_unpad.shape[1:])


def _key_mask(latent_mask, context_key_mask, batch, seqlen, ctx_len, device):
    lat_key = (
        latent_mask.to(torch.bool)
        if latent_mask is not None
        else torch.ones(batch, seqlen, dtype=torch.bool, device=device)
    )
    ctx_key = (
        context_key_mask.to(torch.bool)
        if context_key_mask is not None
        else torch.ones(batch, ctx_len, dtype=torch.bool, device=device)
    )
    return torch.cat([lat_key, ctx_key], dim=1), lat_key


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def model_from_config(cfg, use_speaker_conditioning=False, speaker_emb_dim=192):
    mc = cfg["model"]
    rp = mc["reference_projector"]
    de = mc["decoder"]
    co = mc["conditioning"]

    return DaryaDiT(
        latent_size=mc["latent_size"],
        model_size=de["model_size"],
        num_layers=de["num_layers"],
        num_heads=de["num_heads"],
        intermediate_size=de["intermediate_size"],
        norm_eps=mc.get("norm_eps", 1e-6),
        text_encoder_config=mc["text_encoder"],
        reference_latent_dim=rp["latent_dim"],
        timestep_embed_size=co["timestep_embed_size"],
        adaln_rank=co["adaln_rank"],
        use_speaker_conditioning=use_speaker_conditioning,
        speaker_emb_dim=speaker_emb_dim,
        use_flash_attn=de.get("use_flash_attn", True),
        checkpoint_every_n=de.get("checkpoint_every_n", 4),
    )


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.complex(torch.cos(freqs), torch.sin(freqs))


def apply_rotary_emb(x, freqs_cis):
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:3], -1, 2))
    x_ = x_ * freqs_cis[..., None, :]
    x_ = torch.view_as_real(x_).reshape(x.shape)
    return x_.type_as(x)


def get_timestep_embedding(timestep, embed_size):
    half = embed_size // 2
    freqs = 1000.0 * torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, dtype=torch.float32, device=timestep.device)
        / half
    )
    args = timestep[..., None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(timestep.dtype)


class RMSNorm(nn.Module):
    def __init__(self, model_size, eps):
        super().__init__()
        self.eps = eps
        if isinstance(model_size, int):
            model_size = (model_size,)
        self.weight = nn.Parameter(torch.ones(model_size))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * self.weight
        return x.to(x_dtype)


class LowRankAdaLN(nn.Module):
    def __init__(self, model_size, rank, eps):
        super().__init__()
        self.eps = eps
        self.shift_down = nn.Linear(model_size, rank, bias=False)
        self.scale_down = nn.Linear(model_size, rank, bias=False)
        self.gate_down = nn.Linear(model_size, rank, bias=False)
        self.shift_up = nn.Linear(rank, model_size, bias=True)
        self.scale_up = nn.Linear(rank, model_size, bias=True)
        self.gate_up = nn.Linear(rank, model_size, bias=True)

    def forward(self, x, cond_embed):
        shift, scale, gate = cond_embed.chunk(3, dim=-1)
        shift = self.shift_up(self.shift_down(F.silu(shift))) + shift
        scale = self.scale_up(self.scale_down(F.silu(scale))) + scale
        gate = self.gate_up(self.gate_down(F.silu(gate))) + gate

        x_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * (scale + 1.0) + shift
        gate = torch.tanh(gate)
        return x.to(x_dtype), gate


class SpeakerFiLMAdapter(nn.Module):
    def __init__(
        self,
        speaker_emb_dim,
        model_size,
        num_layers,
        num_modulation_points=2,
        hidden_dim=1024,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modulation_points = num_modulation_points
        self.model_size = model_size

        total_params = num_layers * num_modulation_points * 2 * model_size

        self.mlp = nn.Sequential(
            nn.Linear(speaker_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, total_params),
        )
        
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, speaker_emb):
        bsz = speaker_emb.shape[0]
        params = self.mlp(speaker_emb)
        return params.view(
            bsz,
            self.num_layers,
            self.num_modulation_points,
            2,
            self.model_size,
        )


class ReferenceProjector(nn.Module):
    def __init__(self, latent_dim, text_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, text_dim, bias=False),
            nn.SiLU(),
            nn.Linear(text_dim, text_dim, bias=False),
        )

    def forward(self, x):
        return self.proj(x)


class PretrainedMmBertEncoder(nn.Module):
    def __init__(
        self,
        pretrained_name="jhu-clsp/mmBERT-base",
        use_custom_tokenizer=False,
        custom_vocab_size=None,
        pad_token_id=1,
        init_noise_std=0.02,
        num_layers=None,
        attn_implementation=None,
        torch_dtype=None,
    ):
        super().__init__()
        load_kwargs = {}
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
        self.bert = AutoModel.from_pretrained(pretrained_name, **load_kwargs)
        self.hidden_size = self.bert.config.hidden_size

        if num_layers is not None:
            self._truncate_layers(num_layers)

        if use_custom_tokenizer:
            if custom_vocab_size is None:
                raise ValueError("custom_vocab_size is required when use_custom_tokenizer=True")
            self._swap_embeddings(custom_vocab_size, pad_token_id, init_noise_std)

    def _truncate_layers(self, num_keep):
        total = len(self.bert.layers)
        if num_keep < 1 or num_keep > total:
            raise ValueError(f"num_layers={num_keep} must be in [1, {total}]")
        self.bert.layers = nn.ModuleList(list(self.bert.layers)[:num_keep])
        self.bert.config.num_hidden_layers = num_keep

    @torch.no_grad()
    def _swap_embeddings(self, new_vocab_size, pad_token_id, noise_std):
        old_emb = self.bert.get_input_embeddings()
        emb_mean = old_emb.weight.data.mean(dim=0, keepdim=True)

        new_emb = nn.Embedding(new_vocab_size, self.hidden_size, padding_idx=pad_token_id)
        new_emb.weight.data.copy_(emb_mean.expand(new_vocab_size, -1))
        new_emb.weight.data.add_(torch.randn_like(new_emb.weight.data) * noise_std)
        new_emb.weight.data[pad_token_id].zero_()

        self.bert.set_input_embeddings(new_emb)
        self.bert.config.vocab_size = new_vocab_size
        self.bert.config.pad_token_id = pad_token_id

    def forward(self, input_ids, mask=None):
        attn_mask = mask.to(torch.long) if mask is not None else None
        return self.bert(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state


class Encoder_Reinit(nn.Module):
    def __init__(
        self,
        vocab_size,
        model_size,
        num_layers,
        num_heads,
        intermediate_size,
        max_position_embeddings=8192,
        pad_token_id=1,
        attn_implementation=None,
    ):
        super().__init__()
        config = ModernBertConfig(
            vocab_size=vocab_size,
            hidden_size=model_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            pad_token_id=pad_token_id,
        )
        if attn_implementation is not None:
            self.bert = ModernBertModel._from_config(
                config, attn_implementation=attn_implementation
            )
        else:
            self.bert = ModernBertModel(config)
        self.hidden_size = model_size

    def forward(self, input_ids, mask=None):
        attn_mask = mask.to(torch.long) if mask is not None else None
        return self.bert(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state


def build_text_encoder(cfg):
    kind = cfg.get("kind", "scratch")
    attn_implementation = cfg.get("attn_implementation")

    if kind == "using_pretrained":
        return PretrainedMmBertEncoder(
            pretrained_name=cfg.get("pretrained_name", "jhu-clsp/mmBERT-base"),
            use_custom_tokenizer=cfg.get("use_custom_tokenizer", False),
            custom_vocab_size=cfg.get("vocab_size"),
            pad_token_id=cfg.get("pad_token_id", 1),
            init_noise_std=cfg.get("init_noise_std", 0.02),
            num_layers=cfg.get("num_layers"),
            attn_implementation=attn_implementation,
        )
    if kind == "scratch":
        return Encoder_Reinit(
            vocab_size=cfg["vocab_size"],
            model_size=cfg["model_size"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            intermediate_size=cfg["intermediate_size"],
            max_position_embeddings=cfg.get("max_position_embeddings", 8192),
            pad_token_id=cfg.get("pad_token_id", 1),
            attn_implementation=attn_implementation,
        )
    raise ValueError(f"Unknown text encoder kind: {kind!r}")


class TextJointAttention(nn.Module):
    def __init__(self, model_size, num_heads, text_model_size, norm_eps, use_flash_attn=True):
        super().__init__()
        if model_size % num_heads != 0:
            raise ValueError(f"model_size={model_size} not divisible by num_heads={num_heads}")
        if num_heads % 2 != 0:
            raise ValueError(f"num_heads must be even, got {num_heads}")
        self.num_heads = num_heads
        self.head_dim = model_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if use_flash_attn and self.head_dim > 256:
            raise ValueError(
                f"FlashAttention supports head_dim <= 256, got {self.head_dim}; "
                "set use_flash_attn=False"
            )
        self.use_flash_attn = use_flash_attn

        self.wq = nn.Linear(model_size, model_size, bias=False)
        self.wk = nn.Linear(model_size, model_size, bias=False)
        self.wv = nn.Linear(model_size, model_size, bias=False)
        self.wk_text = nn.Linear(text_model_size, model_size, bias=False)
        self.wv_text = nn.Linear(text_model_size, model_size, bias=False)
        self.q_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)
        self.k_norm = RMSNorm((num_heads, self.head_dim), eps=norm_eps)
        self.gate = nn.Linear(model_size, model_size, bias=False)
        self.wo = nn.Linear(model_size, model_size, bias=False)

    def _apply_rotary_half(self, y, fc):
        y1, y2 = y.chunk(2, dim=-2)
        y1 = apply_rotary_emb(y1, fc)
        return torch.cat([y1, y2], dim=-2)

    def _flash_forward(self, xq, xk, xv, latent_mask, context_key_mask):
        B, S = xq.shape[:2]
        ctx_len = xk.shape[1] - S

        key_mask, lat_key = _key_mask(
            latent_mask, context_key_mask, B, S, ctx_len, xq.device
        )

        q_unpad, q_indices, cu_q, max_q = _unpad(xq, lat_key)
        k_unpad, k_indices, cu_k, max_k = _unpad(xk, key_mask)
        v_unpad = xv.reshape(-1, *xv.shape[2:])[k_indices]

        out = _FA_VARLEN(q_unpad, k_unpad, v_unpad, cu_q, cu_k, max_q, max_k)
        if isinstance(out, tuple):
            out = out[0]

        return _pad(out, q_indices, B, S)

    def _sdpa_forward(self, xq, xk, xv, attn_mask, latent_mask, context_key_mask, context_bias):
        B, S = xq.shape[:2]
        ctx_len = xk.shape[1] - S

        if attn_mask is None:
            key_mask, _ = _key_mask(
                latent_mask, context_key_mask, B, S, ctx_len, xq.device
            )
            mask = torch.zeros(B, 1, S, S + ctx_len, dtype=torch.float32, device=xq.device)
            mask = mask.masked_fill(~key_mask[:, None, None, :], -1e9)
            if context_bias is not None:
                mask[:, :, :, S:] += context_bias.to(mask.dtype)
        else:
            mask = attn_mask

        return F.scaled_dot_product_attention(
            query=xq.transpose(1, 2),
            key=xk.transpose(1, 2),
            value=xv.transpose(1, 2),
            attn_mask=mask.to(xq.dtype),
            is_causal=False,
        ).transpose(1, 2)

    def forward(
        self, x, freqs_cis, kv_cache_text, start_pos,
        attn_mask=None, latent_mask=None, context_key_mask=None, context_bias=None,
    ):
        B, S = x.shape[:2]
        xq = self.wq(x).reshape(B, S, self.num_heads, self.head_dim)
        xk_self = self.wk(x).reshape(B, S, self.num_heads, self.head_dim)
        xv_self = self.wv(x).reshape(B, S, self.num_heads, self.head_dim)
        xq = self.q_norm(xq)
        xk_self = self.k_norm(xk_self)
        gate = self.gate(x)

        if start_pos is None:
            start_pos = 0
        freqs_q = freqs_cis[start_pos: start_pos + S]
        xq = self._apply_rotary_half(xq, freqs_q)
        xk_self = self._apply_rotary_half(xk_self, freqs_q)

        xk_text, xv_text = kv_cache_text
        xk = torch.cat([xk_self, xk_text.to(xk_self.dtype)], dim=1)
        xv = torch.cat([xv_self, xv_text.to(xv_self.dtype)], dim=1)

        use_flash = (
            self.use_flash_attn
            and _FA_VARLEN is not None
            and attn_mask is None
            and context_bias is None
            and xq.is_cuda
            and xq.dtype in (torch.float16, torch.bfloat16)
        )

        if use_flash:
            output = self._flash_forward(xq, xk, xv, latent_mask, context_key_mask)
        else:
            output = self._sdpa_forward(
                xq, xk, xv, attn_mask, latent_mask, context_key_mask, context_bias
            )

        output = output.reshape(B, S, -1)
        output = output * torch.sigmoid(gate)
        return self.wo(output)

    def get_kv_cache_text(self, text_state):
        B = text_state.shape[0]
        xk = self.wk_text(text_state).reshape(B, text_state.shape[1], self.num_heads, self.head_dim)
        xv = self.wv_text(text_state).reshape(B, text_state.shape[1], self.num_heads, self.head_dim)
        xk = self.k_norm(xk)
        return xk, xv


class MLP(nn.Module):
    def __init__(self, model_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(model_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, model_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self, model_size, num_heads, intermediate_size,
        norm_eps, text_model_size, adaln_rank, use_flash_attn=True,
    ):
        super().__init__()
        self.attention = TextJointAttention(
            model_size, num_heads, text_model_size, norm_eps,
            use_flash_attn=use_flash_attn,
        )
        self.mlp = MLP(model_size, intermediate_size)
        self.attention_adaln = LowRankAdaLN(model_size, adaln_rank, norm_eps)
        self.mlp_adaln = LowRankAdaLN(model_size, adaln_rank, norm_eps)

    @staticmethod
    def _apply_speaker_film(x, film_params, point_idx):
        if film_params is None:
            return x
        scale = film_params[:, point_idx, 0, :]
        shift = film_params[:, point_idx, 1, :]
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self, x, cond_embed, context_key_mask, freqs_cis,
        kv_cache_text, start_pos,
        latent_mask=None, context_bias=None, speaker_film=None,
    ):
        x_norm, attention_gate = self.attention_adaln(x, cond_embed)
        x_norm = self._apply_speaker_film(x_norm, speaker_film, point_idx=0)
        x = x + attention_gate * self.attention(
            x=x_norm, freqs_cis=freqs_cis, kv_cache_text=kv_cache_text,
            start_pos=start_pos, latent_mask=latent_mask,
            context_key_mask=context_key_mask, context_bias=context_bias,
        )

        x_norm, mlp_gate = self.mlp_adaln(x, cond_embed)
        x_norm = self._apply_speaker_film(x_norm, speaker_film, point_idx=1)
        x = x + mlp_gate * self.mlp(x_norm)
        return x


class DaryaDiT(nn.Module):
    def __init__(
        self,
        latent_size,
        model_size,
        num_layers,
        num_heads,
        intermediate_size,
        norm_eps,
        text_encoder_config,
        reference_latent_dim,
        timestep_embed_size,
        adaln_rank,
        use_speaker_conditioning=False,
        speaker_emb_dim=192,
        use_flash_attn=True,
        checkpoint_every_n=2,
    ):
        super().__init__()
        if model_size % num_heads != 0:
            raise ValueError(f"model_size={model_size} not divisible by num_heads={num_heads}")
        if checkpoint_every_n < 1:
            raise ValueError(f"checkpoint_every_n must be >= 1, got {checkpoint_every_n}")

        self.latent_size = latent_size
        self.model_size = model_size
        self.num_heads = num_heads
        self.head_dim = model_size // num_heads
        self.timestep_embed_size = timestep_embed_size
        self.num_layers = num_layers
        self.use_speaker_conditioning = use_speaker_conditioning
        self.speaker_emb_dim = speaker_emb_dim
        self.use_flash_attn = use_flash_attn
        self.checkpoint_every_n = checkpoint_every_n

        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {self.head_dim}")
        if timestep_embed_size % 2 != 0:
            raise ValueError(f"timestep_embed_size must be even, got {timestep_embed_size}")

        self.text_encoder = build_text_encoder(text_encoder_config)
        text_model_size = self.text_encoder.hidden_size
        self.text_model_size = text_model_size
        self.text_norm = RMSNorm(text_model_size, norm_eps)

        self.reference_proj = ReferenceProjector(reference_latent_dim, text_model_size)

        self.cond_module = nn.Sequential(
            nn.Linear(timestep_embed_size, model_size, bias=False),
            nn.SiLU(),
            nn.Linear(model_size, model_size, bias=False),
            nn.SiLU(),
            nn.Linear(model_size, model_size * 3, bias=False),
        )

        self.in_proj = nn.Linear(latent_size, model_size, bias=True)
        self.span_mask_proj = nn.Linear(1, model_size, bias=True)
        self.valid_audio_proj = nn.Linear(1, model_size, bias=True)
        nn.init.normal_(self.valid_audio_proj.weight, std=0.02)
        nn.init.zeros_(self.valid_audio_proj.bias)
        nn.init.normal_(self.span_mask_proj.weight, std=0.02)
        nn.init.zeros_(self.span_mask_proj.bias)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                model_size, num_heads, intermediate_size,
                norm_eps, text_model_size, adaln_rank,
                use_flash_attn=use_flash_attn,
            )
            for _ in range(num_layers)
        ])

        self.out_norm = RMSNorm(model_size, norm_eps)
        self.out_proj = nn.Linear(model_size, latent_size, bias=True)
        self.cond_proj = nn.Linear(latent_size, model_size, bias=True)

        if use_speaker_conditioning:
            self.speaker_film_adapter = SpeakerFiLMAdapter(
                speaker_emb_dim=speaker_emb_dim,
                model_size=model_size,
                num_layers=num_layers,
                num_modulation_points=2,
                hidden_dim=1024,
            )
            self.speaker_reg_head = nn.Sequential(
                nn.Linear(model_size, model_size * 2),
                nn.SiLU(),
                nn.Linear(model_size * 2, speaker_emb_dim),
            )
        else:
            self.speaker_film_adapter = None
            self.speaker_reg_head = None

    def encode_text(self, text_input_ids, text_mask):
        return self.text_norm(self.text_encoder(text_input_ids, text_mask))

    def project_reference(self, reference_latent):
        return self.reference_proj(reference_latent)

    def get_kv_cache(self, context):
        return [block.attention.get_kv_cache_text(context) for block in self.blocks]

    def denoise(
        self,
        x,
        t,
        context_mask,
        kv_cache,
        start_pos=None,
        use_checkpoint=False,
        latent_mask=None,
        span_mask=None,
        context_bias=None,
        valid_audio_mask=None,
        cond=None,
        audio_cond_drop=None,
        return_hidden_states=False,
        speaker_emb=None,
        speaker_cond_drop=None,
        speaker_adaln_scale=1.0,
    ):
        if start_pos is None:
            start_pos = 0

        if latent_mask is not None:
            latent_mask = latent_mask.to(torch.bool)
        if span_mask is not None:
            span_mask = span_mask.to(torch.bool)
        if valid_audio_mask is not None:
            valid_audio_mask = valid_audio_mask.to(torch.bool)
        if audio_cond_drop is not None:
            audio_cond_drop = audio_cond_drop.to(torch.bool)

        max_pos = start_pos + x.shape[1]
        freqs_cis = precompute_freqs_cis(self.head_dim, max_pos).to(x.device)

        cond_embed = self.cond_module(get_timestep_embedding(t, self.timestep_embed_size))
        cond_embed = cond_embed[:, None]

        x = self.in_proj(x)

        if cond is not None:
            cond_in = cond.to(x.dtype)
            if audio_cond_drop is not None:
                cond_in = cond_in.masked_fill(audio_cond_drop[:, None, None], 0.0)
            if valid_audio_mask is not None:
                cond_in = cond_in * valid_audio_mask.unsqueeze(-1).to(cond_in.dtype)
            elif latent_mask is not None:
                cond_in = cond_in * latent_mask.unsqueeze(-1).to(cond_in.dtype)
            x = x + self.cond_proj(cond_in)

        if span_mask is not None:
            x = x + self.span_mask_proj(span_mask.unsqueeze(-1).to(x.dtype))
        if valid_audio_mask is not None:
            x = x + self.valid_audio_proj(valid_audio_mask.unsqueeze(-1).to(x.dtype))
        if latent_mask is not None:
            x = x * latent_mask.unsqueeze(-1).to(x.dtype)

        speaker_film_all = None
        if (
            self.use_speaker_conditioning
            and self.speaker_film_adapter is not None
            and speaker_emb is not None
        ):
            spk = speaker_emb.to(x.dtype)
            if speaker_cond_drop is not None:
                spk = spk.masked_fill(speaker_cond_drop.to(torch.bool)[:, None], 0.0)
            speaker_film_all = self.speaker_film_adapter(spk)
            if speaker_adaln_scale != 1.0:
                speaker_film_all = speaker_film_all * float(speaker_adaln_scale)

        hidden_states = [] if return_hidden_states else None

        for i, block in enumerate(self.blocks):
            layer_film = speaker_film_all[:, i] if speaker_film_all is not None else None

            should_checkpoint = (
                use_checkpoint
                and self.training
                and (i % self.checkpoint_every_n == 0)
            )

            if should_checkpoint:
                def _ckpt_fn(x_in, _lf=layer_film, _i=i):
                    return block(
                        x=x_in, cond_embed=cond_embed,
                        context_key_mask=context_mask,
                        freqs_cis=freqs_cis,
                        kv_cache_text=kv_cache[_i],
                        start_pos=start_pos,
                        latent_mask=latent_mask,
                        context_bias=context_bias,
                        speaker_film=_lf,
                    )
                x = cp.checkpoint(_ckpt_fn, x, use_reentrant=False)
            else:
                x = block(
                    x=x, cond_embed=cond_embed,
                    context_key_mask=context_mask,
                    freqs_cis=freqs_cis,
                    kv_cache_text=kv_cache[i],
                    start_pos=start_pos,
                    latent_mask=latent_mask,
                    context_bias=context_bias,
                    speaker_film=layer_film,
                )

            if return_hidden_states:
                hidden_states.append(x)

        x = self.out_norm(x)
        x = self.out_proj(x)
        if latent_mask is not None:
            x = x * latent_mask.unsqueeze(-1).to(x.dtype)
        x = x.float()

        if return_hidden_states:
            return x, hidden_states
        return x

    def forward(
        self,
        x,
        t,
        text_ids,
        text_mask,
        reference_latent,
        reference_mask,
        text_cond_drop=None,
        speaker_cond_drop=None,
        start_pos=None,
        use_checkpoint=False,
        latent_mask=None,
        span_mask=None,
        valid_audio_mask=None,
        cond=None,
        audio_cond_drop=None,
        return_hidden_states=False,
        speaker_emb=None,
        speaker_adaln_scale=1.0,
    ):
        if text_mask is None:
            text_mask = torch.ones(text_ids.shape[:2], dtype=torch.bool, device=text_ids.device)
        else:
            text_mask = text_mask.to(torch.bool)
        if latent_mask is not None:
            latent_mask = latent_mask.to(torch.bool)
        if span_mask is not None:
            span_mask = span_mask.to(torch.bool)
        if valid_audio_mask is not None:
            valid_audio_mask = valid_audio_mask.to(torch.bool)

        text_state = self.encode_text(text_ids, text_mask)

        has_ref = reference_latent is not None and reference_mask is not None
        if has_ref:
            reference_mask = reference_mask.to(torch.bool)
            ref_state = self.project_reference(reference_latent)
            context = torch.cat([text_state, ref_state], dim=1)
            context_mask = torch.cat([text_mask, reference_mask], dim=1)
        else:
            context = text_state
            context_mask = text_mask

        cond_mask = context_mask.clone()
        text_len = text_mask.shape[1]

        if text_cond_drop is not None and text_cond_drop.any():
            cond_mask[text_cond_drop, :text_len] = False
        if has_ref and speaker_cond_drop is not None and speaker_cond_drop.any():
            cond_mask[speaker_cond_drop, text_len:] = False

        return self.denoise(
            x=x, t=t,
            context_mask=cond_mask,
            kv_cache=self.get_kv_cache(context),
            start_pos=start_pos,
            use_checkpoint=use_checkpoint,
            latent_mask=latent_mask,
            span_mask=span_mask,
            context_bias=None,
            valid_audio_mask=valid_audio_mask,
            cond=cond,
            audio_cond_drop=audio_cond_drop,
            return_hidden_states=return_hidden_states,
            speaker_emb=speaker_emb,
            speaker_cond_drop=speaker_cond_drop,
            speaker_adaln_scale=speaker_adaln_scale,
        )

    def predict_speaker(self, hidden, mask):
        if self.speaker_reg_head is None:
            return None
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        return self.speaker_reg_head(pooled)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


DaryaConvDiT = DaryaDiT