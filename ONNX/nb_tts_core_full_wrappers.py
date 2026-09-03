import torch
import torch.nn as nn

from typing import Tuple



class TTSFullEncoderNoRefONNX(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        text_cond_drop: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_mask = text_mask.to(torch.bool)
        text_cond_drop = text_cond_drop.to(torch.bool)
        context = self.model.encode_text(text_ids, text_mask)
        context_mask = text_mask & (~text_cond_drop[:, None])
        return context, context_mask


class TTSFullDecoderDenoiserONNX(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        latent_mask: torch.Tensor,
        span_mask: torch.Tensor,
        valid_audio_mask: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        kv_cache = self.model.get_kv_cache(context)
        return self.model.denoise(
            x=x,
            t=t,
            context_mask=context_mask.to(torch.bool),
            kv_cache=kv_cache,
            start_pos=0,
            use_checkpoint=False,
            latent_mask=latent_mask.to(torch.bool),
            span_mask=span_mask.to(torch.bool),
            context_bias=None,
            valid_audio_mask=valid_audio_mask.to(torch.bool),
            cond=cond,
            audio_cond_drop=None,
            return_hidden_states=False,
            speaker_emb=None,
            speaker_cond_drop=None,
            speaker_adaln_scale=1.0,
        )


class TTSFullDecoderDenoiserWithSpeakerONNX(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        latent_mask: torch.Tensor,
        span_mask: torch.Tensor,
        valid_audio_mask: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        speaker_emb: torch.Tensor,
        speaker_cond_drop: torch.Tensor,
    ) -> torch.Tensor:
        kv_cache = self.model.get_kv_cache(context)
        return self.model.denoise(
            x=x,
            t=t,
            context_mask=context_mask.to(torch.bool),
            kv_cache=kv_cache,
            start_pos=0,
            use_checkpoint=False,
            latent_mask=latent_mask.to(torch.bool),
            span_mask=span_mask.to(torch.bool),
            context_bias=None,
            valid_audio_mask=valid_audio_mask.to(torch.bool),
            cond=cond,
            audio_cond_drop=None,
            return_hidden_states=False,
            speaker_emb=speaker_emb,
            speaker_cond_drop=speaker_cond_drop.to(torch.bool),
            speaker_adaln_scale=1.0,
        )
