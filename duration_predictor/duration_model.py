# a very simple sequence length predictor

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def calculate_remaining_lengths(mel_lengths):
    B = mel_lengths.shape[0]
    max_L = mel_lengths.max().item()

    range_tensor = torch.arange(max_L, device=mel_lengths.device).expand(B, max_L)
    remain_lengths = (mel_lengths[:, None] - 1 - range_tensor).clamp(min=0)

    return remain_lengths


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].to(x.device, x.dtype)
        return x

class SpeechLengthPredictor(nn.Module):

    def __init__(
        self,
        vocab_size=2545,
        latent_dim=52,
        n_mel=None,
        hidden_dim=512,
        n_text_layer=4,
        n_cross_layer=4,
        n_head=8,
        output_dim=1,
        use_speaker_conditioning: bool = False,
        speaker_emb_dim: int = 192,
    ):
        super().__init__()

        if n_mel is not None:
            latent_dim = n_mel

        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.text_pad_id = vocab_size
        self.use_speaker_conditioning = use_speaker_conditioning
        self.speaker_emb_dim = speaker_emb_dim

        self.text_embedder = nn.Embedding(vocab_size + 1, hidden_dim, padding_idx=vocab_size)
        self.text_pe = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_text_layer)

        self.latent_embedder = nn.Linear(latent_dim, hidden_dim)
        self.latent_pe = PositionalEncoding(hidden_dim)

        self.text_only_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.text_only_query, mean=0.0, std=0.02)

      
        if use_speaker_conditioning:
            self.speaker_proj = nn.Sequential(
                nn.Linear(speaker_emb_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.speaker_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.speaker_query, mean=0.0, std=0.02)
        else:
            self.speaker_proj = None
            self.speaker_query = None

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_cross_layer)

        # Final Prediction Layer
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        text_ids: torch.Tensor,
        latent_prompt: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
        latent_padding_mask: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
    ):
        if latent_prompt is None:
            latent_prompt = mel

        if text_ids.dim() != 2:
            raise ValueError(f"expected text_ids shape [B, T_text], got {tuple(text_ids.shape)}")

        batch_size = text_ids.size(0)
        device = text_ids.device

        if text_padding_mask is None:
            text_padding_mask = text_ids.eq(self.text_pad_id)
        else:
            if text_padding_mask.shape != text_ids.shape:
                raise ValueError(
                    f"expected text_padding_mask shape {tuple(text_ids.shape)}, "
                    f"got {tuple(text_padding_mask.shape)}"
                )
            text_padding_mask = text_padding_mask.to(torch.bool)

        # Encode text
        text_embedded = self.text_pe(self.text_embedder(text_ids))
        text_features = self.text_encoder(
            text_embedded,
            src_key_padding_mask=text_padding_mask,
        )

        # Memory defaults to text only
        memory = text_features
        memory_padding_mask = text_padding_mask

        has_speaker = (
            self.use_speaker_conditioning
            and self.speaker_proj is not None
            and speaker_emb is not None
        )

        if has_speaker:
            spk_token = self.speaker_proj(speaker_emb.to(dtype=memory.dtype)).unsqueeze(1)
            memory = torch.cat([memory, spk_token], dim=1)

            spk_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            memory_padding_mask = torch.cat([memory_padding_mask, spk_pad], dim=1)


        use_text_only_mode = latent_prompt is None or latent_prompt.size(1) == 0
        use_speaker_mode = use_text_only_mode and has_speaker

        if use_text_only_mode:
            if use_speaker_mode:
                query = self.speaker_query.expand(batch_size, 1, -1)
            else:
                query = self.text_only_query.expand(batch_size, 1, -1)

            decoder_in = self.latent_pe(query)
            tgt_key_padding_mask = None
        else:
            if latent_prompt.dim() != 3:
                raise ValueError(
                    f"expected latent_prompt shape [B, T_prompt, D], got {tuple(latent_prompt.shape)}"
                )
            if latent_prompt.size(0) != batch_size:
                raise ValueError(
                    f"latent_prompt batch size {latent_prompt.size(0)} "
                    f"does not match text_ids batch size {batch_size}"
                )
            if latent_prompt.size(-1) != self.latent_dim:
                raise ValueError(
                    f"expected latent_prompt last dim {self.latent_dim}, "
                    f"got {latent_prompt.size(-1)}"
                )

            decoder_in = self.latent_pe(self.latent_embedder(latent_prompt))

            if latent_padding_mask is not None:
                expected_shape = latent_prompt.shape[:2]
                if latent_padding_mask.shape != expected_shape:
                    raise ValueError(
                        f"expected latent_padding_mask shape {expected_shape}, "
                        f"got {tuple(latent_padding_mask.shape)}"
                    )
                tgt_key_padding_mask = latent_padding_mask.to(torch.bool)
            else:
                tgt_key_padding_mask = None

        seq_len = decoder_in.size(1)
        causal_mask = self._build_causal_mask(seq_len, device=device)

        decoder_out = self.decoder(
            decoder_in,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )

        length_logits = self.predictor(decoder_out)

        if self.output_dim == 1:
            length_logits = length_logits.squeeze(-1)

        return length_logits