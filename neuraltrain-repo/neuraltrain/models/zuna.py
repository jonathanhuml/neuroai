"""NeuralTrain adapter for ZUNA.

The ZUNA EEG encoder and tokenization helpers live in ``AY2latent/lingua``.
This module only adapts NeuralBench's dense EEG batches ``(B, C, T)`` plus
``channel_positions`` to the packed token format expected by ZUNA.
"""

from __future__ import annotations

import logging
import typing as tp
from pathlib import Path

import torch
import torch.nn as nn
from pydantic import Field

from .base import BaseModelConfig

logger = logging.getLogger(__name__)


def _zuna_model_defaults() -> dict[str, tp.Any]:
    return {
        "dim": 1024,
        "n_layers": 16,
        "head_dim": 64,
        "seqlen_t": False,
        "huber_c": None,
        "input_dim": 32,
        "encoder_input_dim": 32,
        "encoder_output_dim": 32,
        "encoder_sliding_window": 65536,
        "sliding_window": 65536,
        "xattn_sliding_window": 65536,
        "max_seqlen": 256,
        "max_chans": 512,
        "model_dtype": "bf16",
        "stft_global_sigma": 0.1,
        "adaptive_loss_weighting": True,
        "num_fine_time_pts": 32,
        "rope_dim": 4,
        "rope_theta": 10000.0,
        "tok_idx_type": "{x,y,z,tc}",
        "dont_noise_chan_xyz": False,
        "zero_spatial": False,
        "dropout_vec_type": "zeros",
        "register_tok_idx": "mean_all",
    }


class ZUNAEncoder(nn.Module):
    """Expose ZUNA encoder latents with a NeuralBench-compatible signature."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_fine_time_pts: int,
        data_norm: float = 10.0,
        data_clip: float | None = 1.0,
        do_avg_ref: bool = True,
        num_bins_discretize_xyz_chan_pos: int = 100,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_fine_time_pts = num_fine_time_pts
        self.data_norm = data_norm
        self.data_clip = data_clip
        self.do_avg_ref = do_avg_ref
        self.num_bins = num_bins_discretize_xyz_chan_pos
        self.register_buffer(
            "xyz_extremes",
            torch.tensor(
                [[-0.12, -0.12, -0.12], [0.12, 0.12, 0.12]], dtype=torch.float32
            ),
        )

    # Average reference, normalization, and clipping
    # TODO: Figure out ordering wrt filtering, etc, should this be kicked to YAML file?
    # This is ZUNA-specific, so not done in the generic data YAML section
    # Can we add our own preprocess functions there?
    def _preprocess_eeg(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.do_avg_ref:
            x = x - x.mean(dim=1, keepdim=True)

        eps = 1e-6
        x = (x - x.mean(dim=-1, keepdim=True)) / (
            x.std(dim=-1, keepdim=True, unbiased=False) + eps
        )

        x = x / self.data_norm
        if self.data_clip is not None:
            x = x.clamp(-self.data_clip, self.data_clip)
        return x

    # SEQUENCE PACKING
    # NeuralBench forward input: (B, C, T)
    # Re-pack to ZUNA encoder input: (1, B*C*coarse_time, fine_time).
    def _sequence_repacking(
        self,
        x: torch.Tensor,
        channel_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def _tokenize_batch_sample(
            eeg: torch.Tensor,
            channel_positions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Discretize channel positions for RoPE.

            chop_and_reshape_signals() converts (C, T) to
            (C, coarse_time, fine_time). Then build tok_idx: (1, C * coarse_time, 4).
            """
            from apps.AY2latent_bci.eeg_data import (
                chop_and_reshape_signals,
                discretize_chan_pos,
            )

            chan_pos_discrete = discretize_chan_pos(
                channel_positions.float(),
                self.xyz_extremes.to(channel_positions.device),
                self.num_bins,
            )
            (
                encoder_input,
                _chan_pos,
                chan_pos_discrete,
                _chan_id,
                t_coarse,
                seq_len,
                _num_chans,
            ) = chop_and_reshape_signals(
                eeg_signal=eeg,
                chan_pos=channel_positions,
                chan_pos_discrete=chan_pos_discrete,
                tf=self.num_fine_time_pts,
                use_coarse_time="A",
            )

            seq_lens = torch.tensor([seq_len], dtype=torch.long, device=eeg.device)
            tok_idx = torch.cat(
                (
                    chan_pos_discrete.to(eeg.device).unsqueeze(0),
                    t_coarse.to(eeg.device).long().unsqueeze(0),
                ),
                dim=2,
            )
            return encoder_input.to(eeg.device), seq_lens, tok_idx

        # Reshaped data: length B, each tensor (C * coarse_time, fine_time)
        encoder_inputs: list[torch.Tensor] = []
        # Sample boundaries: length B, each tensor (1,) with value C * coarse_time
        seq_lens: list[torch.Tensor] = []
        # Positional indices for RoPE: length B, each tensor (1, C * coarse_time, 4)
        tok_indices: list[torch.Tensor] = []

        for eeg, pos in zip(x, channel_positions, strict=True):
            encoder_input, seq_len, tok_idx = _tokenize_batch_sample(eeg, pos)
            encoder_inputs.append(encoder_input)
            seq_lens.append(seq_len)
            tok_indices.append(tok_idx)

        return (
            torch.cat(encoder_inputs, dim=0), # (dim0, dim1)=(C*coarse_time,fine_time): cat=0 --> B*(dim0, dim1)
            torch.cat(seq_lens, dim=0),
            torch.cat(tok_indices, dim=1),
        )

    def _encoder_forward(
        self,
        encoder_input: torch.Tensor,
        seq_lens: torch.Tensor,
        tok_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return self.encoder(
            token_values=encoder_input.unsqueeze(0),
            seq_lens=seq_lens,
            tok_idx=tok_idx,
            attn_impl="flex_attention",
        )

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return ZUNA encoder latent tokens.

        Parameters
        ----------
        x : Tensor
            EEG window with shape ``(B, C, T)``.
        channel_positions : Tensor
            Electrode coordinates with shape ``(B, C, 3)``.

        Returns
        -------
        Tensor
            Encoder latent embeddings with shape ``(B, C * (T // fine_time), D)``.
        """
        if x.ndim != 3:
            raise ValueError(f"ZUNA expects x with shape (B, C, T), got {tuple(x.shape)}")
        if channel_positions.ndim != 3 or channel_positions.shape[:2] != x.shape[:2]:
            raise ValueError(
                "ZUNA expects channel_positions with shape (B, C, 3) matching x; "
                f"got x={tuple(x.shape)} and "
                f"channel_positions={tuple(channel_positions.shape)}"
            )
        if channel_positions.shape[-1] != 3:
            raise ValueError(
                f"ZUNA uses 3-D channel positions, got {channel_positions.shape[-1]}"
            )
        if x.shape[-1] % self.num_fine_time_pts != 0:
            raise ValueError(
                "ZUNA requires the time dimension to be divisible by "
                f"num_fine_time_pts={self.num_fine_time_pts}; got T={x.shape[-1]}."
            )

        output_device = x.device
        if not x.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("ZUNA FlexAttention inference requires a CUDA device.")
            device = torch.device("cuda")
            self.to(device)
            x = x.to(device)
            channel_positions = channel_positions.to(device)
        else:
            param = next(self.encoder.parameters(), None)
            if param is not None and param.device != x.device:
                self.to(x.device)

        x = self._preprocess_eeg(x)
        encoder_input, seq_lens, tok_idx = self._sequence_repacking(
            x, channel_positions
        )
        latent, _tok_idx_reg, _losses = self._encoder_forward(
            encoder_input, seq_lens, tok_idx
        )
        split_sizes = seq_lens.detach().cpu().tolist()
        outputs = latent.squeeze(0).split(split_sizes, dim=0)
        return torch.stack(outputs, dim=0).to(output_device)


class NtZuna(BaseModelConfig):
    """Config for the ZUNA pretrained EEG encoder."""

    model_kwargs: dict[str, tp.Any] = Field(default_factory=_zuna_model_defaults)
    checkpoint_path: str | None = (
        "/data/groups/bci/checkpoints/bci/ZUNA2_5e-4/checkpoints/0000052500"
    )
    num_fine_time_pts: int = 32
    data_norm: float = 10.0
    data_clip: float | None = 1.0
    do_avg_ref: bool = True
    num_bins_discretize_xyz_chan_pos: int = 100

    def build(
        self,
        n_in_channels: int | None = None,
        n_outputs: int | None = None,
        **kwargs: tp.Any,
    ) -> nn.Module:
        del n_in_channels, n_outputs, kwargs

        from apps.AY2latent_bci.transformer import DecoderTransformerArgs, EncoderDecoder
        from lingua.args import dataclass_from_dict
        from lingua.checkpoint import load_from_checkpoint

        model_kwargs = dict(self.model_kwargs)
        model_kwargs["encoder_latent_downsample_factor"] = 1
        model_args = dataclass_from_dict(DecoderTransformerArgs, model_kwargs)
        model = EncoderDecoder(model_args)
        model.init_weights()

        if self.checkpoint_path is not None:
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"ZUNA checkpoint_path does not exist: {checkpoint_path}"
                )
            logger.info("Loading ZUNA checkpoint from %s", checkpoint_path)
            load_from_checkpoint(str(checkpoint_path), model, model_key="model")

        return ZUNAEncoder(
            model.encoder,
            num_fine_time_pts=self.num_fine_time_pts,
            data_norm=self.data_norm,
            data_clip=self.data_clip,
            do_avg_ref=self.do_avg_ref,
            num_bins_discretize_xyz_chan_pos=self.num_bins_discretize_xyz_chan_pos,
        )
