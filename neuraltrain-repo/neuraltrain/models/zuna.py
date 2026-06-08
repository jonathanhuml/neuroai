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
        channel_position_montage: str = "standard_1005",
        invalid_channel_position: float = -0.1,
        port_checker: tp.Any | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_fine_time_pts = num_fine_time_pts
        self.data_norm = data_norm
        self.data_clip = data_clip
        self.do_avg_ref = do_avg_ref
        self.num_bins = num_bins_discretize_xyz_chan_pos
        self.channel_position_montage = channel_position_montage
        self.invalid_channel_position = invalid_channel_position
        self._warned_invalid_positions = False
        # Kept as a plain object so the temporary decoder is not registered as
        # part of the benchmark model or included in its optimizer/state dict.
        object.__setattr__(self, "_port_checker", port_checker)
        native_to_head = self._native_to_head_transform(channel_position_montage)
        self.register_buffer(
            "native_to_head_rotation",
            native_to_head[:3, :3],
        )
        self.register_buffer(
            "native_to_head_translation",
            native_to_head[:3, 3],
        )
        self.register_buffer(
            "xyz_extremes",
            torch.tensor(
                [[-0.12, -0.12, -0.12], [0.12, 0.12, 0.12]], dtype=torch.float32
            ),
        )

    @staticmethod
    def _native_to_head_transform(montage_name: str) -> torch.Tensor:
        if montage_name in {"standard_1005", "standard_1020"}:
            return torch.tensor(
                [
                    [0.999993681908, 0.003551873844, 0.000202048104, -0.001762724953],
                    [-0.003557568649, 0.998389124870, 0.056625857949, 0.031094350428],
                    [-0.000000594737, -0.056626219302, 0.998395442963, 0.039597249076],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float64,
            )

        import mne

        montage = mne.channels.make_standard_montage(montage_name)
        transform = mne.channels.compute_native_head_t(montage)["trans"]
        return torch.as_tensor(transform, dtype=torch.float64)

    def _valid_channel_mask(self, channel_positions: torch.Tensor) -> torch.Tensor:
        sentinel = torch.isclose(
            channel_positions,
            torch.as_tensor(
                self.invalid_channel_position,
                dtype=channel_positions.dtype,
                device=channel_positions.device,
            ),
            rtol=0.0,
            atol=1e-6,
        ).all(dim=-1)
        valid = torch.isfinite(channel_positions).all(dim=-1) & ~sentinel
        invalid_count = int((~valid).sum().item())
        if invalid_count and not self._warned_invalid_positions:
            logger.warning(
                "Excluding %d channel(s) with invalid/sentinel positions before "
                "ZUNA tokenization.",
                invalid_count,
            )
            self._warned_invalid_positions = True
        if (~valid).all(dim=1).any():
            raise ValueError("ZUNA received a sample with no valid channel positions.")
        return valid

    def _to_zuna_native_frame(
        self,
        channel_positions: torch.Tensor,
        valid_channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convert NeuralBench MNE-head coordinates to ZUNA's native frame."""
        native_positions = channel_positions.clone()
        points = channel_positions[valid_channel_mask].to(torch.float64)
        rotation = self.native_to_head_rotation.to(points.device)
        translation = self.native_to_head_translation.to(points.device)
        native_points = torch.linalg.solve(
            rotation,
            (points - translation).T,
        ).T
        native_positions[valid_channel_mask] = native_points.to(
            native_positions.dtype
        )
        return native_positions

    # Average reference, normalization, and clipping
    # TODO: Figure out ordering wrt filtering, etc, should this be kicked to YAML file?
    # This is ZUNA-specific, so not done in the generic data YAML section
    # Can we add our own preprocess functions there?
    def _preprocess_eeg(
        self,
        x: torch.Tensor,
        valid_channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.float()
        if self.do_avg_ref:
            if valid_channel_mask is None:
                x = x - x.mean(dim=1, keepdim=True)
            else:
                mask = valid_channel_mask.unsqueeze(-1).to(x.dtype)
                valid_count = mask.sum(dim=1, keepdim=True)
                channel_mean = (x * mask).sum(dim=1, keepdim=True) / valid_count
                x = (x - channel_mean) * mask

        eps = 1e-6
        x = (x - x.mean(dim=-1, keepdim=True)) / (
            x.std(dim=-1, keepdim=True, unbiased=False) + eps
        )

        x = x / self.data_norm
        if self.data_clip is not None:
            x = x.clamp(-self.data_clip, self.data_clip)
        if valid_channel_mask is not None:
            x = x * valid_channel_mask.unsqueeze(-1)
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

            valid = self._valid_channel_mask(channel_positions.unsqueeze(0))[0]
            eeg = eeg[valid]
            channel_positions = channel_positions[valid]
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

    def _restore_dense_outputs(
        self,
        latent: torch.Tensor,
        seq_lens: torch.Tensor,
        valid_channel_mask: torch.Tensor,
        n_times: int,
    ) -> torch.Tensor:
        """Restore packed latents as ``(batch, channel, coarse_time, dim)``."""
        coarse_time = n_times // self.num_fine_time_pts
        split_sizes = seq_lens.detach().cpu().tolist()
        sparse_outputs = latent.squeeze(0).split(split_sizes, dim=0)
        dense_outputs = []
        for sparse, valid in zip(
            sparse_outputs, valid_channel_mask, strict=True
        ):
            dense = sparse.new_zeros(
                valid.shape[0],
                coarse_time,
                sparse.shape[-1],
            )
            dense[valid] = sparse.reshape(
                coarse_time,
                int(valid.sum()),
                sparse.shape[-1],
            ).transpose(0, 1)
            dense_outputs.append(dense)
        return torch.stack(dense_outputs)

    def _encoder_forward(
        self,
        encoder_input: torch.Tensor,
        seq_lens: torch.Tensor,
        tok_idx: torch.Tensor,
        do_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        return self.encoder(
            token_values=encoder_input.unsqueeze(0),
            seq_lens=seq_lens,
            tok_idx=tok_idx,
            do_idx=do_idx,
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
            Encoder latent embeddings with shape
            ``(B, C, T // fine_time, D)``.
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

        raw_x = x
        raw_channel_positions = channel_positions
        valid_channel_mask = self._valid_channel_mask(channel_positions)
        zuna_channel_positions = self._to_zuna_native_frame(
            channel_positions,
            valid_channel_mask,
        )
        x = self._preprocess_eeg(x, valid_channel_mask)
        encoder_input, seq_lens, tok_idx = self._sequence_repacking(
            x, zuna_channel_positions
        )
        # Match AY2Latent's normal encoder path. No debug masking is applied;
        # this only identifies tokens that were already all-zero in the data.
        do_idx = encoder_input.sum(dim=-1) == 0
        latent, _tok_idx_reg, _losses = self._encoder_forward(
            encoder_input, seq_lens, tok_idx, do_idx
        )
        if self._port_checker is not None:
            self._port_checker.run_once(
                x=raw_x,
                channel_positions=raw_channel_positions,
                zuna_channel_positions=zuna_channel_positions,
                valid_channel_mask=valid_channel_mask,
                preprocessed_x=x,
                encoder_input=encoder_input,
                seq_lens=seq_lens,
                tok_idx=tok_idx,
                latent=latent,
                fine_time=self.num_fine_time_pts,
            )
        outputs = self._restore_dense_outputs(
            latent,
            seq_lens,
            valid_channel_mask,
            x.shape[-1],
        )
        return outputs.to(output_device)


class NtZuna(BaseModelConfig):
    """Config for the ZUNA pretrained EEG encoder. Hyperparameters are Pydantic fields."""

    model_kwargs: dict[str, tp.Any] = Field(default_factory=_zuna_model_defaults)
    checkpoint_path: str | None = (
        "/data/groups/bci/checkpoints/bci/ZUNA2_5e-4/checkpoints/0000052500"
    )
    num_fine_time_pts: int = 32
    data_norm: float = 10.0
    data_clip: float | None = 1.0
    do_avg_ref: bool = True
    num_bins_discretize_xyz_chan_pos: int = 100
    channel_position_montage: str = "standard_1005"
    invalid_channel_position: float = -0.1
    debug_mode: bool = False

    def build(
        self,
        n_in_channels: int | None = None,
        n_outputs: int | None = None,
        **kwargs: tp.Any,
    ) -> nn.Module:

        from apps.AY2latent_bci.transformer import DecoderTransformerArgs, EncoderDecoder
        from lingua.args import dataclass_from_dict
        from lingua.checkpoint import load_from_checkpoint

        model_kwargs = dict(self.model_kwargs)
        model_kwargs["encoder_latent_downsample_factor"] = 1
        model_kwargs["ape_dim"] = 0
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

        port_checker = None
        if self.debug_mode:
            from .zuna_port_check import ZUNAPortChecker

            port_checker = ZUNAPortChecker(
                model.decoder,
                global_sigma=model.global_sigma,
                dont_noise_chan_xyz=model.dont_noise_chan_xyz,
            )

        return ZUNAEncoder(
            model.encoder,
            num_fine_time_pts=self.num_fine_time_pts,
            data_norm=self.data_norm,
            data_clip=self.data_clip,
            do_avg_ref=self.do_avg_ref,
            num_bins_discretize_xyz_chan_pos=self.num_bins_discretize_xyz_chan_pos,
            channel_position_montage=self.channel_position_montage,
            invalid_channel_position=self.invalid_channel_position,
            port_checker=port_checker,
        )
