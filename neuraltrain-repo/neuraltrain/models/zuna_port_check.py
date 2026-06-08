"""Temporary AY2Latent decoder check for the NeuroAI ZUNA port.

This module is intentionally separate from ``zuna.py`` so the reconstruction
check can be removed without changing the encoder adapter itself.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "check_neuroai_port"


class ZUNAPortChecker:
    """Decode and save the first NeuroAI batch."""

    def __init__(
        self,
        decoder: torch.nn.Module,
        *,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        global_sigma: float,
        dont_noise_chan_xyz: bool,
        sample_steps: int = 50,
        cfg: float = 1.0,
        min_batch_size: int = 1,
    ) -> None:
        if sample_steps <= 0:
            raise ValueError("sample_steps must be positive")
        self.decoder: torch.nn.Module | None = decoder
        self.output_dir = output_dir
        self.global_sigma = global_sigma
        self.dont_noise_chan_xyz = dont_noise_chan_xyz
        self.sample_steps = sample_steps
        self.cfg = cfg
        self.min_batch_size = min_batch_size
        self._has_run = False

    @torch.no_grad()
    def _sample_from_latent(
        self,
        *,
        latent: torch.Tensor,
        encoder_input: torch.Tensor,
        seq_lens: torch.Tensor,
        tok_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Equivalent to ``EncoderDecoder.sample`` without re-running its encoder."""
        bsz, _, latent_dim = latent.shape
        dt_time = torch.full(
            (bsz,),
            1.0 / self.sample_steps,
            device=latent.device,
            dtype=torch.float32,
        )
        dt = dt_time[:, None, None]
        z = self.global_sigma * torch.randn_like(encoder_input).to(latent.device)

        if self.dont_noise_chan_xyz:
            if latent_dim not in (35, 131):
                raise ValueError(
                    "dont_noise_chan_xyz is enabled, but channel positions are "
                    f"not concatenated to the {latent_dim}-D encoder representation"
                )
            z[:, :, :3] = encoder_input[:, :, :3]

        for step in range(self.sample_steps, 0, -1):
            time = (dt_time * step)[:, None, None]
            velocity, _ = self.decoder(
                tokens=z.unsqueeze(1),
                cross_attended=latent,
                timeD=time,
                seq_lens=seq_lens,
                cross_seq_lens=seq_lens,
                tok_idx=tok_idx,
                cross_tok_idx=tok_idx,
            )
            if self.cfg != 1.0:
                unconditioned_velocity, _ = self.decoder(
                    tokens=z.unsqueeze(1),
                    cross_attended=torch.zeros_like(latent),
                    timeD=time,
                    seq_lens=seq_lens,
                    cross_seq_lens=seq_lens,
                    tok_idx=tok_idx,
                    cross_tok_idx=tok_idx,
                )
                velocity = unconditioned_velocity + self.cfg * (
                    velocity - unconditioned_velocity
                )
            z = z - dt * velocity
            if self.dont_noise_chan_xyz:
                z[:, :, :3] = encoder_input[:, :, :3]

        return z

    @staticmethod
    def _unpack_reconstruction(
        packed: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        n_channels: int,
        n_times: int,
        fine_time: int,
        valid_channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Invert ZUNA's coarse-time-major (``use_coarse_time='A'``) packing."""
        coarse_time = n_times // fine_time
        samples = packed.squeeze(0).split(seq_lens.detach().cpu().tolist(), dim=0)
        if valid_channel_mask is None:
            valid_channel_mask = torch.ones(
                len(samples),
                n_channels,
                dtype=torch.bool,
                device=packed.device,
            )
        unpacked = []
        for sample, valid in zip(samples, valid_channel_mask, strict=True):
            n_valid = int(valid.sum())
            expected = n_valid * coarse_time
            if sample.shape != (expected, fine_time):
                raise ValueError(
                    "Unexpected decoder output shape for reconstruction: "
                    f"got {tuple(sample.shape)}, expected {(expected, fine_time)}"
                )
            dense = sample.new_zeros(n_channels, n_times)
            dense[valid] = (
                sample.reshape(coarse_time, n_valid, fine_time)
                .transpose(0, 1)
                .reshape(n_valid, n_times)
            )
            unpacked.append(dense)
        return torch.stack(unpacked)

    @staticmethod
    def _plot_reconstructions(
        encoder_source: torch.Tensor,
        reconstruction: torch.Tensor,
        output_dir: Path,
    ) -> None:
        from apps import AY2latent_bci

        ay2latent_bci_dir = str(Path(AY2latent_bci.__file__).resolve().parent)
        if ay2latent_bci_dir not in sys.path:
            sys.path.insert(0, ay2latent_bci_dir)
        from apps.AY2latent_bci.eeg_eval import (
            compute_nmse,
            compute_pcc,
            plot_compare_eeg_signal,
        )

        for batch_idx, (source, recon) in enumerate(
            zip(encoder_source, reconstruction, strict=True)
        ):
            source_np = source.float().cpu().numpy()
            recon_np = recon.float().cpu().numpy()
            plot_compare_eeg_signal(
                data=source_np,
                reconst=recon_np,
                mse_value=compute_nmse(source_np, recon_np),
                pcc_value=compute_pcc(source_np, recon_np),
                eeg_signal=None,
                fs=256,
                batch=batch_idx,
                sample=0,
                idx=0,
                fname_tag="_neuroai_port_unmasked",
                dir_base=str(output_dir),
            )

    def run_once(
        self,
        *,
        x: torch.Tensor,
        channel_positions: torch.Tensor,
        zuna_channel_positions: torch.Tensor,
        valid_channel_mask: torch.Tensor,
        preprocessed_x: torch.Tensor,
        encoder_input: torch.Tensor,
        seq_lens: torch.Tensor,
        tok_idx: torch.Tensor,
        latent: torch.Tensor,
        fine_time: int,
    ) -> None:
        """Save one real batch and its reconstruction, then disable the check."""
        if self._has_run or x.shape[0] < self.min_batch_size:
            return

        # Set this before decoding so a failed check does not repeatedly run in training.
        self._has_run = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for stale_name in (
            "masked_preprocessed_x.pt",
            "eeg_signal_compare_B0_S0_neuroai_port_first_five_masked.png",
        ):
            (self.output_dir / stale_name).unlink(missing_ok=True)
        if self.decoder is None:
            raise RuntimeError(
                "The one-time ZUNA port-check decoder was already released"
            )
        decoder = self.decoder.to(latent.device)
        decoder.eval()

        logger.warning(
            "Running one-time ZUNA NeuroAI port reconstruction check on batch %s",
            tuple(x.shape),
        )
        self.decoder = decoder
        packed_reconstruction = self._sample_from_latent(
            latent=latent.detach(),
            encoder_input=encoder_input.detach().unsqueeze(0),
            seq_lens=seq_lens,
            tok_idx=tok_idx,
        )
        reconstruction = self._unpack_reconstruction(
            packed_reconstruction,
            seq_lens=seq_lens,
            n_channels=x.shape[1],
            n_times=x.shape[2],
            fine_time=fine_time,
            valid_channel_mask=valid_channel_mask,
        )

        tensors = {
            "x.pt": x,
            "channel_positions.pt": channel_positions,
            "zuna_channel_positions.pt": zuna_channel_positions,
            "valid_channel_mask.pt": valid_channel_mask,
            "preprocessed_x.pt": preprocessed_x,
            "reconstructions.pt": reconstruction,
            "packed_reconstructions.pt": packed_reconstruction,
            "encoder_latents.pt": latent,
            "encoder_input.pt": encoder_input,
            "seq_lens.pt": seq_lens,
            "tok_idx.pt": tok_idx,
        }
        for filename, tensor in tensors.items():
            torch.save(tensor.detach().cpu(), self.output_dir / filename)

        self._plot_reconstructions(
            preprocessed_x.detach(),
            reconstruction.detach(),
            self.output_dir,
        )
        # The benchmark only needs the encoder after this one-time check.
        self.decoder = None
        del decoder
        logger.warning("Saved ZUNA port check to %s", self.output_dir)
