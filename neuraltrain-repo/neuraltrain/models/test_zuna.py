# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ZUNA tests and optional integration smoke utilities.

The command-line utilities preserve the behavior from the former one-off
``tmp_zuna_smoke_compare.py`` and ``tmp_zuna_real_recon_plot.py`` scripts:

    python neuraltrain-repo/neuraltrain/models/test_zuna.py smoke-compare
    python neuraltrain-repo/neuraltrain/models/test_zuna.py real-recon-plot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch._dynamo

try:
    from .zuna import NtZuna, ZUNAEncoder
except ImportError:  # pragma: no cover - direct script execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from neuraltrain.models.zuna import NtZuna, ZUNAEncoder


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
AY2LATENT_LINGUA = Path("/data/groups/bci/jonhuml/workspace/AY2latent/lingua")
AY2LATENT_BCI = AY2LATENT_LINGUA / "apps" / "AY2latent_bci"
DEFAULT_CHECKPOINT = Path(
    "/data/groups/bci/checkpoints/bci/ZUNA2_5e-4/checkpoints/0000052500"
)


torch._dynamo.config.suppress_errors = True
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


def _ensure_external_paths() -> None:
    for path in (
        WORKSPACE_ROOT / "neuraltrain-repo",
        AY2LATENT_BCI,
        AY2LATENT_LINGUA,
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _require_ay2latent() -> None:
    if not AY2LATENT_LINGUA.exists():
        raise RuntimeError(f"AY2latent checkout not found: {AY2LATENT_LINGUA}")
    _ensure_external_paths()


def module_stats(module: torch.nn.Module) -> tuple[int, float]:
    params = list(module.parameters())
    n_params = sum(p.numel() for p in params)
    norm = torch.stack([p.detach().float().norm().cpu() for p in params]).norm().item()
    return n_params, norm


def make_channel_positions(
    batch_size: int,
    n_channels: int,
    device: torch.device,
) -> torch.Tensor:
    base = torch.linspace(-0.08, 0.08, n_channels, device=device)
    pos = torch.stack(
        (
            base,
            torch.sin(torch.linspace(0, 3.14, n_channels, device=device)) * 0.06,
            torch.cos(torch.linspace(0, 3.14, n_channels, device=device)) * 0.04,
        ),
        dim=-1,
    )
    return pos.unsqueeze(0).repeat(batch_size, 1, 1)


def compute_nmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = torch.as_tensor(y_true)
    y_pred = torch.as_tensor(y_pred)
    return (((y_true - y_pred) ** 2).mean() / ((y_true**2).mean() + 1e-8)).item()


def make_tok_idx(batch: dict[str, torch.Tensor], zero_spatial: bool = False) -> torch.Tensor:
    tok_idx = torch.cat(
        (
            batch["chan_pos_discrete"].cpu().unsqueeze(0),
            batch["t_coarse"].cpu().unsqueeze(0),
        ),
        dim=2,
    )
    if zero_spatial:
        tok_idx[:, :, :3] = 0
    return tok_idx


def mask_first_channels(batch: dict[str, torch.Tensor], n_mask_channels: int) -> list[int]:
    chan_ids = batch["chan_id"].squeeze(-1).long()
    unique = torch.unique(chan_ids, sorted=True)
    selected = unique[: min(n_mask_channels, unique.numel())]
    dropout = torch.isin(chan_ids, selected).unsqueeze(-1)
    batch["token_dropout"] = dropout
    return selected.cpu().tolist()


def load_ay2latent_model(
    cfg: NtZuna,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    _require_ay2latent()
    from apps.AY2latent_bci.transformer import DecoderTransformerArgs, EncoderDecoder
    from lingua.args import dataclass_from_dict
    from lingua.checkpoint import load_from_checkpoint

    model_kwargs = dict(cfg.model_kwargs)
    model_kwargs["encoder_latent_downsample_factor"] = 1
    model_kwargs["ape_dim"] = 0
    model_args = dataclass_from_dict(DecoderTransformerArgs, model_kwargs)
    model = EncoderDecoder(model_args)
    model.init_weights()
    load_from_checkpoint(str(checkpoint_path), model, model_key="model")
    model.to(device)
    model.eval()
    return model


def build_reconstruction_model(
    model_cfg: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    _require_ay2latent()
    from apps.AY2latent_bci.transformer import DecoderTransformerArgs, EncoderDecoder
    from lingua.args import dataclass_from_dict
    from lingua.checkpoint import load_from_checkpoint

    model_cfg = dict(model_cfg)
    model_cfg["encoder_latent_downsample_factor"] = 1
    model_cfg["ape_dim"] = 0
    args = dataclass_from_dict(DecoderTransformerArgs, model_cfg)
    model = EncoderDecoder(args)
    model.init_weights()
    load_from_checkpoint(str(checkpoint_path), model, model_key="model")
    model.to(device)
    model.eval()
    return model


def plot_reconstruction(
    *,
    model_output: torch.Tensor,
    batch: dict[str, torch.Tensor],
    data_args: Any,
    encoder_output_dim: int,
    out_dir: Path,
    tag: str,
) -> tuple[float, float]:
    _require_ay2latent()
    from apps.AY2latent_bci.eeg_eval import (
        compute_pcc,
        plot_compare_eeg_signal,
        unwrap_all_the_signals,
    )

    empty_latent = torch.zeros(
        1,
        batch["encoder_input"].shape[0],
        encoder_output_dim,
        device=batch["encoder_input"].device,
    )
    eval_args = SimpleNamespace(data=data_args)
    (
        model_signal_input_unwrapped,
        model_signal_output_unwrapped,
        _model_position_input_unwrapped,
        _model_position_discrete_input_unwrapped,
        _model_position_output_unwrapped,
        eeg_signal_unwrapped,
        _channel_id_unwrapped,
        _latent_data_unwrapped,
        _latent_recon_unwrapped,
        _t_coarse_unwrapped,
    ) = unwrap_all_the_signals(
        model_output=model_output,
        latent_data=empty_latent,
        latent_recon=empty_latent,
        batch=batch,
        args=eval_args,
    )

    input_signal = model_signal_input_unwrapped[0]
    recon_signal = model_signal_output_unwrapped[0]
    original_signal = eeg_signal_unwrapped[0]
    nmse = compute_nmse(original_signal, recon_signal)
    pcc = compute_pcc(original_signal, recon_signal)
    plot_compare_eeg_signal(
        data=input_signal,
        reconst=recon_signal,
        mse_value=nmse,
        pcc_value=pcc,
        eeg_signal=original_signal,
        fs=data_args.sample_rate,
        batch=0,
        sample=0,
        idx=0,
        fname_tag=f"_real_masked_decoder_recon_{tag}",
        dir_base=str(out_dir),
    )
    return nmse, pcc


def run_smoke_compare(args: argparse.Namespace) -> None:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.time % 32 != 0:
        raise ValueError("--time must be divisible by 32 for the current ZUNA adapter.")

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    device = torch.device(args.device)
    cfg = NtZuna(checkpoint_path=str(args.checkpoint))

    raw_ay2latent = load_ay2latent_model(cfg, args.checkpoint, device)
    zuna_wrapper = cfg.build().to(device).eval()

    x = torch.randn(args.batch_size, args.channels, args.time, device=device)
    channel_positions = make_channel_positions(args.batch_size, args.channels, device)

    x_preprocessed = zuna_wrapper._preprocess_eeg(x)
    encoder_input, seq_lens, tok_idx = zuna_wrapper._sequence_repacking(
        x_preprocessed,
        channel_positions,
    )

    raw_latent, raw_tok_idx_reg, raw_losses = raw_ay2latent.encoder(
        token_values=encoder_input.unsqueeze(0),
        seq_lens=seq_lens,
        tok_idx=tok_idx,
        attn_impl="flex_attention",
    )
    raw_outputs = torch.stack(
        raw_latent.squeeze(0).split(seq_lens.detach().cpu().tolist(), dim=0),
        dim=0,
    )

    wrapped_outputs = zuna_wrapper(x, channel_positions)

    diff = (raw_outputs - wrapped_outputs).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    allclose = torch.allclose(raw_outputs, wrapped_outputs, rtol=1e-3, atol=1e-3)
    raw_encoder_params, raw_encoder_norm = module_stats(raw_ay2latent.encoder)
    raw_decoder_params, raw_decoder_norm = module_stats(raw_ay2latent.decoder)
    wrapped_encoder_params, wrapped_encoder_norm = module_stats(zuna_wrapper.encoder)

    print("Loaded full AY2latent model:", type(raw_ay2latent).__name__)
    print("Raw AY2latent encoder params/norm:", raw_encoder_params, f"{raw_encoder_norm:.8g}")
    print("Raw AY2latent decoder params/norm:", raw_decoder_params, f"{raw_decoder_norm:.8g}")
    print("Loaded NeuroAI ZUNA wrapper:", type(zuna_wrapper).__name__)
    print("Wrapped encoder params/norm:", wrapped_encoder_params, f"{wrapped_encoder_norm:.8g}")
    print("Checkpoint:", args.checkpoint)
    print("Input x:", tuple(x.shape))
    print("channel_positions:", tuple(channel_positions.shape))
    print("encoder_input:", tuple(encoder_input.shape))
    print("seq_lens:", seq_lens.detach().cpu().tolist())
    print("tok_idx:", tuple(tok_idx.shape))
    print("raw_latent:", tuple(raw_latent.shape))
    print("raw_tok_idx_reg:", tuple(raw_tok_idx_reg.shape))
    print("raw_losses keys:", sorted(raw_losses.keys()))
    print("raw_outputs:", tuple(raw_outputs.shape), raw_outputs.dtype, raw_outputs.device)
    print("wrapped_outputs:", tuple(wrapped_outputs.shape), wrapped_outputs.dtype, wrapped_outputs.device)
    print(f"max_abs_diff: {max_abs:.8g}")
    print(f"mean_abs_diff: {mean_abs:.8g}")
    print("allclose rtol=1e-3 atol=1e-3:", allclose)

    if not allclose:
        raise SystemExit(1)


def run_real_recon_plot(args: argparse.Namespace) -> None:
    _require_ay2latent()
    from apps.AY2latent_bci.eeg_data import (
        BCIDatasetArgs,
        EEGProcessor,
        create_dataloader_v2,
    )
    from lingua.args import dataclass_from_dict
    from omegaconf import OmegaConf

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    cfg.model.encoder_latent_downsample_factor = 1
    cfg.model.ape_dim = 0
    cfg.data.data_dir = str(args.data_dir)
    cfg.data.batch_size = 1
    cfg.data.target_packed_seqlen = 10000
    cfg.data.num_workers = 0
    cfg.data.prefetch_factor = None
    cfg.data.persistent_workers = False
    cfg.data.pin_memory = False
    cfg.data.shuffle = False
    cfg.data.token_dropout_prob = -1.0
    cfg.data.dropout_scheme = "no-dropout"
    cfg.data.sample_duration_str = "5_seconds"
    cfg.data.z_score_type = "across_channel"

    data_args = dataclass_from_dict(
        BCIDatasetArgs,
        OmegaConf.to_container(cfg.data, resolve=True),
    )
    dataloader = create_dataloader_v2(data_args, seed=args.seed, rank=0)
    raw_batch = next(iter(dataloader))
    raw_batch.pop("ids", None)
    raw_batch.pop("idx", None)
    raw_batch.pop("dataset_id", None)

    raw_batch["eeg_signal"] = raw_batch["eeg_signal"] / data_args.data_norm
    if data_args.data_clip is not None:
        raw_batch["eeg_signal"] = raw_batch["eeg_signal"].clamp(
            min=-data_args.data_clip,
            max=data_args.data_clip,
        )

    masked_channel_ids = mask_first_channels(raw_batch, args.mask_channels)
    processor = EEGProcessor(data_args)
    batch = processor.process(**raw_batch)

    device = torch.device(args.device)
    batch = {k: v.to(device) for k, v in batch.items()}
    tok_idx = make_tok_idx(batch, zero_spatial=bool(cfg.model.zero_spatial)).to(device)
    model = build_reconstruction_model(
        OmegaConf.to_container(cfg.model, resolve=True),
        args.checkpoint,
        device,
    )

    z, inference_steps = model.sample(
        encoder_input=batch["encoder_input"].unsqueeze(0),
        seq_lens=batch["seq_lens"],
        tok_idx=tok_idx,
        cfg=1.0,
        sample_steps=args.sample_steps,
    )

    plot_indices = list(
        range(max(0, len(inference_steps) - args.plot_last_n), len(inference_steps))
    )
    metrics: list[tuple[str, float, float]] = []
    for step_idx in plot_indices:
        tag = f"step{step_idx + 1:03d}_of_{args.sample_steps:03d}"
        step_nmse, step_pcc = plot_reconstruction(
            model_output=inference_steps[step_idx],
            batch=batch,
            data_args=data_args,
            encoder_output_dim=int(cfg.model.encoder_output_dim),
            out_dir=args.out_dir,
            tag=tag,
        )
        metrics.append((tag, step_nmse, step_pcc))

    final_nmse, final_pcc = plot_reconstruction(
        model_output=z,
        batch=batch,
        data_args=data_args,
        encoder_output_dim=int(cfg.model.encoder_output_dim),
        out_dir=args.out_dir,
        tag="final",
    )
    metrics.append(("final", final_nmse, final_pcc))

    print("Saved reconstruction plots to:", args.out_dir)
    print("Masked channel ids:", masked_channel_ids)
    print("raw packed eeg_signal:", tuple(raw_batch["eeg_signal"].shape))
    print("processed encoder_input:", tuple(batch["encoder_input"].shape))
    print("seq_lens:", batch["seq_lens"].detach().cpu().tolist())
    print("tok_idx:", tuple(tok_idx.shape))
    print("reconstruction z:", tuple(z.shape))
    for tag, nmse, pcc in metrics:
        print(f"{tag}: NMSE(original,recon)={nmse:.8g} PCC(original,recon)={pcc:.8g}")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke-compare",
        help="Compare NeuroAI wrapper output against the raw AY2latent encoder.",
    )
    smoke.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--channels", type=int, default=8)
    smoke.add_argument("--time", type=int, default=64)
    smoke.add_argument("--seed", type=int, default=1234)
    smoke.add_argument("--device", default="cuda")
    smoke.set_defaults(func=run_smoke_compare)

    recon = subparsers.add_parser(
        "real-recon-plot",
        help="Run ZUNA decoder reconstruction on one real sample and save plots.",
    )
    recon.add_argument(
        "--config",
        type=Path,
        default=AY2LATENT_BCI / "configs" / "config_bci_eval.yaml",
    )
    recon.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    recon.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data/groups/bci/datasets/v7_evalB"),
    )
    recon.add_argument(
        "--out-dir",
        type=Path,
        default=WORKSPACE_ROOT / "tmp_zuna_recon_plots",
    )
    recon.add_argument("--sample-steps", type=int, default=50)
    recon.add_argument("--mask-channels", type=int, default=3)
    recon.add_argument("--plot-last-n", type=int, default=5)
    recon.add_argument("--seed", type=int, default=316)
    recon.add_argument("--device", default="cuda")
    recon.set_defaults(func=run_real_recon_plot)
    return parser


def test_make_channel_positions_shape() -> None:
    pos = make_channel_positions(2, 8, torch.device("cpu"))

    assert pos.shape == (2, 8, 3)
    torch.testing.assert_close(pos[0], pos[1])


def test_zuna_preprocess_normalizes_and_clips() -> None:
    model = ZUNAEncoder(torch.nn.Identity(), num_fine_time_pts=32, data_norm=1.0)
    x = torch.randn(2, 4, 64) * 10.0

    out = model._preprocess_eeg(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_zuna_forward_validates_time_divisibility() -> None:
    model = ZUNAEncoder(torch.nn.Identity(), num_fine_time_pts=32)
    x = torch.randn(1, 4, 65)
    channel_positions = torch.randn(1, 4, 3)

    try:
        model(x, channel_positions)
    except ValueError as exc:
        assert "time dimension" in str(exc)
    else:
        raise AssertionError("Expected ValueError for indivisible time dimension")


if __name__ == "__main__":
    parsed_args = _make_parser().parse_args()
    parsed_args.func(parsed_args)
