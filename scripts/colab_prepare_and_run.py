"""Colab helper for preparing manifests and optionally starting training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ipl_reasoner.cli import cmd_build_sft_artifacts, cmd_run_grpo, cmd_run_sft_warmup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="colab-prepare-and-run",
        description="Prepare Colab-friendly manifests and optionally launch SFT/GRPO training.",
    )
    parser.add_argument(
        "--stage",
        choices=["sft", "grpo", "both"],
        default="sft",
        help="Which training stage to prepare.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually start training instead of only writing manifests.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model for SFT, or SFT checkpoint for GRPO when --stage=grpo.",
    )
    parser.add_argument(
        "--grpo-model",
        default="artifacts/sft/sft_warmup",
        help="Model/checkpoint to use as the GRPO starting point when --stage=both or grpo.",
    )
    parser.add_argument(
        "--sft-output-dir",
        default="",
        help="Optional SFT output directory.",
    )
    parser.add_argument(
        "--sft-dataset-variant",
        default="gold_v1",
        choices=["gold_v1", "first_pass_all", "reviewed_only", "review_pack", "draft"],
        help="Which SFT dataset variant to train on.",
    )
    parser.add_argument(
        "--grpo-output-dir",
        default="",
        help="Optional GRPO output directory.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    root = Path.cwd()
    expected = root / "src" / "ipl_reasoner"
    if not expected.exists():
        parser.error("Run this script from the project root inside Colab.")

    # Rebuild these inside Colab so manifests point at Colab-local paths, not laptop paths.
    cmd_build_sft_artifacts()

    run = bool(args.run)

    if args.stage in {"sft", "both"}:
        status = cmd_run_sft_warmup(
            model=args.model,
            output_dir=args.sft_output_dir,
            dataset_variant=args.sft_dataset_variant,
            dry_run=not run,
        )
        if status != 0:
            return status

    if args.stage in {"grpo", "both"}:
        status = cmd_run_grpo(
            model=args.grpo_model,
            output_dir=args.grpo_output_dir,
            dry_run=not run,
        )
        if status != 0:
            return status

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
