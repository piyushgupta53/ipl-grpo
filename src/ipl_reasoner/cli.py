"""Small CLI for recurring project tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from ipl_reasoner.constants import (
    AS_OF_SEASONS,
    RAW_DELIVERIES_FILENAME,
    RAW_MATCHES_FILENAME,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
)
from ipl_reasoner.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipl-reasoner",
        description="Utility commands for the IPL win probability reasoner project.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "init-workspace",
        help="Create the expected project directories for raw data and generated artifacts.",
    )
    subparsers.add_parser(
        "doctor",
        help="Show current workspace status and check whether the expected raw files exist.",
    )
    subparsers.add_parser(
        "validate-raw-data",
        help="Load raw CSVs, detect schema variants, and print a validation report.",
    )
    subparsers.add_parser(
        "normalize-raw-data",
        help="Write canonicalized raw CSVs into data/interim/ for downstream pipeline work.",
    )
    subparsers.add_parser(
        "build-cleaned-data",
        help="Apply hard exclusions and write cleaned canonical datasets plus merged deliveries.",
    )
    subparsers.add_parser(
        "build-player-season-stats",
        help="Build pre-season player prior tables and league-average fallbacks.",
    )
    subparsers.add_parser(
        "build-venue-artifacts",
        help="Build venue alias report, venue metadata, and pre-season venue prior tables.",
    )
    subparsers.add_parser(
        "build-snapshots",
        help="Build structured end-of-over snapshots and the canonical baseline artifact.",
    )
    subparsers.add_parser(
        "build-training-dataset",
        help="Build the final prompt-bearing training dataset from snapshots and priors.",
    )
    subparsers.add_parser(
        "audit-training-dataset",
        help="Run QA checks on the final training dataset and write a JSON audit report.",
    )
    subparsers.add_parser(
        "build-sft-artifacts",
        help="Select recent SFT warmup candidates and export draft SFT data for review.",
    )
    sft_run_parser = subparsers.add_parser(
        "run-sft-warmup",
        help="Write the SFT warmup manifest or run the warmup training scaffold.",
    )
    sft_run_parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model to use for the warmup run.",
    )
    sft_run_parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory for the warmup checkpoint and manifest.",
    )
    sft_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write the manifest and print the planned training inputs.",
    )
    sft_run_parser.add_argument(
        "--dataset-variant",
        default="first_pass_all",
        choices=["first_pass_all", "reviewed_only", "review_pack", "draft"],
        help="Which SFT dataset variant to train on.",
    )
    grpo_run_parser = subparsers.add_parser(
        "run-grpo",
        help="Write GRPO artifacts/manifest or run the GRPO training scaffold.",
    )
    grpo_run_parser.add_argument(
        "--model",
        default="artifacts/sft/sft_warmup",
        help="SFT checkpoint or base model to start GRPO from.",
    )
    grpo_run_parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory for the GRPO checkpoint and manifest.",
    )
    grpo_run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only prepare datasets/manifests without starting training.",
    )
    return parser


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def cmd_init_workspace() -> int:
    paths = ProjectPaths.discover()
    created = paths.ensure()

    print("Initialized workspace directories:")
    for path in created:
        print(f"- {_relative_to_root(path, paths.root)}")
    return 0


def cmd_doctor() -> int:
    paths = ProjectPaths.discover()
    matches_path = paths.raw / RAW_MATCHES_FILENAME
    deliveries_path = paths.raw / RAW_DELIVERIES_FILENAME

    print("Project root:", paths.root)
    print("Training seasons:", ", ".join(TRAIN_SEASONS))
    print("Validation seasons:", ", ".join(VALIDATION_SEASONS))
    print("Test seasons:", ", ".join(TEST_SEASONS))
    print("As-of seasons:", f"{AS_OF_SEASONS[0]}-{AS_OF_SEASONS[-1]}")
    print("Raw matches CSV:", _status_line(matches_path, paths.root))
    print("Raw deliveries CSV:", _status_line(deliveries_path, paths.root))

    if matches_path.exists() and deliveries_path.exists():
        print("Status: ready for raw-data validation and schema normalization.")
    else:
        print("Status: waiting on raw CSVs in data/raw/.")
    return 0


def cmd_validate_raw_data() -> int:
    paths = ProjectPaths.discover()
    matches_path = paths.raw / RAW_MATCHES_FILENAME
    deliveries_path = paths.raw / RAW_DELIVERIES_FILENAME

    if not matches_path.exists() or not deliveries_path.exists():
        print("Raw CSVs are missing. Expected:")
        print("-", _relative_to_root(matches_path, paths.root))
        print("-", _relative_to_root(deliveries_path, paths.root))
        return 1

    from ipl_reasoner.raw_data import build_validation_summary, load_raw_deliveries, load_raw_matches
    from ipl_reasoner.raw_data import validate_raw_tables

    matches = load_raw_matches(paths)
    deliveries = load_raw_deliveries(paths)

    reports = validate_raw_tables(matches, deliveries)
    print(build_validation_summary(reports))
    return 0 if all(report.is_valid for report in reports) else 1


def cmd_normalize_raw_data() -> int:
    paths = ProjectPaths.discover()
    matches_path = paths.raw / RAW_MATCHES_FILENAME
    deliveries_path = paths.raw / RAW_DELIVERIES_FILENAME

    if not matches_path.exists() or not deliveries_path.exists():
        print("Raw CSVs are missing. Expected:")
        print("-", _relative_to_root(matches_path, paths.root))
        print("-", _relative_to_root(deliveries_path, paths.root))
        return 1

    from ipl_reasoner.raw_data import (
        build_validation_summary,
        load_raw_deliveries,
        load_raw_matches,
        write_canonical_raw_outputs,
    )

    matches = load_raw_matches(paths)
    deliveries = load_raw_deliveries(paths)
    matches_out, deliveries_out, reports = write_canonical_raw_outputs(matches, deliveries, paths)

    print(build_validation_summary(reports))
    print("Wrote canonical files:")
    print("-", _relative_to_root(matches_out, paths.root))
    print("-", _relative_to_root(deliveries_out, paths.root))
    return 0


def cmd_build_cleaned_data() -> int:
    from ipl_reasoner.preprocess import build_clean_datasets
    from ipl_reasoner.raw_data import (
        build_validation_summary,
        load_raw_deliveries,
        load_raw_matches,
        normalize_raw_tables,
    )

    paths = ProjectPaths.discover()
    matches_path = paths.raw / RAW_MATCHES_FILENAME
    deliveries_path = paths.raw / RAW_DELIVERIES_FILENAME

    if not matches_path.exists() or not deliveries_path.exists():
        print("Raw CSVs are missing. Expected:")
        print("-", _relative_to_root(matches_path, paths.root))
        print("-", _relative_to_root(deliveries_path, paths.root))
        return 1

    matches = load_raw_matches(paths)
    deliveries = load_raw_deliveries(paths)
    canonical_matches, canonical_deliveries, reports = normalize_raw_tables(matches, deliveries)
    artifacts = build_clean_datasets(canonical_matches, canonical_deliveries, paths)

    print("Canonical schema check:")
    print(_indent_block(build_validation_summary(reports)))
    print("Wrote cleaned outputs:")
    print("-", _relative_to_root(artifacts.matches_clean_path, paths.root))
    print("-", _relative_to_root(artifacts.deliveries_clean_path, paths.root))
    print("-", _relative_to_root(artifacts.merged_deliveries_path, paths.root))
    print("-", _relative_to_root(artifacts.exclusions_path, paths.root))
    return 0


def cmd_build_player_season_stats() -> int:
    from ipl_reasoner.player_stats import build_player_season_stats

    paths = ProjectPaths.discover()
    merged_path = paths.processed / "merged_deliveries.csv"
    if not merged_path.exists():
        print("Missing cleaned merged deliveries. Run `build-cleaned-data` first.")
        return 1

    import pandas as pd

    merged = pd.read_csv(merged_path)
    artifacts = build_player_season_stats(merged, paths)
    print("Wrote player prior artifacts:")
    print("-", _relative_to_root(artifacts.player_stats_path, paths.root))
    print("-", _relative_to_root(artifacts.league_averages_path, paths.root))
    return 0


def cmd_build_venue_artifacts() -> int:
    import pandas as pd

    from ipl_reasoner.venue_data import build_venue_artifacts

    paths = ProjectPaths.discover()
    matches_path = paths.processed / "matches_clean.csv"
    merged_path = paths.processed / "merged_deliveries.csv"
    if not matches_path.exists() or not merged_path.exists():
        print("Missing cleaned datasets. Run `build-cleaned-data` first.")
        return 1

    matches = pd.read_csv(matches_path)
    merged = pd.read_csv(merged_path)
    artifacts = build_venue_artifacts(matches, merged, paths)
    print("Wrote venue artifacts:")
    print("-", _relative_to_root(artifacts.venue_metadata_path, paths.root))
    print("-", _relative_to_root(artifacts.venue_season_stats_path, paths.root))
    print("-", _relative_to_root(artifacts.venue_alias_report_path, paths.root))
    return 0


def cmd_build_snapshots() -> int:
    import pandas as pd

    from ipl_reasoner.snapshots import build_snapshot_dataset_and_baseline

    paths = ProjectPaths.discover()
    matches_path = paths.processed / "matches_clean.csv"
    merged_path = paths.processed / "merged_deliveries.csv"
    if not matches_path.exists() or not merged_path.exists():
        print("Missing cleaned datasets. Run `build-cleaned-data` first.")
        return 1

    matches = pd.read_csv(matches_path)
    merged = pd.read_csv(merged_path)
    artifacts = build_snapshot_dataset_and_baseline(matches, merged, paths)
    print("Wrote snapshot and baseline artifacts:")
    print("-", _relative_to_root(artifacts.snapshot_dataset_path, paths.root))
    print("-", _relative_to_root(artifacts.baseline_model_path, paths.root))
    print("-", _relative_to_root(artifacts.baseline_metadata_path, paths.root))
    return 0


def cmd_build_training_dataset() -> int:
    import pandas as pd

    from ipl_reasoner.training_dataset import build_training_dataset

    paths = ProjectPaths.discover()
    required_paths = {
        "snapshots": paths.processed / "snapshot_dataset.csv",
        "matches": paths.processed / "matches_clean.csv",
        "player_stats": paths.metadata / "player_season_stats.csv",
        "league_avgs": paths.metadata / "league_avg_by_season.csv",
        "venue_stats": paths.metadata / "venue_season_stats.csv",
        "venue_metadata": paths.metadata / "venue_metadata.csv",
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        print("Missing prerequisite artifacts:", ", ".join(missing))
        print("Run the earlier build commands first.")
        return 1

    artifacts = build_training_dataset(
        snapshots=pd.read_csv(required_paths["snapshots"]),
        matches=pd.read_csv(required_paths["matches"]),
        player_stats=pd.read_csv(required_paths["player_stats"]),
        league_avgs=pd.read_csv(required_paths["league_avgs"]),
        venue_stats=pd.read_csv(required_paths["venue_stats"]),
        venue_metadata=pd.read_csv(required_paths["venue_metadata"]),
        paths=paths,
    )
    print("Wrote training dataset:")
    print("-", _relative_to_root(artifacts.training_dataset_path, paths.root))
    return 0


def cmd_audit_training_dataset() -> int:
    import pandas as pd

    from ipl_reasoner.qa import audit_training_dataset

    paths = ProjectPaths.discover()
    dataset_path = paths.processed / "training_dataset.csv"
    if not dataset_path.exists():
        print("Missing training dataset. Run `build-training-dataset` first.")
        return 1

    audit_path = audit_training_dataset(pd.read_csv(dataset_path), paths)
    print("Wrote training dataset audit:")
    print("-", _relative_to_root(audit_path, paths.root))
    return 0


def cmd_build_sft_artifacts() -> int:
    import pandas as pd

    from ipl_reasoner.sft import build_sft_artifacts

    paths = ProjectPaths.discover()
    dataset_path = paths.processed / "training_dataset.csv"
    if not dataset_path.exists():
        print("Missing training dataset. Run `build-training-dataset` first.")
        return 1

    artifacts = build_sft_artifacts(pd.read_csv(dataset_path), paths)
    print("Wrote SFT prep artifacts:")
    print("-", _relative_to_root(artifacts.candidate_csv_path, paths.root))
    print("-", _relative_to_root(artifacts.draft_jsonl_path, paths.root))
    print("-", _relative_to_root(artifacts.review_pack_csv_path, paths.root))
    print("-", _relative_to_root(artifacts.reviewed_jsonl_path, paths.root))
    print("-", _relative_to_root(artifacts.reviewed_only_jsonl_path, paths.root))
    print("-", _relative_to_root(artifacts.first_pass_all_jsonl_path, paths.root))
    print("-", _relative_to_root(artifacts.warmup_training_jsonl_path, paths.root))
    return 0


def cmd_run_sft_warmup(model: str, output_dir: str, dataset_variant: str, dry_run: bool) -> int:
    from ipl_reasoner.sft_train import run_sft_warmup

    paths = ProjectPaths.discover()
    warmup_path = paths.processed / "sft_warmup_training.jsonl"
    if not warmup_path.exists():
        print("Missing SFT warmup dataset. Run `build-sft-artifacts` first.")
        return 1

    target_dir = Path(output_dir) if output_dir else None
    try:
        manifest_path = run_sft_warmup(
            paths=paths,
            model_name=model,
            output_dir=target_dir,
            dataset_variant=dataset_variant,
            dry_run=dry_run,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1

    if dry_run:
        print("Wrote SFT warmup manifest:")
    else:
        print("SFT warmup finished. Manifest:")
    print("-", _relative_to_root(manifest_path, paths.root))
    return 0


def cmd_run_grpo(model: str, output_dir: str, dry_run: bool) -> int:
    from ipl_reasoner.grpo_train import run_grpo_training

    paths = ProjectPaths.discover()
    target_dir = Path(output_dir) if output_dir else None
    try:
        manifest_path = run_grpo_training(
            paths=paths,
            model_name_or_path=model,
            output_dir=target_dir,
            dry_run=dry_run,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    if dry_run:
        print("Wrote GRPO manifest:")
    else:
        print("GRPO training finished. Manifest:")
    print("-", _relative_to_root(manifest_path, paths.root))
    return 0


def _status_line(path: Path, root: Path) -> str:
    state = "present" if path.exists() else "missing"
    return f"{_relative_to_root(path, root)} ({state})"


def _indent_block(text: str) -> str:
    return "\n".join(f"  {line}" for line in text.splitlines())


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init-workspace":
        return cmd_init_workspace()
    if args.command == "doctor":
        return cmd_doctor()
    if args.command == "validate-raw-data":
        return cmd_validate_raw_data()
    if args.command == "normalize-raw-data":
        return cmd_normalize_raw_data()
    if args.command == "build-cleaned-data":
        return cmd_build_cleaned_data()
    if args.command == "build-player-season-stats":
        return cmd_build_player_season_stats()
    if args.command == "build-venue-artifacts":
        return cmd_build_venue_artifacts()
    if args.command == "build-snapshots":
        return cmd_build_snapshots()
    if args.command == "build-training-dataset":
        return cmd_build_training_dataset()
    if args.command == "audit-training-dataset":
        return cmd_audit_training_dataset()
    if args.command == "build-sft-artifacts":
        return cmd_build_sft_artifacts()
    if args.command == "run-sft-warmup":
        return cmd_run_sft_warmup(
            model=args.model,
            output_dir=args.output_dir,
            dataset_variant=args.dataset_variant,
            dry_run=args.dry_run,
        )
    if args.command == "run-grpo":
        return cmd_run_grpo(model=args.model, output_dir=args.output_dir, dry_run=args.dry_run)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
