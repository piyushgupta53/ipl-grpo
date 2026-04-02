"""QA checks for generated training artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ipl_reasoner.constants import TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS
from ipl_reasoner.paths import ProjectPaths


def audit_training_dataset(training_dataset: pd.DataFrame, paths: ProjectPaths) -> Path:
    paths.ensure()
    df = training_dataset.copy()

    duplicate_keys = int(df.duplicated(subset=["match_id", "snapshot_over"]).sum())
    standalone_nan_prompts = int(df["prompt"].str.contains(r"\bnan\b", case=False, regex=True).sum())
    prompt_char_lengths = df["prompt"].str.len()
    prompt_word_lengths = df["prompt"].str.split().str.len()
    prompt_token_audit = _build_prompt_token_audit(df["prompt"], paths)

    train = df.loc[df["split"] == "train"].copy()
    train_season_ok = bool(set(train["season"].astype(str)).issubset(set(TRAIN_SEASONS)))
    validation_season_ok = bool(
        set(df.loc[df["split"] == "validation", "season"].astype(str)).issubset(set(VALIDATION_SEASONS))
    )
    test_season_ok = bool(set(df.loc[df["split"] == "test", "season"].astype(str)).issubset(set(TEST_SEASONS)))

    cooldown_violations = _count_cooldown_violations(train, cooldown=8)
    adjacent_duplicates = _count_adjacent_duplicate_match_ids(train)
    max_repeats_in_16 = _max_match_repeats_in_window(train, window_size=16)

    report = {
        "row_count": int(len(df)),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().sort_index().items()},
        "duplicate_snapshot_keys": duplicate_keys,
        "standalone_nan_prompts": standalone_nan_prompts,
        "prompt_char_length": {
            "min": int(prompt_char_lengths.min()),
            "max": int(prompt_char_lengths.max()),
            "mean": float(prompt_char_lengths.mean()),
        },
        "prompt_word_length": {
            "min": int(prompt_word_lengths.min()),
            "max": int(prompt_word_lengths.max()),
            "mean": float(prompt_word_lengths.mean()),
        },
        "train_season_ok": train_season_ok,
        "validation_season_ok": validation_season_ok,
        "test_season_ok": test_season_ok,
        "train_cooldown_violations_within_8": cooldown_violations,
        "train_adjacent_duplicate_match_ids": adjacent_duplicates,
        "train_max_match_repeats_in_any_16": max_repeats_in_16,
        "train_plan_cooldown_ok": bool(adjacent_duplicates == 0 and max_repeats_in_16 <= 2),
        "prompt_token_audit": prompt_token_audit,
    }

    output_path = paths.reports / "training_dataset_audit.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def _count_cooldown_violations(train_df: pd.DataFrame, cooldown: int) -> int:
    match_ids = train_df["match_id"].astype(str).tolist()
    violations = 0
    for idx, match_id in enumerate(match_ids):
        window = match_ids[max(0, idx - cooldown) : idx]
        if match_id in window:
            violations += 1
    return violations


def _count_adjacent_duplicate_match_ids(train_df: pd.DataFrame) -> int:
    match_ids = train_df["match_id"].astype(str).tolist()
    return sum(1 for idx in range(1, len(match_ids)) if match_ids[idx] == match_ids[idx - 1])


def _max_match_repeats_in_window(train_df: pd.DataFrame, window_size: int) -> int:
    match_ids = train_df["match_id"].astype(str).tolist()
    if not match_ids:
        return 0

    max_repeat = 1
    for idx in range(max(1, len(match_ids) - window_size + 1)):
        counts = pd.Series(match_ids[idx : idx + window_size]).value_counts()
        max_repeat = max(max_repeat, int(counts.max()))
    return max_repeat


def _build_prompt_token_audit(prompts: pd.Series, paths: ProjectPaths) -> dict[str, object]:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return {
            "tokenizer_model": None,
            "available": False,
            "reason": "transformers_not_installed",
        }

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    local_tokenizer_dir = paths.artifacts / "tokenizers" / "qwen2.5-1.5b-instruct"
    try:
        if local_tokenizer_dir.exists():
            tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_dir, local_files_only=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "tokenizer_model": str(local_tokenizer_dir if local_tokenizer_dir.exists() else model_name),
            "available": False,
            "reason": f"tokenizer_unavailable: {exc.__class__.__name__}",
        }

    token_lengths = prompts.apply(lambda prompt: len(tokenizer(prompt, add_special_tokens=True)["input_ids"]))
    return {
        "tokenizer_model": str(local_tokenizer_dir if local_tokenizer_dir.exists() else model_name),
        "available": True,
        "min": int(token_lengths.min()),
        "max": int(token_lengths.max()),
        "mean": float(token_lengths.mean()),
        "over_512": int((token_lengths > 512).sum()),
        "all_under_512": bool((token_lengths <= 512).all()),
    }
