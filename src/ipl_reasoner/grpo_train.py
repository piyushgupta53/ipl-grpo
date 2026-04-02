"""GRPO dataset preparation, reward functions, and training scaffold."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ipl_reasoner.paths import ProjectPaths

DEFAULT_GRPO_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_GRPO_OUTPUT_DIRNAME = "grpo_run"
DEFAULT_SFT_CHECKPOINT = "artifacts/sft/sft_warmup"

FORMAT_REWARD_WEIGHT = 0.15
RATIONALE_REWARD_WEIGHT = 0.10
ACCURACY_REWARD_WEIGHT = 0.75


@dataclass(frozen=True)
class GRPOArtifacts:
    train_jsonl_path: Path
    validation_jsonl_path: Path
    test_jsonl_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class GRPOManifest:
    model_name_or_path: str
    train_dataset_path: str
    validation_dataset_path: str
    test_dataset_path: str
    output_dir: str
    baseline_version: str
    baseline_artifact_path: str
    reward_weights: dict[str, float]
    num_train_examples: int
    num_validation_examples: int
    num_test_examples: int
    max_prompt_length: int = 512
    max_completion_length: int = 256
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_generations: int = 4
    learning_rate: float = 2e-6
    num_train_epochs: int = 3
    temperature: float = 0.7
    beta: float = 0.04
    epsilon: float = 0.2
    max_grad_norm: float = 0.1
    logging_steps: int = 5
    save_steps: int = 100


def prepare_grpo_artifacts(
    paths: ProjectPaths,
    model_name_or_path: str | None = None,
    output_dir: Path | None = None,
) -> GRPOArtifacts:
    paths.ensure()
    model_name_or_path = model_name_or_path or DEFAULT_SFT_CHECKPOINT
    output_dir = output_dir or (paths.artifacts / "grpo" / DEFAULT_GRPO_OUTPUT_DIRNAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = paths.processed / "training_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError("Missing training dataset. Run `build-training-dataset` first.")

    baseline_metadata_path = paths.baseline_artifacts / "baseline_v1.metadata.json"
    if not baseline_metadata_path.exists():
        raise FileNotFoundError("Missing baseline metadata. Run `build-snapshots` first.")

    training_dataset = pd.read_csv(dataset_path)
    export_cols = [
        "prompt",
        "did_chasing_team_win",
        "run_rate_baseline_prob",
        "match_id",
        "season",
        "snapshot_over",
        "required_run_rate",
        "wickets_in_hand",
        "partnership_runs",
        "prompt_version",
        "venue_code",
        "is_any_player_estimated",
    ]
    train_df = training_dataset.loc[training_dataset["split"] == "train", export_cols].copy()
    validation_df = training_dataset.loc[training_dataset["split"] == "validation", export_cols].copy()
    test_df = training_dataset.loc[training_dataset["split"] == "test", export_cols].copy()

    train_jsonl_path = paths.processed / "grpo_train.jsonl"
    validation_jsonl_path = paths.processed / "grpo_validation.jsonl"
    test_jsonl_path = paths.processed / "grpo_test.jsonl"
    _write_jsonl(train_df, train_jsonl_path)
    _write_jsonl(validation_df, validation_jsonl_path)
    _write_jsonl(test_df, test_jsonl_path)

    baseline_metadata = json.loads(baseline_metadata_path.read_text(encoding="utf-8"))
    manifest = GRPOManifest(
        model_name_or_path=str(model_name_or_path),
        train_dataset_path=str(train_jsonl_path),
        validation_dataset_path=str(validation_jsonl_path),
        test_dataset_path=str(test_jsonl_path),
        output_dir=str(output_dir),
        baseline_version=str(baseline_metadata["baseline_version"]),
        baseline_artifact_path=str(paths.baseline_artifacts / "baseline_v1.joblib"),
        reward_weights={
            "format_reward": FORMAT_REWARD_WEIGHT,
            "rationale_reward": RATIONALE_REWARD_WEIGHT,
            "accuracy_reward": ACCURACY_REWARD_WEIGHT,
        },
        num_train_examples=int(len(train_df)),
        num_validation_examples=int(len(validation_df)),
        num_test_examples=int(len(test_df)),
    )
    manifest_path = output_dir / "grpo_manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")

    return GRPOArtifacts(
        train_jsonl_path=train_jsonl_path,
        validation_jsonl_path=validation_jsonl_path,
        test_jsonl_path=test_jsonl_path,
        manifest_path=manifest_path,
    )


def run_grpo_training(
    paths: ProjectPaths,
    model_name_or_path: str | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> Path:
    artifacts = prepare_grpo_artifacts(paths=paths, model_name_or_path=model_name_or_path, output_dir=output_dir)
    if dry_run:
        return artifacts.manifest_path

    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover - depends on training runtime
        raise RuntimeError(
            "Missing GRPO training dependencies. Install the pinned training stack in a GPU runtime before running this command."
        ) from exc

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    train_dataset = load_dataset("json", data_files=manifest["train_dataset_path"], split="train")
    eval_dataset = load_dataset("json", data_files=manifest["validation_dataset_path"], split="train")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=manifest["output_dir"],
        per_device_train_batch_size=int(manifest["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(manifest["gradient_accumulation_steps"]),
        num_generations=int(manifest["num_generations"]),
        max_prompt_length=int(manifest["max_prompt_length"]),
        max_completion_length=int(manifest["max_completion_length"]),
        learning_rate=float(manifest["learning_rate"]),
        num_train_epochs=int(manifest["num_train_epochs"]),
        temperature=float(manifest["temperature"]),
        beta=float(manifest["beta"]),
        epsilon=float(manifest["epsilon"]),
        max_grad_norm=float(manifest["max_grad_norm"]),
        logging_steps=int(manifest["logging_steps"]),
        save_steps=int(manifest["save_steps"]),
        report_to="none",
        log_completions=True,
        num_completions_to_print=2,
        fp16=True,
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=manifest["model_name_or_path"],
        args=training_args,
        reward_funcs=[format_reward, rationale_reward, accuracy_reward],
        reward_weights=[
            FORMAT_REWARD_WEIGHT,
            RATIONALE_REWARD_WEIGHT,
            ACCURACY_REWARD_WEIGHT,
        ],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(manifest["output_dir"])
    return artifacts.manifest_path


def format_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    rewards: list[float] = []
    for completion in completions:
        text = _completion_to_text(completion)
        has_analysis = bool(re.search(r"<analysis>.*?</analysis>", text, re.DOTALL))
        answer_match = re.search(r"<answer>([\d.]+)</answer>", text)

        if has_analysis and answer_match:
            try:
                probability = float(answer_match.group(1))
            except ValueError:
                rewards.append(0.0)
                continue
            rewards.append(1.0 if 0.0 <= probability <= 1.0 else 0.5)
        elif has_analysis or answer_match:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def rationale_reward(completions: list[Any], **kwargs: Any) -> list[float]:
    banned_generic_phrases = [
        "it is a close game",
        "anything can happen",
        "both teams have a chance",
        "it depends on many factors",
    ]
    rewards: list[float] = []
    for completion in completions:
        text = _completion_to_text(completion)
        match = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        analysis = match.group(1).strip()
        words = analysis.split()
        word_count = len(words)
        sentence_count = len([chunk for chunk in re.split(r"[.!?]+", analysis) if chunk.strip()])
        lower = analysis.lower()
        has_number = bool(re.search(r"\d", analysis))
        generic_penalty = any(phrase in lower for phrase in banned_generic_phrases)
        repeated_bigram_penalty = _has_repeated_bigram(words)

        if 35 <= word_count <= 120 and sentence_count >= 2 and has_number and not generic_penalty and not repeated_bigram_penalty:
            rewards.append(1.0)
        elif 20 <= word_count <= 140 and sentence_count >= 1 and not generic_penalty:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def accuracy_reward(
    completions: list[Any],
    did_chasing_team_win: list[Any],
    run_rate_baseline_prob: list[Any],
    **kwargs: Any,
) -> list[float]:
    rewards: list[float] = []
    for completion, actual_outcome, baseline_prob in zip(
        completions,
        did_chasing_team_win,
        run_rate_baseline_prob,
        strict=False,
    ):
        predicted = _extract_probability(_completion_to_text(completion))
        if predicted is None:
            rewards.append(0.0)
            continue

        predicted = max(0.01, min(0.99, predicted))
        actual = float(actual_outcome)
        baseline = float(baseline_prob)

        brier = (predicted - actual) ** 2
        brier_reward = 1.0 - brier

        is_correct_direction = (predicted > 0.5 and actual == 1.0) or (predicted < 0.5 and actual == 0.0)
        confidence = abs(predicted - 0.5) * 2.0
        confidence_bonus = 0.1 * confidence if is_correct_direction else 0.0

        baseline_brier = (baseline - actual) ** 2
        beat_baseline = brier < baseline_brier
        diverged_from_baseline = abs(predicted - baseline) > 0.15
        baseline_bonus = 0.05 if beat_baseline and diverged_from_baseline else 0.0

        rewards.append(brier_reward + confidence_bonus + baseline_bonus)
    return rewards


def _write_jsonl(df: pd.DataFrame, output_path: Path) -> None:
    records = df.to_dict("records")
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content")
                if content is not None:
                    parts.append(str(content))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _extract_probability(text: str) -> float | None:
    match = re.search(r"<answer>([\d.]+)</answer>", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _has_repeated_bigram(words: list[str]) -> bool:
    if len(words) < 8:
        return False
    bigrams = [" ".join(words[idx : idx + 2]).lower() for idx in range(len(words) - 1)]
    repeats = len(bigrams) - len(set(bigrams))
    return repeats >= 3
