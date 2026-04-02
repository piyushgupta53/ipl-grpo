"""SFT warmup training scaffold."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ipl_reasoner.paths import ProjectPaths

DEFAULT_SFT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_SFT_OUTPUT_DIRNAME = "sft_warmup"


@dataclass(frozen=True)
class SFTWarmupConfig:
    model_name: str
    dataset_path: str
    output_dir: str
    num_examples: int
    reviewed_examples: int
    draft_examples: int
    dataset_variant: str = "first_pass_all"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    learning_rate: float = 1e-4
    max_seq_length: int = 1024
    packing: bool = False
    gradient_checkpointing: bool = True
    report_to: str = "none"


def write_sft_warmup_manifest(
    paths: ProjectPaths,
    model_name: str = DEFAULT_SFT_MODEL,
    output_dir: Path | None = None,
    dataset_variant: str = "first_pass_all",
) -> Path:
    paths.ensure()
    output_dir = output_dir or (paths.artifacts / "sft" / DEFAULT_SFT_OUTPUT_DIRNAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_variants = {
        "first_pass_all": paths.processed / "sft_warmup_training.jsonl",
        "reviewed_only": paths.processed / "sft_warmup_reviewed_only.jsonl",
        "review_pack": paths.processed / "sft_warmup_reviewed.jsonl",
        "draft": paths.processed / "sft_warmup_drafts.jsonl",
    }
    if dataset_variant not in dataset_variants:
        raise ValueError(f"Unknown SFT dataset variant: {dataset_variant}")

    dataset_path = dataset_variants[dataset_variant]
    if not dataset_path.exists():
        raise FileNotFoundError("Missing SFT warmup dataset. Run `build-sft-artifacts` first.")

    total_examples = 0
    reviewed_examples = 0
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            total_examples += 1
            payload = json.loads(line)
            source = payload.get("metadata", {}).get("response_source")
            if source in {"first_pass_reviewed", "first_pass_all", "reviewed_only"}:
                reviewed_examples += 1

    config = SFTWarmupConfig(
        model_name=model_name,
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        num_examples=total_examples,
        reviewed_examples=reviewed_examples,
        draft_examples=total_examples - reviewed_examples,
        dataset_variant=dataset_variant,
    )

    manifest_path = output_dir / "sft_warmup_manifest.json"
    manifest_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    return manifest_path


def run_sft_warmup(
    paths: ProjectPaths,
    model_name: str = DEFAULT_SFT_MODEL,
    output_dir: Path | None = None,
    dataset_variant: str = "first_pass_all",
    dry_run: bool = False,
) -> Path:
    manifest_path = write_sft_warmup_manifest(
        paths=paths,
        model_name=model_name,
        output_dir=output_dir,
        dataset_variant=dataset_variant,
    )
    if dry_run:
        return manifest_path

    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover - depends on training environment
        raise RuntimeError(
            "Missing training dependencies. Install the `train` dependency group and a compatible PyTorch build before running SFT."
        ) from exc

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(manifest["output_dir"])
    dataset_path = manifest["dataset_path"]

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
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

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=int(manifest["num_train_epochs"]),
        per_device_train_batch_size=int(manifest["per_device_train_batch_size"]),
        learning_rate=float(manifest["learning_rate"]),
        max_length=int(manifest["max_seq_length"]),
        packing=bool(manifest["packing"]),
        gradient_checkpointing=bool(manifest["gradient_checkpointing"]),
        completion_only_loss=True,
        report_to=str(manifest["report_to"]),
        save_strategy="epoch",
        logging_steps=5,
        eos_token="<|im_end|>" if "Qwen/" in model_name else None,
    )

    trainer = SFTTrainer(
        model=model_name,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return manifest_path
