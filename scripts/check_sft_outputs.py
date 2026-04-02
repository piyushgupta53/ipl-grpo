"""Sample generations from the saved SFT checkpoint for quick sanity checks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_sft_outputs",
        description="Load the SFT checkpoint and print sample generations from saved prompts.",
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/sft/sft_warmup",
        help="Path to the saved SFT checkpoint or adapter directory.",
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/sft_warmup_reviewed.jsonl",
        help="JSONL file containing prompts to sample.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of prompts to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting prompts.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=180,
        help="Maximum number of generated tokens.",
    )
    return parser


def load_rows(dataset_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_model_and_tokenizer(checkpoint_path: str):
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            torch_dtype="auto",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            torch_dtype="auto",
        )
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rows = load_rows(dataset_path)
    if not rows:
        parser.error(f"No rows found in {dataset_path}")

    rng = random.Random(args.seed)
    sample_size = min(args.num_samples, len(rows))
    sampled = rng.sample(rows, k=sample_size)

    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    for idx, row in enumerate(sampled, start=1):
        prompt = str(row.get("prompt") or row["messages"][0]["content"])
        metadata = row.get("metadata", {})
        completion = generate_response(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        print("=" * 80)
        print(f"SAMPLE {idx}")
        print(json.dumps(metadata, ensure_ascii=True))
        print("-" * 80)
        print(completion)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
