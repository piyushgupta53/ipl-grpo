"""Quick post-SFT evaluation with basic format and sanity heuristics."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from ipl_reasoner.prompt_format import ensure_assistant_turn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_sft_checkpoint",
        description="Generate from the SFT checkpoint and flag obvious format/coherence failures.",
    )
    parser.add_argument("--checkpoint", default="artifacts/sft/sft_warmup")
    parser.add_argument("--dataset", default="data/processed/grpo_validation.jsonl")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    sampled = random.Random(args.seed).sample(rows, k=min(args.num_samples, len(rows)))

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    format_failures = 0
    sanity_failures = 0

    for idx, row in enumerate(sampled, start=1):
        prompt = ensure_assistant_turn(str(row["prompt"]))
        generated = generate(model, tokenizer, prompt, args.max_new_tokens)
        probability = extract_probability(generated)
        format_ok = bool(re.search(r"<analysis>.*?</analysis>\s*<answer>[\d.]+</answer>", generated, re.DOTALL))
        sanity_ok, reasons = sanity_check(row, probability)

        if not format_ok:
            format_failures += 1
        if not sanity_ok:
            sanity_failures += 1

        print("=" * 80)
        print(
            json.dumps(
                {
                    "sample": idx,
                    "match_id": row.get("match_id"),
                    "season": row.get("season"),
                    "snapshot_over": row.get("snapshot_over"),
                    "format_ok": format_ok,
                    "sanity_ok": sanity_ok,
                    "sanity_reasons": reasons,
                    "predicted_probability": probability,
                },
                ensure_ascii=True,
            )
        )
        print("-" * 80)
        print(generated)
        print()

    print("=" * 80)
    print(
        json.dumps(
            {
                "samples_checked": len(sampled),
                "format_failures": format_failures,
                "sanity_failures": sanity_failures,
            },
            ensure_ascii=True,
        )
    )
    return 0


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def extract_probability(text: str) -> float | None:
    match = re.search(r"<answer>([\d.]+)</answer>", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def sanity_check(row: dict[str, object], probability: float | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if probability is None:
        return False, ["missing_probability"]

    rrr = float(row.get("required_run_rate", 0.0) or 0.0)
    wickets = int(row.get("wickets_in_hand", 0) or 0)

    if rrr >= 18.0 and wickets <= 3 and probability > 0.25:
        reasons.append("too_high_for_extreme_chase")
    if rrr >= 12.0 and wickets <= 2 and probability > 0.20:
        reasons.append("too_high_for_late_collapse")
    if rrr <= 6.0 and wickets >= 7 and probability < 0.60:
        reasons.append("too_low_for_easy_chase")
    if rrr <= 8.0 and wickets >= 8 and probability < 0.50:
        reasons.append("too_low_for_strong_resource_position")

    return len(reasons) == 0, reasons


if __name__ == "__main__":
    raise SystemExit(main())
