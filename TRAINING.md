# Training Runbook

## Status

The data pipeline, prompt dataset, SFT review pack, and dry-run training scaffolds are ready.

## Artifacts

- Manual gold review pack: `data/manual/sft_gold_review_pack_v1.csv`
- SFT reviewed set: `data/processed/sft_warmup_reviewed.jsonl`
- SFT reviewed-only prompt/completion set: `data/processed/sft_warmup_reviewed_only.jsonl`
- SFT first-pass full prompt/completion set: `data/processed/sft_warmup_first_pass_all.jsonl`
- SFT gold-only prompt/completion set: `data/processed/sft_warmup_gold_v1.jsonl`
- SFT warmup training set: `data/processed/sft_warmup_training.jsonl`
- SFT manifest: `artifacts/sft/sft_warmup/sft_warmup_manifest.json`
- GRPO train set: `data/processed/grpo_train.jsonl`
- GRPO validation set: `data/processed/grpo_validation.jsonl`
- GRPO test set: `data/processed/grpo_test.jsonl`
- GRPO manifest: `artifacts/grpo/grpo_run/grpo_manifest.json`

## Pinned Training Stack

These are the versions explicitly locked by the project plan for the first GPU training runs:

```txt
trl==0.29.0
peft==0.18.1
bitsandbytes==0.49.2
```

After the first successful GPU dry run, freeze the full runtime to a notebook-level lockfile.

## Local Dry Runs

Write the SFT warmup manifest:

```bash
PYTHONPATH=src python3 -m ipl_reasoner.cli run-sft-warmup --dry-run --dataset-variant gold_v1
```

Write the GRPO manifest and export the GRPO datasets:

```bash
PYTHONPATH=src python3 -m ipl_reasoner.cli run-grpo --dry-run
```

## Intended GPU Sequence

1. Install the pinned training stack plus compatible `torch`, `transformers`, `accelerate`, `datasets`, and `sentencepiece` in the GPU runtime.
2. Run the SFT warmup from the prompt/completion dataset, starting with `gold_v1`.
3. Verify format quality and basic coherence on sampled SFT outputs.
4. Start GRPO from the saved SFT checkpoint, not from the base model.
5. Use the validation split for frozen-model checks before any 2025 evaluation.

## Notes

- The next SFT rerun should use prompt/completion data so loss applies only to the assistant answer.
- The default rerun dataset is `gold_v1`, which uses the curated manual gold labels instead of the earlier synthetic warmup labels.
- The GRPO rewards are implemented in `src/ipl_reasoner/grpo_train.py`.
- `training_dataset.csv` already satisfies the plan's prompt-length and cooldown checks.
