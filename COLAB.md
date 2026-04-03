# Colab Setup

## Goal

Use Google Colab GPU for the **first SFT warmup run** before attempting GRPO.

## Recommended Order

1. Upload or sync this project into Colab.
2. Install the pinned training stack.
3. Rebuild the SFT artifacts and manifests inside Colab so all paths point to the Colab filesystem.
4. Run SFT first.
5. Only after SFT completes and the output format looks good, prepare GRPO.

## Option A: Work From Google Drive

Mount Drive and `cd` into the project folder that contains this repo snapshot.

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/ipl-grpo
```

## Option B: Upload A Zip / Repo Copy

Unpack the project under `/content/ipl-grpo` and then:

```python
%cd /content/ipl-grpo
```

## Install Dependencies

First remove Colab's preinstalled `wandb` package if it is present. We are not using `wandb`
for the first run (`report_to="none"`), and a broken preinstalled `wandb` can cause TRL imports
to fail even before training starts.

```bash
!python -m pip uninstall -y wandb
```

Then install the project training stack:

```bash
!python -m pip install -r requirements-train-colab.txt
```

If Colab already has a suitable `torch`, keep it. The project plan specifically pins:

- `trl==0.29.0`
- `peft==0.18.1`
- `bitsandbytes==0.49.2`

## Prepare Colab-Local Manifests

This step is important because manifests written on the laptop contain laptop-specific absolute paths.

```bash
!PYTHONPATH=src python scripts/colab_prepare_and_run.py --stage sft --sft-dataset-variant gold_v1
```

That will:

- rebuild SFT artifacts if needed
- write a fresh Colab-local SFT manifest

## Start SFT Warmup

```bash
!PYTHONPATH=src python scripts/colab_prepare_and_run.py --stage sft --sft-dataset-variant gold_v1 --run
```

If you still see a `wandb` import error after uninstalling it, restart the Colab runtime once and
rerun the install cell plus the SFT command.

Expected output location:

- `artifacts/sft/sft_warmup/`

## Check The SFT Output

After SFT, sample a few generations before moving on:

- confirm `<analysis>...</analysis><answer>0.XX</answer>` format
- confirm summaries are short and situation-specific
- confirm probabilities are parseable decimals

You can use the evaluation helper:

```bash
!PYTHONPATH=src python scripts/eval_sft_checkpoint.py --checkpoint artifacts/sft/sft_warmup --dataset data/processed/grpo_validation.jsonl --num-samples 12
```

## Prepare GRPO After SFT

Dry-run only:

```bash
!PYTHONPATH=src python scripts/colab_prepare_and_run.py --stage grpo --grpo-model artifacts/sft/sft_warmup
```

Or prepare both in sequence:

```bash
!PYTHONPATH=src python scripts/colab_prepare_and_run.py --stage both --grpo-model artifacts/sft/sft_warmup
```

## Notes

- Start with SFT only. Do not jump straight into GRPO on the first Colab run.
- The improved rerun uses prompt/completion examples so the model is trained on assistant answers only.
- The default rerun dataset is `gold_v1`, built from the curated manual gold review pack at `data/manual/sft_gold_review_pack_v1.csv`.
- `training_dataset.csv` already passes the prompt length and cooldown checks needed by the plan.
