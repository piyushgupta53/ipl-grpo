# Git + Colab Workflow

This is the clean workflow going forward:

- `code` lives in Git
- `training runs` live in Google Drive
- Colab clones the code fresh each time into `/content/ipl-grpo`
- every SFT / GRPO run writes to a separate Drive folder

Do **not** keep re-uploading the whole project folder to Drive after every code change.

## One-Time Setup

### 1. Keep the existing Drive folder as an archive

Do not delete the current Drive copy yet. Treat it as:

- archive of the first SFT run
- backup location for the earlier checkpoint zip

If you want, rename it later to something like:

- `ipl-grpo-archive-run1`

### 2. Create a Git repo for the code

Best option: create a private GitHub repo.

Then from your laptop inside this project:

```bash
git init
git add .
git commit -m "Initial IPL reasoner pipeline and training scaffold"
git branch -M main
git remote add origin <your-private-github-repo-url>
git push -u origin main
```

### 3. Keep training outputs out of Git

This repo already ignores:

- processed data
- reports
- SFT outputs
- GRPO outputs
- zip checkpoints

That means code sync stays light and fast.

## Day-To-Day Workflow

### On your laptop

When you change code:

```bash
git add .
git commit -m "Describe the change"
git push
```

### On Colab

#### 1. Start a fresh GPU runtime

Use a T4 GPU to start.

#### 2. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

#### 3. Clone the latest code into Colab local disk

```bash
!git clone <your-private-github-repo-url> /content/ipl-grpo
```

If the repo is already cloned in the current runtime:

```bash
%cd /content/ipl-grpo
!git pull
```

#### 4. Go into the repo

```python
%cd /content/ipl-grpo
```

#### 5. Install dependencies

```bash
!python -m pip uninstall -y wandb
!python -m pip install -r requirements-train-colab.txt
```

## Drive Output Layout

Create a dedicated Drive folder for training runs, for example:

- `/content/drive/MyDrive/ipl-grpo-runs/`

Then keep each run separate:

- `/content/drive/MyDrive/ipl-grpo-runs/sft_run_001`
- `/content/drive/MyDrive/ipl-grpo-runs/sft_run_002`
- `/content/drive/MyDrive/ipl-grpo-runs/grpo_run_001`

Never overwrite the same run directory unless you intentionally want to replace that run.

## SFT Run

### 1. Prepare artifacts in the cloned repo

```bash
!PYTHONPATH=src python scripts/colab_prepare_and_run.py --stage sft --sft-dataset-variant first_pass_all
```

### 2. Start SFT and write outputs to Drive

```bash
!PYTHONPATH=src python scripts/colab_prepare_and_run.py \
  --stage sft \
  --sft-dataset-variant first_pass_all \
  --sft-output-dir /content/drive/MyDrive/ipl-grpo-runs/sft_run_002 \
  --run
```

### 3. Evaluate the saved checkpoint

```bash
!PYTHONPATH=src python scripts/eval_sft_checkpoint.py \
  --checkpoint /content/drive/MyDrive/ipl-grpo-runs/sft_run_002 \
  --dataset data/processed/grpo_validation.jsonl \
  --num-samples 12
```

If the outputs still look repetitive or obviously wrong, improve SFT before GRPO.

## GRPO Run

Only do this after SFT outputs pass the sanity check.

### 1. Prepare GRPO manifest

```bash
!PYTHONPATH=src python scripts/colab_prepare_and_run.py \
  --stage grpo \
  --grpo-model /content/drive/MyDrive/ipl-grpo-runs/sft_run_002
```

### 2. Start GRPO and write outputs to a new Drive folder

```bash
!PYTHONPATH=src python scripts/colab_prepare_and_run.py \
  --stage grpo \
  --grpo-model /content/drive/MyDrive/ipl-grpo-runs/sft_run_002 \
  --grpo-output-dir /content/drive/MyDrive/ipl-grpo-runs/grpo_run_001 \
  --run
```

## Best Practices

- Use a fresh cloned repo in Colab local disk for code
- Use Drive only for long-lived outputs
- Keep one folder per training run
- Do not mix checkpoints into the code repo
- Do not manually re-upload the whole project after every change
- If you need a portable backup, zip only the final run folder

## What You Need To Do Next

1. Create a private GitHub repo for this project
2. Push the local code to GitHub
3. Use the clone-and-run workflow above for the next Colab session

Once that is done, the iteration loop becomes:

- laptop: edit -> commit -> push
- colab: clone/pull -> run -> save outputs to Drive
