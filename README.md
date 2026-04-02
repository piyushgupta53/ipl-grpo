# IPL Win Probability Reasoner

Project scaffold for the Week 1 data pipeline described in `PLAN.md`.

## Why this setup

- `uv` for Python version, dependency, and environment management
- `src/` layout for clean imports as the project grows
- A very small CLI so recurring data tasks have one stable entrypoint

## Quick start

```bash
uv sync
uv run ipl-reasoner init-workspace
uv run ipl-reasoner doctor
uv run ipl-reasoner validate-raw-data
uv run ipl-reasoner normalize-raw-data
uv run ipl-reasoner build-cleaned-data
uv run ipl-reasoner build-player-season-stats
uv run ipl-reasoner build-venue-artifacts
uv run ipl-reasoner build-snapshots
uv run ipl-reasoner build-training-dataset
uv run ipl-reasoner audit-training-dataset
uv run ipl-reasoner build-sft-artifacts
```

## Training workflow

For the clean long-term Colab workflow, use:

- [WORKFLOW.md](/Users/piyushgupta/Documents/ipl-grpo/WORKFLOW.md)
- [COLAB.md](/Users/piyushgupta/Documents/ipl-grpo/COLAB.md)
- [TRAINING.md](/Users/piyushgupta/Documents/ipl-grpo/TRAINING.md)

## Expected raw data layout

Place the historical IPL CSVs here once downloaded:

- `data/raw/matches.csv`
- `data/raw/deliveries.csv`

## First implementation target

The first milestone is the offline data pipeline:

1. Validate raw files and schema variants
2. Normalize teams, venues, and player names
3. Build season-scoped priors and snapshot features
4. Serialize the baseline model and generated artifacts
