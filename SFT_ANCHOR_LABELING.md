# SFT Anchor Labeling

Use [sft_anchor_review_pack_v1.csv](/Users/piyushgupta/Documents/ipl-grpo/data/manual/sft_anchor_review_pack_v1.csv) as the anchor worksheet.

## What This Sheet Is For

This sheet is the small, high-trust seed set for future SFT data.

The goal is not coverage of every IPL state. The goal is to create a compact set of examples with:

- correct directionality
- match-specific reasoning
- probability and rationale alignment
- diverse scenario coverage

## Fill Only These Columns

- `label_status`
- `gold_probability`
- `gold_analysis`
- `reviewer_notes`

Leave all other columns unchanged.

## Allowed Values

`label_status`
- `approved`
- `skip`
- `needs_discussion`

`gold_probability`
- decimal between `0.00` and `1.00`
- use two decimals when possible, for example `0.67`

`gold_analysis`
- 2 to 4 sentences
- plain text only
- no XML tags
- mention only the decisive factors

`reviewer_notes`
- optional
- use this for ambiguity, missing context, or why a row should be skipped

## What A Good Anchor Label Should Do

- Get the cricket direction right first.
- Mention only 2 to 4 decisive signals.
- Use venue, dew, player priors, or finishers only when they materially matter.
- Sound like an actual over-by-over analyst, not a template.
- Keep the probability consistent with the written explanation.

## What To Avoid

- Arithmetic contradictions.
- Generic filler.
- Repeating the same structure across unrelated states.
- Mentioning every available feature just because it exists in the prompt.

## Example

`label_status`
- `approved`

`gold_probability`
- `0.31`

`gold_analysis`
- `The chase is still alive because wickets remain, but the required rate is now clearly above the current scoring pace. A fresh batter arriving at this point would make the next two overs especially important. This is a pressure chase rather than a balanced one.`

## After You Fill It

Run:

```bash
PYTHONPATH=src python -m ipl_reasoner.cli build-sft-artifacts
```

If the sheet has approved rows, the pipeline will automatically create:

- [sft_warmup_anchor_v1.jsonl](/Users/piyushgupta/Documents/ipl-grpo/data/processed/sft_warmup_anchor_v1.jsonl)

That file is the model-ready anchor SFT dataset.
