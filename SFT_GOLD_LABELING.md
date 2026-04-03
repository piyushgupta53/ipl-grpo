# SFT Gold Labeling

Use [sft_gold_review_pack_v1.csv](/Users/piyushgupta/Documents/ipl-grpo/data/manual/sft_gold_review_pack_v1.csv) as the manual gold-label worksheet.

## What You Need To Fill

Fill only these 4 columns:

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
- mention only the most important signals
- do not mention every field from the prompt
- do not use hindsight from the real result
- do not include XML tags here

`reviewer_notes`
- optional
- use this if something feels ambiguous, noisy, or not worth training on

## Good Label Rules

- Keep the probability and reasoning aligned.
- Mention only 2 to 4 decisive factors.
- Use venue, dew, or player history only when they materially matter.
- Avoid canned phrases unless the state is truly obvious.
- If the equation is hard, say why in cricket terms.
- If the batting side is ahead, say what resource or condition is driving that edge.

## Bad Label Examples

- Arithmetic contradictions like saying the required rate is below the current rate when it is not.
- Generic filler like "one over can change the game" when it adds nothing.
- Repeating the same template across different situations.
- Mentioning venue or player priors when they are not actually important.

## How I Will Use Your Inputs

After you fill the sheet, I will convert each approved row into:

```xml
<analysis>
...
</analysis>
<answer>0.XX</answer>
```

Then I will build a clean gold-only SFT dataset from those approved rows.

## Suggested Workflow

1. Start with 20 rows.
2. Mark any unclear row as `needs_discussion`.
3. Once the first 20 look good, continue through the rest.

## One Example

`gold_probability`
- `0.34`

`gold_analysis`
- `The chase is still alive because the set batter remains in, but the equation is now pushing well above the current scoring rate. With only four wickets left, another wicket would make the death overs much harder to manage. This is a pressure chase now rather than a balanced one.`
