# IPL Win Probability Reasoner — Complete Project Plan

> A small LLM (1.5B params) trained via GRPO on 12 years of IPL ball-by-ball data to produce
> calibrated win probability estimates with explicit cricket reasoning summaries,
> running live during IPL 2026 matches and published on a public website.

---

## Table of Contents

1. [What We're Building and Why](#1-what-were-building-and-why)
2. [Foundational Design Decisions](#2-foundational-design-decisions)
3. [Dataset Selection](#3-dataset-selection)
4. [Raw Data: Columns and Schema](#4-raw-data-columns-and-schema)
5. [Data Cleaning and Exclusions](#5-data-cleaning-and-exclusions)
6. [Computed Columns: Point-in-Time Stats](#6-computed-columns-point-in-time-stats)
7. [Snapshot Construction](#7-snapshot-construction)
8. [Train / Validation / Test Split](#8-train--validation--test-split)
9. [SFT Warmup Phase](#9-sft-warmup-phase)
10. [GRPO Training](#10-grpo-training)
11. [TRL Configuration Reference](#11-trl-configuration-reference)
12. [Metrics: What to Track and What They Mean](#12-metrics-what-to-track-and-what-they-mean)
13. [Evaluation Framework](#13-evaluation-framework)
14. [Inference Pipeline (Live Matches)](#14-inference-pipeline-live-matches)
15. [Website Architecture](#15-website-architecture)
16. [Failure Modes and Fixes](#16-failure-modes-and-fixes)
17. [Project Timeline](#17-project-timeline)
18. [Resources and References](#18-resources-and-references)

---

## 1. What We're Building and Why

### The Problem with Existing IPL Prediction Work

Every existing IPL win prediction project is logistic regression or random forest trained on cumulative match features. They output a number with no explanation. They treat this as a tabular ML problem. They are all the same project.

### What We're Doing Differently

We train a small LLM using GRPO (Group Relative Policy Optimization) to:

1. **Read** a match state described in natural language (current score, wickets, batsmen, bowler, venue, player stats)
2. **Reason** through the situation the way a knowledgeable cricket analyst would — considering player form, venue conditions, required rate, phase of the game
3. **Output** a calibrated probability with a short public-facing reasoning summary
4. **Be rewarded** based purely on whether the prediction was accurate (Brier score against the actual match outcome)

The key insight from DeepSeek-R1: GRPO allows verifiable-reward RL training where the reward signal is a deterministic function (did the team win?) rather than a learned reward model. Cricket is a perfect GRPO domain because:

- The ground truth is a hard binary: chasing team wins (1) or doesn't (0)
- There are rich contextual features that a model *can* reason about (player names, venue, form)
- A naive baseline (run-rate model) exists to compare against — our model must beat it
- Ball-by-ball data going back to 2008 gives ~12 years of supervised signal

The public reasoning summary is the differentiating artifact. A model that says *"Bumrah has already completed 3 overs and the asking rate is still under 9, so the batting side remains ahead despite recent pressure"* is doing something no existing IPL predictor does.

### The Public Website

During IPL 2026, a public website will show:
- All 2026 matches with live/completed predictions
- Per-match probability timeline at every completed over from over 1 through over 19 in the 2nd innings
- Full expandable public reasoning summaries for each prediction
- Season-long calibration stats vs. a naive baseline

This is the blog post, the demo, and the proof that the project works.

**Schedule note (implementation lock):** As of **April 1, 2026**, the official IPL site has announced
only the **first phase** of the TATA IPL 2026 schedule: **March 28, 2026 through April 12, 2026**.
The full season fixture list should be treated as provisional until the official later-phase
announcement lands. The inference system, database, and website must therefore ingest fixture
updates incrementally rather than assuming the entire season calendar is already known.

---

## 2. Foundational Design Decisions

These were explicitly decided and should not be revisited without strong reason.

### Decision 1: Second Innings Only (for training and inference)

**Decision:** Train exclusively on 2nd innings (chase) snapshots. Do not predict 1st innings.

**Reasoning:** During a 1st innings there is no fixed target. Win probability requires predicting two things: what score will be set AND whether the opposition can chase it. That is a two-step probabilistic chain requiring a much richer model.

During the 2nd innings:
- Target is known and fixed
- Required runs is calculable at every ball
- Balls remaining is known
- The problem collapses to a single clean question: does this batting team reach the target?

First innings prediction is a natural v2 extension once this project is working.

### Decision 2: End-of-Over Snapshots for Every Over 1-19

**Decision:** Generate one training example per match at the end of every completed over from over 1 through over 19. Do not predict ball-by-ball and do not predict after over 20 is complete.

**Reasoning:** Ball-by-ball would give 120 training examples per match but with extreme temporal correlation and tiny reward variance between adjacent balls. Over-level snapshots are still meaningful, but unlike the earlier 5-over design they fully match the desired live product. 19 snapshots × ~700 training matches = ~13,000 examples — still tractable for a 1.5B LoRA GRPO run, provided same-match samples are deliberately separated in training order.

Why start at over 1:
- The target is already known from the start of the chase
- The website goal is an every-over probability timeline
- A simple "every completed over until over 19" rule removes inference ambiguity

Why stop at over 19:
- After over 20 is complete the outcome is already known
- Over-19 is the last genuinely predictive boundary with one over still to play

### Decision 3: Exclude 2008-2012 from Training Examples

**Decision:** Use 2008-2012 matches only for computing historical player stats. Do not use them as training examples.

**Reasoning:** Players who appear in 2013+ IPL matches already have 4-5 years of IPL history to draw stats from. Before 2013, most players have too few deliveries for reliable point-in-time stats (cold-start problem). The noise of unreliable player features in early-season examples is not worth the extra ~200 training examples.

### Decision 4: SFT Before GRPO (Not Cold-Start GRPO)

**Decision:** Run a brief SFT warmup (200-300 examples, 1-2 hours) before GRPO training.

**Reasoning:** AlphaMaze research showed cold-start GRPO struggles when the model has zero domain intuition — it "overthinks," generates excessively long outputs, and wastes hundreds of GRPO steps just learning the output format. A cricket domain is even more alien to a base model than maze navigation. SFT gives the model: (a) the output format, (b) cricket vocabulary, (c) the structure of public reasoning summaries. GRPO then optimizes correctness, not format.

### Decision 5: Qwen2.5-1.5B-Instruct as the Base Model

**Decision:** Use `Qwen/Qwen2.5-1.5B-Instruct`.

**Reasoning:**
- Fits in free Colab T4 (15GB VRAM) with 4-bit quantization during training
- Instruct variant has instruction-following already built in
- Qwen family is empirically the most GRPO-responsive at small scales (most GRPO papers use Qwen)
- LoRA allows efficient fine-tuning: only ~1% of parameters updated
- 1.5B is large enough to reason about cricket concepts, small enough to iterate fast

For v2: `Qwen2.5-3B-Instruct` on Colab A100 if 1.5B results disappoint.

Do NOT use base (non-instruct) variant — requires much more aggressive SFT warmup to produce coherent output format.

---

## 3. Dataset Selection

### Primary Source: Kaggle IPL Ball-by-Ball Dataset

**Dataset:** [IPL Ball by Ball Dataset (2008-2025)](https://www.kaggle.com/datasets/anukaggle81/ipl-ball-by-ball-dataset-2008-2025)

Two CSV files:
- `deliveries.csv` — every ball bowled, ~300,000+ rows
- `matches.csv` — every match, ~1,000+ rows

This dataset is sourced from [Cricsheet](https://cricsheet.org/) which provides free, open-source ball-by-ball cricket data updated regularly. Cricsheet is the authoritative free source.

**Why this dataset over alternatives:**
- Complete ball-by-ball granularity (required to compute player stats and match snapshots)
- Covers all 17 IPL seasons 2008-2025
- Sourced from Cricsheet — clean, standardized, well-maintained
- Free and widely used in cricket analytics research

**Supplementary:** If the Kaggle dataset is missing 2025 data, Cricsheet directly (cricsheet.org/matches/ipl/) provides YAML files for every IPL match that can be parsed and converted to the same CSV format.

**Note on naming variants:** Some Kaggle versions use `batsman` (older), others use `striker` (newer Cricsheet naming). Verify which version you download and be consistent. The rest of this plan uses the newer `striker`/`batter` naming.

---

## 4. Raw Data: Columns and Schema

### `matches.csv` — All Columns

| Column | Type | Description | Used? |
|---|---|---|---|
| `id` | int | Unique match ID (ESPNCricinfo) | ✅ Primary key |
| `season` | str | IPL season e.g. "2021" | ✅ For splits |
| `city` | str | City name | ✅ Cross-ref for venue |
| `date` | date | Match date **YYYY-MM-DD** | ✅ Critical for point-in-time |
| `team1` | str | One of the two teams | ✅ |
| `team2` | str | The other team | ✅ |
| `toss_winner` | str | Team that won toss | ✅ Used to derive `chasing_team` |
| `toss_decision` | str | "bat" or "field" | ✅ Used to derive `chasing_team` |
| `result` | str | "normal", "tie", "no result" | ✅ Filter no-result matches |
| `dl_applied` | int | 0 or 1 — DLS applied | ✅ **Exclude if 1** |
| `method` | str | "D/L" or null (newer datasets) | ✅ **Exclude if not null** |
| `winner` | str | Winning team name | ✅ **Ground truth label source** |
| `win_by_runs` | int | >0 if team batting first won | ✅ Derives who chased |
| `win_by_wickets` | int | >0 if chasing team won | ✅ Derives who chased |
| `result_margin` | int | Margin of victory (some versions) | ✅ |
| `player_of_match` | str | POTM | ❌ Not used |
| `venue` | str | Full venue name e.g. "Wankhede Stadium, Mumbai" | ✅ Venue stats lookup |
| `neutral_venue` | int | 0 or 1 | ❌ Not used |
| `eliminator` | str | "Y"/"N" — playoff match | 🔶 Flag for analysis |
| `umpire1` | str | On-field umpire 1 | ❌ Not used |
| `umpire2` | str | On-field umpire 2 | ❌ Not used |

### `deliveries.csv` — All Columns

| Column | Type | Description | Used? |
|---|---|---|---|
| `match_id` | int | FK → matches.csv `id` | ✅ Join key |
| `inning` | int | 1 or 2 (3/4 = super over) | ✅ Filter to inning==2 |
| `batting_team` | str | Batting team name | ✅ |
| `bowling_team` | str | Bowling team name | ✅ |
| `over` | int | Over number **(check: 0-indexed or 1-indexed in your version)** | ✅ |
| `ball` | int | Ball number within over | ✅ |
| `striker` / `batsman` | str | Batsman on strike | ✅ |
| `non_striker` | str | Batsman at non-striker end | ✅ |
| `bowler` | str | Bowler | ✅ |
| `runs_off_bat` / `batsman_runs` | int | Runs off bat (0-6) | ✅ |
| `extras` / `extra_runs` | int | Total extra runs on delivery | ✅ |
| `wides` | int | Wide runs (NaN if none) | ✅ Exclude from ball count |
| `noballs` | int | No-ball runs (NaN if none) | ✅ Exclude from ball count |
| `byes` | int | Bye runs | ✅ Count toward total |
| `legbyes` | int | Leg-bye runs | ✅ Count toward total |
| `total_runs` | int | `runs_off_bat + extras` | ✅ Primary run counter |
| `is_wicket` | int | 0 or 1 | ✅ |
| `dismissal_kind` / `wicket_type` | str | "caught", "bowled", "lbw", "run out"... | ✅ |
| `player_dismissed` | str | Name of dismissed player | ✅ Track current batsmen |
| `fielder` | str | Fielder in dismissal | ❌ Not used |
| `extras_type` | str | "wides", "noballs", "byes", "legbyes" | ✅ Distinguish extra types |

**Critical nuance on ball counting:** A wide or no-ball is NOT a legal delivery. It does not count toward the 6 balls of an over. When computing `balls_bowled` and `balls_remaining`, count only legal deliveries: `ball is NOT wide AND ball is NOT no-ball`. Wides and no-balls run through the `wides` and `noballs` columns being non-null/non-zero. Getting this wrong invalidates all RRR calculations.

**Critical nuance on over numbering:** Some dataset versions are 0-indexed (overs 0-19) and some are 1-indexed (overs 1-20). Check yours and normalize to 1-indexed throughout.

---

## 5. Data Cleaning and Exclusions

Apply these filters before any feature computation. These are hard exclusions — never include these in training or test data.

### Exclusion 1: DLS / Rain-Affected Matches
```
EXCLUDE IF: dl_applied == 1 OR method IS NOT NULL OR method == 'D/L'
```
**Reason:** DLS changes the target mid-match. A snapshot computed at over 10 uses the original target, but the team is chasing a revised target. The model receives contradictory information. ~5-8% of matches.

**Live inference rule for v1:** if a live match becomes DLS-affected or reduced-overs, skip predictions for that match entirely. Do not try to support DLS live unless it is also represented in training and evaluation.

### Exclusion 2: Super Over Matches
```
EXCLUDE IF: any deliveries row with inning >= 3 for this match_id
```
**Reason:** Super overs happen after the main innings tied. Your snapshot at over 17 might show the chasing team close to tying, but the final outcome (super over loss) doesn't map cleanly to the prediction made in over 17. The model gets punished for reasonable predictions.

### Exclusion 3: No-Result Matches
```
EXCLUDE IF: result == 'no result' OR winner IS NULL
```
**Reason:** No ground truth label available.

### Exclusion 4: Corrupted or Unreconstructable 2nd Innings
```
EXCLUDE IF: inning==2 delivery log is missing
         OR (
              result == 'normal'
              AND second innings terminal state cannot be reconciled with one of:
                  (a) chasing team reached target
                  (b) 10 wickets fell
                  (c) scheduled overs were exhausted
            )
```
**Reason:** We want to exclude genuinely broken scorecards, not valid short chases. A team can legally finish a chase in fewer than 30 balls, and those matches are especially informative high-confidence examples. The correct rule is structural integrity, not a minimum-ball threshold.

### Exclusion 5: Early Seasons as Training Examples (Not Stats Source)
```
EXCLUDE FROM TRAINING IF: season IN ['2007/08', '2009', '2010', '2011', '2012']
```
**Note the naming:** Early IPL seasons are named "2007/08" in some datasets. Map these consistently. Use 2008-2012 data only for computing historical player stats (the stats source for the 2013 season's point-in-time table).

### Team Name Normalization

**Do this before any aggregation — it is a silent data bug if skipped.**

IPL franchises have changed names mid-history. If you compute "venue stats for Delhi Capitals at Arun Jaitley" but the dataset has all pre-2019 matches tagged as "Delhi Daredevils", you silently discard 11 years of home match history. Same for player stats — a player who played for Deccan Chargers early in their career gets stats split across two team names in any `groupby('team')` operation.

```python
TEAM_NAME_CANONICAL = {
    # Renamed franchises — normalize to current name
    "Deccan Chargers":             "Sunrisers Hyderabad",
    "Delhi Daredevils":            "Delhi Capitals",
    "Kings XI Punjab":             "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",

    # Defunct franchises — keep as-is (never appear in 2026 inference)
    "Rising Pune Supergiant":      "Rising Pune Supergiants",   # Typo variant in dataset
    "Pune Warriors":               "Pune Warriors",
    "Kochi Tuskers Kerala":        "Kochi Tuskers Kerala",
    "Gujarat Lions":               "Gujarat Lions",
}

# Apply to BOTH matches.csv and deliveries.csv before any processing
for col in ['team1', 'team2', 'toss_winner', 'winner']:
    matches[col] = matches[col].replace(TEAM_NAME_CANONICAL)
for col in ['batting_team', 'bowling_team']:
    deliveries[col] = deliveries[col].replace(TEAM_NAME_CANONICAL)
```

Run `print(matches['team1'].unique())` after normalization to confirm no stray variants remain.

### Venue Name Normalization
Venues appear with inconsistent full names. Create a venue_code mapping that covers **all venues
present in the 2013-2025 training window**, not just the venues likely to appear in live IPL 2026.
This explicitly includes historical neutral/home venues such as the UAE grounds used in 2014, 2020,
and 2021.

Implementation rule:
- Maintain an explicit alias table first
- Use fuzzy matching only as a warning-producing fallback for unseen string variants
- Fail dataset generation if any venue remains unmapped after normalization

Create a venue_code mapping:
```
"Wankhede Stadium, Mumbai" → "wankhede"
"MA Chidambaram Stadium, Chepauk, Chennai" → "chepauk"
"M. Chinnaswamy Stadium" → "chinnaswamy"
"Eden Gardens" → "eden_gardens"
... etc. for every venue string present in the historical dataset
```
This is needed because your point-in-time venue stats table uses venue_code as the key.

### Venue Metadata Artifact (Required)

The prompt builder needs more than numeric venue stats. Materialize a separate
`venue_metadata.csv` with **one row per `venue_code`** and load it during both offline prompt
assembly and live inference.

Minimum columns:
```text
venue_code,
pitch_type,
boundary_size,
dew_factor,
climate,
spin_assist,
pace_assist,
surface_descriptor,
notes
```

Implementation rule:
- `venue_season_stats.csv` holds numeric historical priors
- `venue_metadata.csv` holds qualitative venue traits used for prompt text
- Appendix C is the seed reference for this file, but the implementation must materialize it as a real checked-in artifact
- Fail the build if any `venue_code` used anywhere in offline training, replay evaluation, or live inference lacks a `venue_metadata.csv` row
- Treat `venue_metadata.csv` as a versioned artifact checked into the repo alongside the prompt template, not as an ad-hoc notebook output

---

## 6. Computed Columns: Point-in-Time Stats

This is the most important and most complex part of the data pipeline. Getting this right eliminates temporal leakage — the model only ever sees information that was genuinely available at the time of each match.

### The Core Principle

For v1, use **season-start point-in-time priors** rather than match-date point-in-time stats:

- For any snapshot from season **S**, the historical player and venue priors are computed using only matches completed **before season S began**
- This prevents future leakage across seasons and keeps the preprocessing tractable
- Within-season adaptation for live IPL 2026 inference is handled separately via rolling 2026 buffers (see Sections 6 and 14)

This means a May 2024 snapshot and an April 2024 snapshot share the same pre-2024 historical priors in the offline dataset. That is an explicit design choice for v1, not a bug.

### Step 1: Attach Dates to Every Delivery

```
merged_deliveries = deliveries.merge(
    matches[['id', 'date', 'season', 'venue']],
    left_on='match_id',
    right_on='id'
)
```

Every row in `deliveries` now has a `date` column. Sort by date. Offline historical prior builds then
filter to matches completed before the relevant season-start cutoff.

### Step 2: Precompute the Player Stats Table

Build one CSV: `player_season_stats.csv`
One row per `(player, as_of_season)`. `as_of_season` means "pre-season prior computed using all IPL data before this season started."

```
Rows:
  player=V Kohli, as_of_season=2013 → stats from 2008-2012 only
  player=V Kohli, as_of_season=2014 → stats from 2008-2013 only
  ...
  player=V Kohli, as_of_season=2025 → stats from 2008-2024 only
  player=V Kohli, as_of_season=2026 → stats from 2008-2025 only
```

**Implementation lock:** build `player_season_stats.csv` rows for every `as_of_season` from **2013 through 2026 inclusive**.
The `as_of_season=2026` row is mandatory for live IPL 2026 inference and must use **all completed IPL
data from 2008-2025 only**.

**For each (player, as_of_season), compute these columns:**

**Batsman Stats:**

| Column | Computation |
|---|---|
| `innings_count` | Count of innings (inning==2, is_striker) before this season |
| `chase_sr` | `(sum runs_off_bat / count legal balls as striker in inning==2) * 100`, EWMA-weighted |
| `death_sr` | Same but filtered to over >= 17, EWMA-weighted |
| `chase_boundary_pct` | `(count of 4s + 6s as striker in inning==2) / count legal balls` |
| `chase_dot_pct` | `count of zero-run legal balls as striker in inning==2 / count legal balls` |
| `death_boundary_pct` | Boundary % in overs 17-20 specifically |
| `recent_form_sr` | Strike rate in most recent completed season only (no EWMA, just last season) |
| `is_estimated` | `True if innings_count < 30` |

**Bowler Stats:**

| Column | Computation |
|---|---|
| `overs_bowled` | Count distinct (match_id, over) where bowler == this player, before this season |
| `death_economy` | `(sum runs_off_bat + extras conceded in over >= 17) / legal_balls_bowled_in_death * 6`, EWMA-weighted |
| `death_wickets_per_over` | `sum is_wicket in over >= 17 / overs_in_death`, EWMA-weighted |
| `overall_economy` | `(sum runs_off_bat + extras conceded) / legal_balls_bowled * 6`, EWMA-weighted |
| `death_dot_pct` | Dot ball % as bowler in overs 17-20 |
| `is_estimated` | `True if overs_bowled < 20` |

**EWMA Weighting Formula:**

For each delivery, compute its weight as:
```python
# Use April 1 of each season year as the canonical EWMA reference date.
# IPL starts roughly April 1 each year, so this is the "as of season start" reference.
# Using Jan 1 would make a March delivery (days_ago ≈ 90) appear much more recent
# than a May delivery from the prior season (days_ago ≈ 245), which is misleading —
# both are "pre-season" data for the same as_of_season row.
ewma_reference_date = pd.Timestamp(f"{season_year}-04-01")
days_ago = (ewma_reference_date - delivery_date).days
weight = exp(-0.003 * days_ago)
```
Then: `stat = sum(value * weight) / sum(weight)` instead of simple mean.

`λ = 0.003` gives roughly half-weight to deliveries from 9 months ago, quarter-weight to deliveries from 18 months ago. This handles players returning from injury or changing form without explicit logic.

**Cold Start Fallback:**
If `innings_count < 30` for a batsman, or `overs_bowled < 20` for a bowler, replace all their stats with the **league average** for that season computed from all players with sufficient data. Set `is_estimated = True`.

Compute the league averages themselves as a separate mini-table: `league_avg_by_season.csv`.

**Player Name Standardization:**
Cricsheet uses "V Kohli" style (initial + surname). Kaggle versions sometimes use "Virat Kohli". Pick one format and normalize across all data. The simplest approach: use the Cricsheet format as canonical, apply it consistently everywhere.

### Step 2B: Season-Scoped Live Player Buffer (Replay + Inference)

Offline training uses the pre-season priors above. Replay evaluation and live inference should additionally maintain a
season-scoped live player buffer so that active-player form does not go stale mid-season.

Implementation rule:

- Historical lookup for season `S` starts from `player_season_stats.csv` with `as_of_season=S`
- Maintain one season-aware live buffer keyed by `(season, player)` rather than hard-coding 2026 paths
- Use the same read/write interface for 2024 replay, 2025 frozen test replay, and live IPL 2026 inference
- Source of truth must be transactional storage (`player_live_season_stats` in Supabase or equivalent DB table), not a process-local mutable CSV
- Optional CSV exports such as `artifacts/player_live_2026_export.csv` are for debugging only and must never be the authoritative read path
- At inference time, blend the pre-season prior with the in-season aggregate for the current player

Minimal v1 blending formula:

```python
def blend_player_stat(historical_val, live_val, live_events, full_trust_at=60):
    alpha = min(1.0, live_events / full_trust_at)
    return (alpha * live_val) + ((1 - alpha) * historical_val)
```

Use `live_events` = legal balls faced for batting stats, and legal balls bowled for bowling stats.
This is intentionally simple. Do not try to reproduce the offline EWMA logic online in v1.

Implementation note:
- expose this through one helper such as `get_player_stat_live(player, season, stat_name)`
- the helper must read from the same season-scoped table in replay and production
- the only difference between replay and live is the season value (`2024`, `2025`, or `2026`) and the event source driving updates

### Step 3: Precompute the Venue Stats Table

Build one CSV: `venue_season_stats.csv`  
One row per `(venue_code, as_of_season)`. `as_of_season` means "pre-season venue prior computed using all IPL data before this season started."

**Implementation lock:** build `venue_season_stats.csv` rows for every `as_of_season` from **2013 through 2026 inclusive**.
The `as_of_season=2026` row is mandatory for live IPL 2026 inference and must use **all completed IPL
data from 2008-2025 only**.

| Column | Computation |
|---|---|
| `matches_count` | Number of matches at this venue before this season |
| `avg_first_innings_score` | Mean of inning==1 totals at this venue |
| `avg_second_innings_score` | Mean of inning==2 totals at this venue |
| `chase_success_rate` | % of matches where chasing team won at this venue |
| `avg_death_rpo` | Mean RPO in overs 17-20 in inning==2 at this venue |
| `avg_par_score_at_over_10` | Mean inning==2 score at end of over 10 at this venue |
| `confidence_alpha` | `min(1.0, matches_count / 20)` — blend weight toward own data |

**Do not use a binary `is_estimated` flag.** Instead use graduated confidence blending (see below).

---

### Venue Confidence Blending (Graduated, not Binary)

IPL 2026 uses venues with very different amounts of historical data. Three sparse venues
that appear frequently in 2026 fixtures have only 4-12 historical matches in training data:

| Venue | Approx Training Matches | Status |
|---|---|---|
| Wankhede, Chinnaswamy, Eden Gardens, Chepauk, Rajiv Gandhi, Sawai Mansingh, Arun Jaitley | 55-125 | ✅ Rich |
| Narendra Modi Ahmedabad, Ekana Lucknow | ~25 | 🟡 Moderate |
| HPCA Dharamsala | ~12 | 🟠 Sparse |
| Mullanpur (New Chandigarh) | ~8 | 🔴 Very Sparse |
| Barsapara / ACA Guwahati | ~6 | 🔴 Very Sparse |
| Nava Raipur (Shaheed Veer Narayan Singh) | ~4 | 🔴 Effectively New |

**Note on retired venues:** Some venues in the training data (Pune MCA, Visakhapatnam ACA-VDCA, DY Patil, Brabourne) do not appear in IPL 2026. They exist in `venue_season_stats.csv` and contribute to player history stats, but are never queried during live inference. No special handling required — they are simply dead code that never runs.

**The formula:**

```python
# α = how much to trust this venue's own historical data
# Reaches full trust (1.0) only at 20+ matches
alpha = min(1.0, matches_count / 20.0)

# Blended stat = weighted average of own data and the "similar venue" prior
blended_stat = (alpha * venue_own_stat) + ((1 - alpha) * similar_venue_stat)
```

- **0 matches:** 100% from similar venue (pure prior)
- **8 matches (Mullanpur):** 40% own, 60% prior
- **6 matches (Guwahati):** 30% own, 70% prior
- **4 matches (Raipur):** 20% own, 80% prior
- **20+ matches:** 100% own data

This is Bayesian updating: the "similar venue" is the prior, and real match data at that ground updates it toward reality as the season accumulates.

**Venue Similarity Pairings (hardcoded):**

```python
VENUE_SIMILAR_TO = {
    "mullanpur":           "mohali",                   # Same region (Punjab), similar pitch profile
    "aca_guwahati":        "eden_gardens",             # Similar climate zone, batting-friendly
    "raipur":              "narendra_modi_ahmedabad",  # Flat central India, batting surface
    "hpca_dharamshala":    "chinnaswamy",              # Elevated ground, pace-assisted early
    "dy_patil":            "wankhede",                 # Mumbai metro, similar conditions
    "brabourne":           "wankhede",                 # Mumbai metro, similar conditions
}
# All other venues: use their own stats directly (alpha = 1.0 for matches_count >= 20)
```

**Implementation:**

```python
def get_venue_stat(venue_code, as_of_season, stat_col, venue_season_stats):
    row = venue_season_stats[
        (venue_season_stats.venue_code == venue_code) &
        (venue_season_stats.as_of_season == as_of_season)
    ].iloc[0]

    alpha = row['confidence_alpha']          # min(1.0, matches_count / 20)
    own_val = row[stat_col]

    if alpha >= 1.0:
        return own_val, False                # is_estimated = False

    # Fall back to similar venue
    similar_code = VENUE_SIMILAR_TO.get(venue_code)
    if similar_code is None:
        return own_val, True                 # No pairing defined, use as-is

    similar_row = venue_season_stats[
        (venue_season_stats.venue_code == similar_code) &
        (venue_season_stats.as_of_season == as_of_season)
    ].iloc[0]
    similar_val = similar_row[stat_col]

    blended = (alpha * own_val) + ((1 - alpha) * similar_val)
    return blended, True                     # is_estimated = True (prompt will flag this)
```

Store `confidence_alpha` as a precomputed column in `venue_season_stats.csv` for fast lookup.

---

### Live Season Updating (IPL 2026 Inference)

After each completed IPL 2026 match, feed its data back into the venue stats table. This
means sparse venues accumulate real 2026 data during the season itself — reducing dependence
on the prior as the tournament progresses.

See Section 14 for the `update_venue_stats_live()` implementation called by the inference
pipeline after each match completes.

---

## 7. Snapshot Construction

A snapshot is a single training example: a natural language description of a match state at the end of a specific over, with the ground truth label of whether the chasing team ultimately won.

### Which Overs to Snapshot

Take snapshots at the end of every completed over: **1 through 19 inclusive**

Rationale:
- Matches the live product exactly: one prediction per completed over
- Gives full innings shape rather than only five checkpoints
- Still keeps prediction cadence sparse enough to avoid ball-by-ball noise

Only create a snapshot if the match hadn't ended before that over. Examples:
- Chase completed in over 12.3 → create snapshots for overs 1-12 only
- All-out in over 14 → create snapshots for overs 1-13 only
- Do not create an over-20 snapshot because the outcome is already known

### Per-Snapshot Computed Features

For each (match_id, snapshot_over), compute from deliveries filtered to `match_id == X AND inning == 2 AND over <= snapshot_over`:

| Feature | Computation |
|---|---|
| `runs_scored` | `sum(total_runs)` — all runs including extras |
| `wickets_fallen` | Count of wickets that reduce wickets in hand for the batting side |
| `wickets_in_hand` | `10 - wickets_fallen` |
| `balls_bowled` | Count of legal deliveries (wides and no-balls are NOT legal) |
| `balls_remaining` | `120 - balls_bowled` |
| `required_runs` | `target - runs_scored` |
| `current_run_rate` | `(runs_scored / balls_bowled) * 6` if balls_bowled > 0 else 0 |
| `required_run_rate` | `(required_runs / balls_remaining) * 6` if balls_remaining > 0 else 999 |
| `last_over_runs` | Runs in the most recently completed over |
| `last_3_overs_runs` | Sum of runs in the min(3, completed_overs) most recently completed overs |
| `last_3_overs_wickets` | Wickets in the min(3, completed_overs) most recently completed overs |
| `dot_ball_pct_last_5` | Dot ball % over the min(5, completed_overs) most recently completed overs |
| `boundary_pct_last_5` | (4s + 6s) / legal balls over the min(5, completed_overs) most recently completed overs |
| `batter_a` | One of the two batting-side players currently on the field at the over break |
| `batter_b` | The other batting-side player currently on the field, or `NEW_BATTER_PENDING` if a wicket fell on the over-ending ball and the replacement batter has not yet faced |
| `last_over_bowler` | Bowler who completed the most recent over |
| `last_over_bowler_overs_used` | How many complete overs this last-over bowler has already bowled in this innings |
| `last_wicket_over` | The over in which the most recent wicket fell (0 if no wickets yet) |
| `partnership_balls` | Balls since last wicket fell (proxy for current partnership length) |
| `partnership_runs` | Runs scored since last wicket fell — distinguishes a set 50-ball/70-run partnership from a 50-ball/25-run survival grind |

**Important state-definition rule:** this is an **end-of-over snapshot**, not a "start of next ball" snapshot. Therefore:
- Do not label the two batters as striker / non-striker in the prompt
- Do not present `last_over_bowler` as if he is the next bowler
- If a wicket falls on the final legal ball of the over, do not look ahead into the future innings data to recover the incoming batter for training; use `NEW_BATTER_PENDING` offline and live

**Wicket accounting rule (implementation lock):**
- `wickets_fallen` increments only for dismissals that reduce wickets in hand for the batting side
- Include: bowled, caught, lbw, stumped, hit wicket, run out, obstructing the field, timed out
- Exclude: retired hurt / retired out rows that do not consume one of the 10 wickets in the score-state representation
- The same rule must be used in offline preprocessing, live snapshot reconstruction, and replay evaluation

### Deriving `target`

```python
# Get all deliveries for this match, inning==1
inn1 = deliveries[(deliveries.match_id == match_id) & (deliveries.inning == 1)]
target = inn1['total_runs'].sum() + 1  # +1 because chaser needs one more
```

### Attaching Player Stats

For each snapshot, look up:
```python
# Offline training snapshots use pre-season priors only:
player_row = player_season_stats[
    (player_season_stats.player == player_name) &
    (player_season_stats.as_of_season == match_season)
]
venue_row = venue_season_stats[
    (venue_season_stats.venue_code == match_venue_code) &
    (venue_season_stats.as_of_season == match_season)
]
```

This is O(1) with proper indexing — no repeated filtering of the full deliveries dataset.

For live IPL 2026 inference, use:
- `player_season_stats.csv` row for `as_of_season=2026`
- plus `player_live_season_stats` where `season=2026` blended on top
- plus `venue_live_season_stats` where `season=2026` blended on top of `venue_season_stats.csv`

### Deriving `is_day_match` / `match_type`

This feature is useful but lower-confidence than score state and player context, so define it
explicitly and keep the fallback simple.

Offline historical rule:
- If the source data has a reliable start-time field, use it
- Otherwise maintain a small `match_start_time_lookup.csv` for seasons where the data is available
  from fixtures or scorecards
- If a historical match start time is missing, default to `is_day_match=False` (treat as evening)

Live 2026 rule:
- Use the live API match start time if available
- `is_day_match=True` only for standard afternoon starts
- Everything else is treated as evening / dew-likely

Prompting rule:
- Keep the `Match type:` line in the prompt
- Treat it as a contextual hint, not a dominant feature
- Do not block dataset generation or live inference on missing historical start times

### Assembling the Natural Language Prompt

Each snapshot becomes a structured natural language prompt. Template:

```
SYSTEM:
You are an expert cricket analyst specializing in IPL T20 matches. 
You will be given a match situation mid-chase and must estimate the 
probability (0.0 to 1.0) that the batting team wins the match.

Write a short public reasoning summary. Focus on the most important
signals in the situation at hand, and weigh them proportionally.
Do not force a checklist. Different situations are dominated by different evidence,
and the summary should focus only on what materially matters in that specific state.
Do not produce hidden chain-of-thought. The summary will be shown on a public website.

Format your response EXACTLY as:
<analysis>
[2-5 sentence public reasoning summary here]
</analysis>
<answer>0.XX</answer>

The answer must be a decimal between 0.0 and 1.0.

USER:
IPL Match — {chasing_team} chasing {target} set by {bowling_team}
Venue: {venue_name}
Match type: {match_type}  ← "Evening (7:30 PM IST start) — dew factor likely" or "Afternoon (3:30 PM IST start)"
Toss: {toss_winner} chose to {toss_decision} first

Current Situation (End of Over {snapshot_over}):
Score: {runs_scored}/{wickets_fallen} ({balls_bowled} balls faced)
Required: {required_runs} off {balls_remaining} balls (RRR: {required_run_rate:.2f})
Current run rate: {current_run_rate:.2f}

Recent Momentum:
Last over: {last_over_runs} runs
Last 3 overs: {last_3_overs_runs} runs, {last_3_overs_wickets} wickets
Dot ball %: {dot_ball_pct_last_5:.0%} | Boundary %: {boundary_pct_last_5:.0%}

Batters on Field:
{batter_a} (career IPL chase SR: {b1_chase_sr:.0f}, death SR: {b1_death_sr:.0f})
{batter_b_display}
Partnership: {partnership_balls} balls, {partnership_runs} runs since last wicket

Last Over Bowler: {last_over_bowler} ({last_over_bowler_overs_used}/4 overs used,
career death economy: {bowler_death_economy:.2f}, death wickets/over: {bowler_death_wpo:.2f})
Note: the next over must be bowled by someone else.

Venue Context ({venue_name}):
{venue_context_block}

What is the probability that {chasing_team} wins this match?
```

Build `venue_context_block` from **both** `venue_season_stats.csv` and `venue_metadata.csv`.
The stats table supplies numeric priors; the metadata table supplies qualitative surface/dew context.

**Venue Context Block — Two variants based on `confidence_alpha`:**

For rich venues (`confidence_alpha >= 0.75`, 15+ historical matches):
```
Venue Context (Wankhede Stadium, Mumbai):
Typical surface: batting-friendly, medium boundaries
Historical chase success rate: 58% (131 matches)
Average score batting second: 172
Avg death over RPO: 11.4
Dew factor: High in evening matches — historically favors chasing team
```

For sparse venues (`confidence_alpha < 0.75`, fewer than 15 historical matches):
```
Venue Context (Barsapara/ACA Stadium, Guwahati):
⚠️  Limited IPL history ({matches_count} matches). Stats below have elevated uncertainty.
Typical surface prior: batting-friendly, medium boundaries
Historical chase success rate: ~{venue_chase_rate:.0%} (estimated)
Average score batting second: ~{venue_avg_2nd:.0f} (estimated)
Avg death over RPO: ~{venue_death_rpo:.2f} (estimated)
Dew factor: Medium in evening matches
```

The `⚠️` flag and the word "estimated" appear consistently during both SFT and GRPO
training — the model learns to weight sparse venue context less heavily in its reasoning summary,
and will produce outputs that say things like *"Given limited venue data here, I'm weighting
the run rate and wickets in hand more heavily than venue norms."*

Note: if `is_estimated == True` for any **player**, append "(career data limited)" after their stats.

### Final Dataset Structure

Each row in `training_dataset.csv`:

| Column | Type | Description |
|---|---|---|
| `match_id` | int | For debugging and deduplication |
| `season` | str | For train/test split |
| `snapshot_over` | int | 1 through 19 |
| `chasing_team` | str | Team chasing |
| `venue_code` | str | Venue identifier |
| `prompt` | str | Full formatted prompt (as above) |
| `did_chasing_team_win` | int | **0 or 1 — ground truth label** |
| `run_rate_baseline_prob` | float | Canonical logistic baseline probability using `required_run_rate`, `balls_remaining`, and `wickets_in_hand` (pre-computed) |
| `is_any_player_estimated` | bool | True if any player stat was estimated |
| `required_run_rate` | float | For analysis |
| `wickets_in_hand` | int | For stratified analysis |
| `partnership_runs` | int | Runs in current partnership at snapshot |
| `is_day_match` | bool | True if afternoon match (dew factor low) |
| `toss_decision` | str | "bat" or "field" — toss winner's choice |
| `batter_a` | str | First on-field batter at over break |
| `batter_b` | str | Second on-field batter or `NEW_BATTER_PENDING` |
| `last_over_bowler` | str | Bowler who completed the most recent over |

The `did_chasing_team_win` and `run_rate_baseline_prob` columns flow through to the reward functions via TRL's `**kwargs` mechanism — they are not consumed by the model, only by the reward functions during training.

---

## 8. Train / Validation / Test Split

### Split Rationale

| Split | Seasons | Matches (approx) | Snapshots (approx) | Purpose |
|---|---|---|---|---|
| Stats source only | 2007/08–2012 | ~400 | — | Build player/venue history for 2013 season |
| **Training** | 2013–2023 | ~700 | ~13,300 | GRPO training |
| **Validation** | 2024 | ~74 | ~1,400 | Hyperparameter tuning, early stopping, replay-mode validation |
| **Test** | 2025 | ~74 | ~1,400 | Final evaluation, never touched during training |
| **Live inference** | 2026 | 74 (ongoing) | — | Public website predictions |

**Why temporal (not random) split:** If you split randomly, a 2016 snapshot could be in the validation set while another 2016 snapshot is in training. The model could effectively memorize match outcomes rather than generalizing. Temporal splits prevent this entirely.

**Why 2024 for validation, not a random 10%:** Validation should reflect how the model will perform on unseen future data. Using the most recent full historical season (2024) for validation mirrors the actual deployment scenario.

**Implementation lock:** 2024 is the **only** season used for iterative tuning. Run both static
validation and replay-mode validation on 2024. Keep 2025 completely untouched until the final
frozen-model evaluation. Do not tune prompt wording, live-buffer weights, or reward coefficients
against 2025 results.

### Correlated Snapshot Shuffling (GRPO-Specific)

We generate up to 19 snapshots per match. If a team collapses, many later snapshots share `did_chasing_team_win = 0`. In a GRPO batch, if multiple samples in the same group of G=4 completions come from the same match, the reward signal is artificially correlated — same outcome → similar Brier score → low reward variance → weak gradient.

Fix: build the training order with a **match cooldown scheduler**, not a naive round-robin over over-buckets.

```python
# Goal: keep the same match_id at least 8 rows apart whenever feasible
import random
from collections import defaultdict, deque

def build_training_order(df, cooldown=8, seed=42):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in df.sample(frac=1.0, random_state=seed).to_dict("records"):
        groups[row["match_id"]].append(row)

    recent = deque(maxlen=cooldown)
    ordered = []

    while groups:
        eligible = [mid for mid in groups if mid not in recent]
        if not eligible:
            eligible = list(groups.keys())  # fallback only when unavoidable

        match_id = rng.choice(eligible)
        ordered.append(groups[match_id].pop())
        recent.append(match_id)

        if not groups[match_id]:
            del groups[match_id]

    return pd.DataFrame(ordered)

train_dataset = build_training_order(df, cooldown=8, seed=42)
```

This does not mathematically guarantee zero collisions in every possible window, but it is the
correct v1 implementation strategy: it aggressively separates same-match samples and then relies
on a post-build validation check. If the validation check fails, reshuffle with a different seed.

### Computing the Baseline Probability

Before training, compute the **canonical training baseline** `run_rate_baseline_prob` for every snapshot
using a simple logistic regression trained on the training set only (2013-2023):

```
Features: required_run_rate, balls_remaining, wickets_in_hand
Target: did_chasing_team_win
```

This is the single baseline used inside the reward function, stored in the dataset, and compared
against during validation. Keep the column name `run_rate_baseline_prob` for convenience even though
it includes wickets and balls remaining.

Optional reporting-only baseline:
- Train a second, simpler logistic regression on `required_run_rate` alone if you want a weaker public baseline for charts
- Do **not** use that simpler model in the reward function or dataset

This gives you a strong, interpretable baseline. Every GRPO prediction must beat this to be considered useful.

### Baseline Artifact (Required)

The canonical logistic baseline is not just an offline notebook calculation. It is a production
artifact that must be serialized, versioned, and loaded by the same scoring code in:
- offline dataset generation
- replay evaluation
- live inference
- post-match finalization / season metrics

Implementation lock:
- Train exactly one canonical logistic model on 2013-2023 training snapshots
- Persist it as a versioned artifact, for example:
  - `artifacts/baseline/baseline_v1.joblib`
  - `artifacts/baseline/baseline_v1.metadata.json`
- Metadata must include at minimum:
  - `baseline_version`
  - `trained_on_seasons`
  - `feature_order = ["required_run_rate", "balls_remaining", "wickets_in_hand"]`
  - fitted coefficients / intercept if you are not relying on `joblib`
  - training timestamp and code commit hash
- Never re-fit the live baseline mid-season
- Record `baseline_version` on every stored prediction row and in every evaluation report

Use one scorer everywhere:

```python
def score_baseline(snapshot_features, baseline_model) -> float:
    x = [[
        snapshot_features["required_run_rate"],
        snapshot_features["balls_remaining"],
        snapshot_features["wickets_in_hand"],
    ]]
    return float(baseline_model.predict_proba(x)[0, 1])
```

The `run_rate_baseline_prob` column in `training_dataset.csv` must be generated by this scorer, and
the live `baseline_probability` written to Supabase must also come from this scorer. No duplicate
implementations.

Expected validation Brier scores:
- Always-50% baseline: ~0.25
- Run-rate logistic regression: ~0.16-0.18
- Our GRPO model target: < 0.14 overall, significantly better on "interesting" snapshots

---

## 9. SFT Warmup Phase

### Purpose

Give the model three things before GRPO starts:
1. The `<analysis>...</analysis><answer>X.XX</answer>` output format
2. Cricket vocabulary and domain context
3. The notion that 0.XX probabilities are the expected output unit

The model does NOT need to be highly accurate at this stage — it just needs to produce sensible,
well-structured public reasoning summaries in the right format.

### SFT Dataset Construction

**200-300 examples, hand-crafted or Claude-generated.**

**Prefer 2019-2023 seasons for SFT examples.** IPL T20 scoring has trended significantly upward — average death over RPO rose from ~8.5 in 2013 to ~10.5 in 2023. An SFT example from 2013 where "8 RPO in death is threatening" teaches the wrong baseline for 2026 inference. Recent seasons better represent the run-rate environment the model will face live.

For each example: a match state prompt (same format as training) + a well-structured response.

Generate responses by:
1. Pick 50 "easy" snapshots (very high or very low RRR — probability nearly obvious)
2. Pick 50 "interesting" snapshots (moderate RRR, interesting player context)
3. Use Claude/GPT-4 to write short public reasoning summaries for each, given the match state
4. Manually review 30-40 of them to ensure cricket accuracy
5. The rest can be lightly edited AI-generated output

### SFT Reasoning Philosophy

The SFT phase should teach **holistic, selective public reasoning**, not a rigid checklist.

Principles:
- The summary should focus on the factors that matter most **for that specific snapshot**
- It should be acceptable for some summaries to emphasize only 2-3 dominant signals
- Less relevant context may be ignored entirely when it does not materially change the estimate
- The reasoning should feel like an analyst prioritizing evidence, not a model exhaustively enumerating fields

What to avoid in SFT:
- Do not force every example to mention venue, player stats, bowler, momentum, and required rate
- Do not enforce a fixed order like "RRR first, then venue, then players"
- Do not reward exhaustiveness for its own sake
- Do not write summaries that read like template filling

What good SFT examples should demonstrate:
- Selective emphasis: identify the dominant factor(s) in the situation
- Proportionality: spend more words on what matters most, fewer on marginal details
- Coherence: the probability should follow naturally from the reasoning
- Variety: summaries across the dataset should differ in emphasis, ordering, and style while staying concise

Typical strong summaries will mention the most important 2-4 factors, not every available field.
They should still end with a specific probability and a clear rationale for why the estimate is above or below 0.5.

### SFT Training Setup

Using `trl.SFTTrainer`:
```python
from trl import SFTTrainer, SFTConfig

config = SFTConfig(
    output_dir="./sft_warmup",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,  # Higher than GRPO — SFT can be aggressive
    max_seq_length=1024,
    packing=False,
)
```

Estimated time: 1-2 hours on Colab T4. Save this checkpoint — it's the starting point for GRPO.

---

## 10. GRPO Training

### The Core Setup

```python
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

# LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# GRPOConfig
training_args = GRPOConfig(
    output_dir="./ipl_grpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch = 16 prompts
    num_generations=4,               # G=4 (reduce to fit T4 VRAM)
    max_prompt_length=512,
    max_completion_length=256,
    learning_rate=2e-6,
    num_train_epochs=3,
    temperature=0.7,
    beta=0.04,                       # KL penalty coefficient
    epsilon=0.2,                     # Clipping parameter
    max_grad_norm=0.1,               # Aggressive gradient clipping
    logging_steps=5,
    save_steps=100,
    report_to="wandb",               # or "tensorboard"
    log_completions=True,            # Log sample summaries to wandb
    num_completions_to_print=2,      # Print 2 completions per log step
    bf16=False,                      # T4 does not support native bf16 training
    fp16=True,                       # Use fp16 on Colab T4; switch to bf16 only on A100/H100
    gradient_checkpointing=True,
)

trainer = GRPOTrainer(
    model="./sft_warmup",           # Start from SFT checkpoint
    args=training_args,
    reward_funcs=[format_reward, rationale_reward, accuracy_reward],
    reward_weights=[0.15, 0.10, 0.75],
    train_dataset=train_dataset,
    peft_config=lora_config,
)
```

### Training Environment Lock

Do not mix TRL doc versions or float the runtime stack during implementation.

Pin a single tested training stack before writing trainer code:

```txt
trl==0.29.0
peft==0.18.1
bitsandbytes==0.49.2
```

Then install TRL's compatible `transformers` / `accelerate` versions and freeze the full notebook
environment to `requirements-lock.txt` after the first successful dry run. All code and docs in this
project should target that single pinned TRL release only.

### Reward Function 1: Format Reward (weight 0.15)

Checks if the completion is parseable and valid. Returns 0.0, 0.5, or 1.0.

```python
import re

def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        has_analysis = bool(re.search(r'<analysis>.*?</analysis>', completion, re.DOTALL))
        answer_match = re.search(r'<answer>([\d.]+)</answer>', completion)
        
        if has_analysis and answer_match:
            try:
                prob = float(answer_match.group(1))
                if 0.0 <= prob <= 1.0:
                    rewards.append(1.0)   # Full format reward
                else:
                    rewards.append(0.5)   # Has format but invalid probability
            except ValueError:
                rewards.append(0.0)
        elif has_analysis or answer_match:
            rewards.append(0.5)           # Partial format
        else:
            rewards.append(0.0)           # No format at all
    return rewards
```

### Reward Function 2: Rationale Quality Heuristic (weight 0.10)

The public reasoning summary is part of the product, so give it a small explicit reward.
Do **not** try to reward hidden chain-of-thought. Reward only public-facing qualities:
- summary exists
- concise enough for the website
- non-generic enough to read like situation-specific analysis rather than filler

Crucially, **do not hardcode a required factor list** such as run rate, wickets, venue,
momentum, player quality, or bowler context. Any subset of evidence is valid if it is
the right subset for that match state.

```python
def rationale_reward(completions, **kwargs):
    rewards = []
    banned_generic_phrases = [
        "it is a close game",
        "anything can happen",
        "both teams have a chance",
        "it depends on many factors",
    ]

    for completion in completions:
        m = re.search(r'<analysis>(.*?)</analysis>', completion, re.DOTALL)
        if not m:
            rewards.append(0.0)
            continue

        text = m.group(1).strip()
        word_count = len(text.split())
        lower = text.lower()
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        has_number = bool(re.search(r'\d', text))
        generic_penalty = any(p in lower for p in banned_generic_phrases)

        # Reward concise, situation-specific public summaries without prescribing
        # which cricket factors must be mentioned.
        if 35 <= word_count <= 120 and sentence_count >= 2 and has_number and not generic_penalty:
            rewards.append(1.0)
        elif 20 <= word_count <= 140 and sentence_count >= 1 and not generic_penalty:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards
```

This is intentionally lightweight. The main job of GRPO is still probability calibration;
SFT does most of the stylistic shaping for the public rationale.

### Reward Function 3: Accuracy / Brier Reward (weight 0.75)

Core learning signal. Based on Brier score with a confidence bonus.

```python
import math

def accuracy_reward(completions, did_chasing_team_win, run_rate_baseline_prob, **kwargs):
    rewards = []
    
    for completion, actual_outcome, baseline_prob in zip(
        completions, did_chasing_team_win, run_rate_baseline_prob
    ):
        # Try to extract probability from completion
        answer_match = re.search(r'<answer>([\d.]+)</answer>', completion)
        
        if not answer_match:
            rewards.append(0.0)  # Can't score what we can't parse
            continue
        
        try:
            predicted = float(answer_match.group(1))
            predicted = max(0.01, min(0.99, predicted))  # Clip to avoid log(0)
        except ValueError:
            rewards.append(0.0)
            continue
        
        actual = float(actual_outcome)
        
        # Brier score component: lower is better, so negate and shift
        # Brier score = (predicted - actual)^2, range [0, 1]
        # Reward = 1 - brier_score, range [0, 1]
        brier = (predicted - actual) ** 2
        brier_reward = 1.0 - brier
        
        # Confidence bonus: reward well-calibrated confident predictions
        # Only applies when direction is correct
        is_correct_direction = (predicted > 0.5 and actual == 1.0) or \
                               (predicted < 0.5 and actual == 0.0)
        confidence = abs(predicted - 0.5) * 2  # Scales [0, 1]
        confidence_bonus = 0.1 * confidence if is_correct_direction else 0.0
        
        # Beat-the-baseline bonus: reward confident-and-right over vague-and-safe
        # Activates only when model diverges from baseline by >0.15 AND is correct
        # Prevents Mode B reward hacking (collapsing to ~0.5 on everything)
        baseline_brier = (float(baseline_prob) - actual) ** 2
        beat_baseline = brier < baseline_brier
        diverged_from_baseline = abs(predicted - float(baseline_prob)) > 0.15
        baseline_bonus = 0.05 if (beat_baseline and diverged_from_baseline) else 0.0

        total = brier_reward + confidence_bonus + baseline_bonus
        rewards.append(total)
    
    return rewards
```

**Why Brier score and not log loss:** Brier score penalizes overconfident wrong predictions more harshly than modest wrong predictions, which is exactly the behavior you want to incentivize calibration. `(0.9 - 0)^2 = 0.81` (big punishment for confident wrong prediction) vs `(0.6 - 0)^2 = 0.36` (smaller punishment for cautious wrong prediction).

### Two Reward Hacking Failure Modes to Monitor

**Mode A — Extreme confidence bias.** The confidence bonus rewards correct-direction confidence. The model may learn to output `0.05` / `0.95` more than warranted, maximizing the bonus at the cost of calibration.

Detection: calibration chart. If your model's predicted-0.85 situations win only 60% of the time, Mode A is active.

Fix: reduce `confidence_bonus` weight coefficient from 0.1 to 0.05. Or cap the bonus at predictions within [0.15, 0.85] — i.e., never give confidence bonus for predictions outside that range.

**Mode B — Vague center collapse.** The model learns that outputting `0.45-0.55` always yields Brier ≤ 0.25 (a "safe" reward floor). It stops differentiating situations at all.

Detection: `reward_std < 0.05` past step 200; short completion lengths; predictions clustered around 0.5 in wandb histograms.

Fix: the `baseline_bonus` in the reward function above directly counters this. A model that always says 0.5 never diverges from baseline by >0.15, so never earns the bonus. Over time GRPO learns that safe-but-vague is suboptimal vs. differentiated-and-right.

### How `**kwargs` Works in TRL

TRL's `GRPOTrainer` automatically passes all non-`prompt` columns of your dataset as keyword arguments to the reward function. This means your dataset needs:

```python
# Dataset must have these columns:
dataset = Dataset.from_dict({
    "prompt": [...],                    # Required by TRL
    "did_chasing_team_win": [...],      # Passed as **kwargs to reward functions
    "run_rate_baseline_prob": [...],    # Passed as **kwargs
    "match_id": [...],                  # Passed as **kwargs (useful for logging)
    "snapshot_over": [...],             # Passed as **kwargs
})
```

The `did_chasing_team_win` column is never seen by the model — only by the reward function. This is the correct way to pass ground truth in TRL.

---

## 11. TRL Configuration Reference

### Key GRPOConfig Parameters Explained

| Parameter | Value | Why |
|---|---|---|
| `num_generations` | 4 | Group size G. GRPO needs variance within groups. ≥4 minimum. Reduce to 4 for T4 VRAM. Use 8 on A100. |
| `temperature` | 0.7 | Controls diversity within generation group. Too low: all completions identical (no gradient). Too high: gibberish. 0.7-0.8 is the right range. |
| `beta` | 0.04 | KL coefficient. Higher = model stays closer to reference (less learning). Lower = more freedom but risk of reward hacking. Start 0.04, increase to 0.1 if hacking observed. |
| `epsilon` | 0.2 | Clipping range for policy ratio. Standard value. |
| `max_grad_norm` | 0.1 | Aggressive gradient clipping. Cricket probability is continuous — stable updates are important. |
| `max_completion_length` | 256 | Public reasoning summaries should be short. 256 gives enough headroom without rewarding verbosity. |
| `learning_rate` | 2e-6 | Conservative. GRPO is sensitive. Too high and training collapses. |
| `gradient_accumulation_steps` | 16 | With batch_size=1 and num_generations=4, effective batch = 16 prompts. |
| `log_completions` | True | **Essential.** Lets you inspect public reasoning summaries in wandb during training. |
| `reward_weights` | [0.15, 0.10, 0.75] | Format and rationale quality are guardrails. Accuracy is still the goal. |

### VRAM Budget (T4 = 15GB)

```
Qwen2.5-1.5B-Instruct in 4-bit:        ~1.5 GB
Reference model copy (with LoRA):       disabled — LoRA handles this
Activations (batch=1, gen_len=256):     ~2.5 GB
4 generations per prompt:               ~4 GB (inference + scoring)
Gradient states (LoRA only):            ~2 GB
Overhead:                               ~2 GB
Total:                                  ~12 GB ✅ Fits in T4
```

If you run out of VRAM, first reduce `num_generations` from 4 to 2, then reduce `max_completion_length` to 192.

---

## 12. Metrics: What to Track and What They Mean

### Auto-Logged by TRL (shown in wandb/tensorboard)

| Metric | Healthy Range | What to Do If Wrong |
|---|---|---|
| `reward` | Trending upward, slowly | Core signal. If flat after 300 steps: check reward variance |
| `reward_std` | 0.1 – 0.3 | Too low (<0.05): increase temperature or confidence bonus. Too high (>0.5): reduce temperature |
| `reward/format_reward` | Reaches 0.9+ by step 100 | If not: system prompt isn't clear enough |
| `reward/accuracy_reward` | Trends up after step 100 | Main learning. Expect slow progress |
| `kl` | 0.01 – 0.1, stable | Rising: increase beta. Near zero: decrease beta |
| `entropy` | Gradually declining, 3.0→2.0 | Fast collapse (<1.0): reward hacking. Investigate summaries |
| `clip_ratio/high_mean` | < 0.3 | Consistently high: reduce learning rate |
| `clip_ratio/low_mean` | < 0.3 | Same |
| `loss` | Decreasing then stabilizing | Spikes or NaN: reduce learning rate, increase max_grad_norm slightly |
| `completion_length` | 60-140 tokens, stable | Growing: add length penalty. Shrinking: fine if reward improves |
| `grad_norm` | < 1.0 most steps | Spikes >10: your max_grad_norm=0.1 clips it, but investigate why |

### Custom Metrics (Log Manually from Reward Function)

Add to your reward function:
```python
# These can be passed to a custom logger or computed post-hoc
format_success_rate      # % completions with valid format per batch
predicted_prob_mean      # Should hover ~0.45-0.55 (IPL chase rate)
predicted_prob_std       # Should be ~0.2 — model is taking positions
brier_score_raw          # The actual metric (before reward transformation)
correct_direction_rate   # % where model got high/low probability correct
```

### Custom Evaluation Metrics (Run Every 100 Steps on Validation Set)

| Metric | Description | Target |
|---|---|---|
| **Brier Score** | Mean `(predicted - actual)^2` over validation set | < 0.16 (beat logistic regression) |
| **Calibration ECE** | Expected Calibration Error — how well probabilities reflect true rates | < 0.08 |
| **vs Baseline Improvement** | `(baseline_brier - model_brier) / baseline_brier` | > 10% improvement |
| **Brier by Over Bucket** | Separate Brier scores for overs 1-5, 6-10, 11-15, 16-19 | Should improve most in death overs (16-19) |
| **Calibration by Bucket** | At predicted probability 0.7, does the chasing team win ~70% of the time? | Visual check — should be near-diagonal |

### The "Aha Moment" Metric

Periodically (every 200 steps), run inference on your qualitative eval suite (20-30 hand-picked scenarios). Look for summaries that:

- identify the dominant factor in the situation rather than listing everything mechanically
- selectively incorporate player, bowler, venue, or momentum context when those factors genuinely matter
- ignore marginal details when the game state is already overwhelmingly decisive
- produce a probability that feels consistent with the reasoning, not bolted on at the end

Concrete positive signs include summaries such as:
- "The asking rate is manageable, but two wickets in the last three overs make this chase less stable than the raw numbers suggest"
- "Dhoni being one of the two batters still on the field changes this materially because the finishing burden is concentrated in one proven batter"
- "Venue history matters less here because the match state itself is already dominant"
- "Recent wickets matter more than venue context in this exact situation because the chase has become fragile"

These qualitative indicators are as important as the quantitative metrics for assessing whether genuine learning is happening.

---

## 13. Evaluation Framework

### Three Levels of Evaluation

**Level 1: Quantitative (automated)**

Run after each epoch on the 2024 validation set:
- Brier score
- ECE (Expected Calibration Error)
- Percentage of predictions that beat the canonical logistic baseline

**Deployment-consistent replay evaluation (required):**

Pure offline validation uses pre-season priors only, which is correct for training but does not fully
match live deployment because live inference blends rolling in-season player and venue buffers.
Therefore add a second automated evaluation track on the **same validation season**:

- **2024 replay mode**: simulate the entire 2024 season in chronological order
- Start from `player_season_stats.csv` and `venue_season_stats.csv` with `as_of_season=2024`
- Before match 1, season-scoped live buffer tables contain zero rows for `season=2024`
- After each completed 2024 match, update `player_live_season_stats(season=2024, ...)` and `venue_live_season_stats(season=2024, ...)`
- When evaluating match N, only use information available before match N started
- The replay evaluator must call the same helpers used by production, e.g. `get_player_stat_live(player, season)` and `get_venue_stat_live(venue_code, season, stat_col)`
- The canonical baseline probability used in replay must be scored by the serialized baseline artifact, not recomputed from a new fit
- Replay runs must use an isolated evaluation store or clear/rebuild the `season=2024` / `season=2025` live-buffer partitions before each run so stale state cannot leak across evaluations

Report both:
- **static validation** on 2024 using pre-season priors only
- **replay validation** on 2024 using rolling in-season updates

Implementation gate: do not treat the model as deployment-ready unless replay-mode calibration
and Brier score are acceptable in addition to static validation.

**Level 2: Qualitative Eval Suite (manual, ~30 min)**

20-30 hand-crafted scenarios, run every 200 GRPO steps. Scenarios:

```
Group 1 — Easy cases (model should be obvious):
  - Team needs 20 off 30 balls, 8 wickets in hand → expected: >0.90
  - Team needs 70 off 18 balls, 5 wickets → expected: <0.10
  
Group 2 — Pressure situations (tests player knowledge):
  - 25 off 12 balls, Dhoni at crease, 5 wickets in hand → expected: ~0.65
  - Same situation, unknown batsman → expected: ~0.40
  - 32 off 18 balls, Bumrah bowled the 18th over and has already completed 3 overs → expected: lower probability than the same state against a weaker recent-over bowler
  
Group 3 — Venue effects:
  - 55 off 30 balls at Wankhede (batting pitch) → expected: higher probability
  - 55 off 30 balls at Chepauk (spin assists bowling) → expected: lower probability
  
Group 4 — Momentum:
  - 40 off 24 balls, 2 wickets in last 3 overs → expected: moderate, concern about collapse
  - 40 off 24 balls, no wickets last 5 overs → expected: higher probability
```

**Level 3: Final Test Set (2025 season, run once)**

After model, prompt, baseline artifact, release manifest, and live-buffer logic are frozen, run final evaluation on all 2025 match
snapshots. Report:
- **static 2025 test** using pre-season priors only
- **replay 2025 test** using chronological rolling in-season updates
- Overall Brier score vs. three baselines:
  - always-0.5
  - the canonical logistic baseline used in training (`required_run_rate + balls_remaining + wickets_in_hand`)
  - optional simpler RRR-only reporting baseline
- Calibration chart (predicted probability buckets vs. actual win rates)
- Brier score by over number and by over bucket (model should be most accurate at overs 16-19)
- Top 5 most confident correct predictions (show the public reasoning summaries)
- Top 5 biggest mispredictions (diagnose why)

No model, prompt, baseline, or live-buffer logic changes are allowed after inspecting 2025 results.
If changes are needed, the 2025 test is considered spent and a new holdout period must be designated.

---

## 14. Inference Pipeline (Live Matches)

### Architecture Overview

```
[Cricket Data Source] ──(poll every 60-90 sec)──→ [Python Script]
                                                      │
                                             detect over change
                                                      │
                                        backfill any missed overs
                                                      │
                                          assemble match state text
                                                      │
                                       look up player stats table (CSV)
                                                      │
                                        run model inference (local/Modal)
                                                      │
                                       parse <analysis> and <answer> tags
                                                      │
                                         save to Supabase database
                                                      │
                                     website reads + displays in real-time
```

### Cricket Data Sources

Two sources only, in priority order. No paid APIs needed.

---

**Primary: `cricdata` Python package (free, no API key)**

An unofficial Python client that wraps ESPNCricinfo's internal endpoints. Designed
specifically for ML/analytics pipelines — returns plain Python dicts, no HTML parsing.

```bash
pip install cricdata
```

```python
from cricdata import CricinfoClient

ci = CricinfoClient()

# Discover IPL 2026 matches (series ID is stable for the whole season)
fixtures = ci.series_fixtures("ipl-2026-1510719")
matches  = fixtures["content"]["matches"]

# Build slugs for a specific match (needed for all subsequent calls)
m            = matches[0]
series_slug  = f"{m['series']['slug']}-{m['series']['objectId']}"
match_slug   = f"{m['slug']}-{m['objectId']}"

# Live scorecard at any point
scorecard = ci.match_scorecard(series_slug, match_slug)
innings2  = scorecard["content"]["innings"][1]   # 2nd innings

# Ball-by-ball (full delivery log, use to compute last-3-overs stats)
balls = ci.match_ball_by_ball(series_slug, match_slug)

# Extract what we need
runs     = innings2["runs"]
wickets  = innings2["wickets"]
overs    = innings2["overs"]          # float, e.g. 13.4
batsmen  = innings2["batsmen"]        # list of {name, runs, balls, sr}
bowlers  = innings2["bowlers"]        # list of {name, overs, runs, wickets, economy}
target   = innings2["target"]         # live target (handles DLS revision automatically)
```

**Why this is the right primary source:**
- Zero cost, no API key, no signup
- Returns structured dicts — slot directly into the prompt assembler
- Covers ball-by-ball, scorecard, fixtures from the same client
- IPL series ID (`1510719`) is already in the codebase from earlier sections
- Built for ML pipelines — same audience as this project

**Risk:** Unofficial. ESPNCricinfo can change internal endpoints without notice. Mitigated
by the fallback below, and by the fact that our over-boundary polling (up to 19 calls/match)
is still low-stress on their infrastructure.

---

**Fallback: CricketData.org free tier ($0/month, API key required)**

Use when `cricdata` throws a connection error or returns stale data. Free tier gives
100 hits/day. Our worst case (double-header day) is ~38 calls. Comfortably within limit
for the entire IPL season.

Sign up at https://cricketdata.org to get a free API key — takes 2 minutes.

```python
import requests

CRICDATA_KEY = "your_free_key_here"   # Store in .env, never hardcode
BASE         = "https://api.cricapi.com/v1"

def get_scorecard_fallback(match_id: str) -> dict:
    r = requests.get(
        f"{BASE}/match_scorecard",
        params={"apikey": CRICDATA_KEY, "id": match_id},
        timeout=10
    )
    r.raise_for_status()
    return r.json()["data"]

def get_current_matches_fallback() -> list:
    r = requests.get(
        f"{BASE}/currentMatches",
        params={"apikey": CRICDATA_KEY},
        timeout=10
    )
    r.raise_for_status()
    return r.json()["data"]
```

**Key note:** Free tier data is intentionally delayed by a few minutes for legal compliance.
This is still acceptable for us because the system backfills any missed overs from the ball log.
Even if the API lags and we only see the match at "end of over 13" while over 14 is being bowled,
the over-13 state is already complete and can still be reconstructed correctly.

---

**Switching logic in the polling script:**

```python
def fetch_match_state(series_slug: str, match_slug: str,
                      fallback_match_id: str) -> dict:
    """Try cricdata first. Fall back to CricketData.org on any error."""
    try:
        ci = CricinfoClient()
        scorecard = ci.match_scorecard(series_slug, match_slug)
        return parse_cricdata_scorecard(scorecard)
    except Exception as e:
        print(f"[warn] cricdata failed: {e} — switching to CricketData.org fallback")
        raw = get_scorecard_fallback(fallback_match_id)
        return parse_cricketdata_scorecard(raw)
```

### Canonical Match Identity (Implementation Lock)

The system must use **one internal match key everywhere**. Do not let provider-specific IDs leak
into primary keys.

Rule:
```python
match_id = f"espn_{espn_object_id}"
```

This `match_id` is the canonical key stored in the database, event store, predictions table,
logs, and replay utilities. Provider-specific identifiers are stored only as attributes used to
fetch data.

Maintain a `match_provider_map.csv` with:
```text
match_id,
espn_object_id,
espn_match_slug,
espn_series_slug,
cricketdata_match_id
```

Build it from the primary fixtures feed before each match week. Fallback requests must resolve
through this map rather than inventing a second primary key.

### Release Manifest (Implementation Lock)

Production predictions must always be attributable to one immutable release bundle.

Define:
```text
release_version = <model_version>__<prompt_version>__<baseline_version>
```

Each release bundle contains:
- LoRA / merged model weights
- prompt template version
- canonical logistic baseline artifact
- parsing rules / output schema version

Implementation rule:
- For v1, prefer one production `release_version` for the entire IPL 2026 season
- Freeze one `release_version` on each `matches` row once the chase begins
- Do not switch prompt/model/baseline halfway through a live match
- Replays, backfills, and experiments must write under their own `release_version`
- If an emergency mid-season production release change is unavoidable, split public season metrics by `release_version` rather than merging them
- Website production views must respect the `assigned_release_version` recorded on each match

### Polling Cadence and Over Backfill

Do not rely on a single "latest completed over" check every 4-5 minutes. T20 overs can finish
quickly, especially in collapses or boundary-heavy phases, and the website promise is one prediction
per completed over.

Use this v1 rule:
- Poll every **60-90 seconds**
- Track `last_saved_over` per match in memory or in the database
- On each poll, compute `latest_completed_over`
- If `latest_completed_over > last_saved_over`, generate predictions for **every missing over**
  from `last_saved_over + 1` through `latest_completed_over`, inclusive

```python
def backfill_missing_overs(match_id, latest_completed_over, last_saved_over):
    for over in range(last_saved_over + 1, latest_completed_over + 1):
        state = build_snapshot_from_live_ball_log(match_id, snapshot_over=over)
        if state is not None:
            run_inference_and_upsert(state)
```

This guarantees that if polling jumps from over 12 to over 14, the system still generates and stores
the missing over-13 prediction before writing over 14.

### What to Poll For

At each poll interval, detect:
1. Is the match in the 2nd innings?
2. What is the **latest completed over**?
3. What is the **scheduled chase length** in overs, and is it still a standard 20-over innings?
4. Is that completed over in the range **1 through 19**?
5. What is the most recent stored prediction over for this match?
6. **Has the target changed from the 1st innings total + 1?** (DLS revision detection)

Trigger inference only if:
- the match is in the 2nd innings
- the innings length is still a standard 20-over chase
- the latest completed over is between 1 and 19 inclusive
- the chase has not already ended
- there exists at least one missing over between `last_saved_over + 1` and `latest_completed_over`
- the match is not DLS / reduced-overs

**DLS / Reduced-Overs Detection:**
```python
computed_target = first_innings_total + 1
live_target = api_response['target_runs']
scheduled_overs = api_response.get('scheduled_overs', 20)

if scheduled_overs != 20 or live_target != computed_target:
    # Reduced-overs or revised target detected
    # v1 rule: skip this match entirely rather than introducing train/deploy mismatch
    is_dl_match = True
    release_version = get_match_assigned_release_version(match_id) or ACTIVE_RELEASE_VERSION
    supabase.table('matches').update({
            'target': live_target,
            'is_dl_match': True,
            'status': 'live'
        }) \
            .eq('id', match_id).execute()
    # Invalidate any predictions already written before the weather interruption
    supabase.table('predictions').delete() \
        .eq('match_id', match_id) \
        .eq('release_version', release_version) \
        .eq('run_scope', 'prod') \
        .execute()
    return None
```

If the API does not expose a reliable revised target or scheduled-over count, abort inference for
that match and flag it manually. A wrong target or wrong innings length produces incorrect RRR in
every subsequent prompt — better to skip than to silently mislead.

### Persisted Live Event Store (Required for Robustness)

The ball-by-ball log is the single source of truth for live feature reconstruction, so it must be
persisted durably rather than held only in process memory.

Implementation rule:
- Normalize every polled live ball event into a canonical schema
- Upsert it into a persistent store (`live_ball_events` in Supabase, or an equivalent restart-safe parquet/CSV log)
- Rebuild match state from that persisted event store on restart
- Match finalization, replay debugging, and live-buffer updates must read from this persisted event store, not from transient in-memory objects

Minimum canonical event fields:
```text
match_id,
innings,
over_number,
event_index_in_over,
legal_ball_number,
is_legal_ball,
batting_team,
bowling_team,
striker,
non_striker,
bowler,
runs_off_bat,
extras,
wides,
noballs,
byes,
legbyes,
total_runs,
is_wicket,
dismissal_kind,
player_dismissed,
provider_event_id,
event_ts
```

This makes the system restart-safe, auditable, and able to recompute any displayed over snapshot exactly.

Implementation lock:
- `event_index_in_over` is the sequential event number within the over, including wides and no-balls
- `legal_ball_number` is the legal-delivery count within the over, ranging from 0 to 6
- `event_ts` is metadata only and must never be part of the uniqueness key
- If the provider exposes a stable event ID, store it in `provider_event_id`, but still normalize into the canonical schema above

### Single Source of Truth for Live Feature Reconstruction

The prompt needs more than the top-level scorecard fields. To keep offline snapshots and live
inference in sync, implement **one normalized ball-event builder** and use it everywhere:

- Offline: `build_snapshot_from_deliveries(match_id, snapshot_over)`
- Live: `build_snapshot_from_live_ball_log(match_id, snapshot_over)`

Both builders must derive the same feature set from ball events rather than trusting partial
API summaries. In particular, compute these directly from the delivery log:
- `runs_scored`
- `wickets_fallen`
- `balls_bowled`
- `last_over_runs`
- `last_3_overs_runs`
- `last_3_overs_wickets`
- `dot_ball_pct_last_5`
- `boundary_pct_last_5`
- `partnership_balls`
- `partnership_runs`
- `batter_a`, `batter_b`, including `NEW_BATTER_PENDING`
- `last_over_bowler`
- `last_over_bowler_overs_used`

Rule: the live scorecard is useful for sanity checks and name lookups, but the persisted normalized
delivery log is the authoritative source for reconstructing end-of-over state.

**Impact Player Detection (IPL 2023+):**

The Impact Player rule (introduced 2023) lets teams substitute one player mid-match. This creates two subtle inference problems:

- A bowler showing `bowler_overs_used = 0` at over 18 could be a fresh impact substitute brought in for the death (massive bowling advantage) — or just a regular opener who saved their overs.
- A batter substituted in at over 6 may be a specialist pinch-hitter playing very differently from their career average.

If the live API exposes impact player status, add to the prompt:
```
Last Over Bowler: {bowler_name} ({overs_used}/4 overs)
{" ← Impact substitute (fresh, brought in specifically for death)" if is_impact_player else ""}
```

If the API doesn't expose this, accept the limitation silently — don't try to infer it from `bowler_overs_used` alone.

### Player Name Normalization Map (Build in Week 4)

Cricsheet canonical format ("V Kohli") differs from ESPNCricinfo live API display names ("Virat Kohli"). For ~250 active IPL 2026 players, a missing normalization means the lookup falls back to `is_estimated=True` — worst case for death-over predictions involving key players.

Build `player_name_map.csv` before first live prediction:

```
cricsheet_name,espncricinfo_name
V Kohli,Virat Kohli
MS Dhoni,MS Dhoni
HH Pandya,Hardik Pandya
A Russell,Andre Russell
AB de Villiers,AB de Villiers
...
```

Populate by: (a) list all player names seen in 2024-2025 Cricsheet data, (b) add players from officially announced 2026 squads / fixture scorecards / replacement announcements as they appear, (c) cross-reference ESPNcricinfo player pages for the highest-profile active players manually, (d) use fuzzy matching (`difflib.get_close_matches`) for the rest with a 0.85 threshold, (e) manually review any match that returns >2 `is_estimated=True` lookups.

Data needed from the API at each over boundary:
```
batting_team, bowling_team, target (live — use this, not computed)
current_score (runs/wickets)
current_over (completed)
scheduled_overs
ball_by_ball_log (authoritative source for all derived features below)
match_start_time (for is_day_match)
toss_winner, toss_decision
is_impact_player (if available)
```

Derive the following from `ball_by_ball_log`, not from commentary prose:
- `batter_a_name`, `batter_b_name_or_pending`
- `last_bowler_name`
- `last_bowler_overs_used`
- `last_3_overs_runs`
- `last_3_overs_wickets`
- `dot_ball_pct_last_5`
- `boundary_pct_last_5`
- `partnership_balls`
- `partnership_runs`

`last_bowler_overs_used` should be derived from the scorecard bowler figures for the bowler who completed the most recent over when that structured scorecard is available; otherwise compute it from the ball log.

### Player Stats Lookup

Load `player_season_stats.csv` once at process start as the historical prior table.

Player names from the live API must match your training data naming convention (the Cricsheet
"V Kohli" format). You'll need a name normalization mapping for IPL 2026 players to Cricsheet canonical names.

At inference time:
- historical prior = `player_season_stats.csv` row for `as_of_season=2026`
- live adjustment = blended value from `player_live_season_stats` where `season=2026`
- fallback = league-average 2026 prior if the player is missing from both

**Ownership / consistency rule:**
- the match finalization job is the only writer to `player_live_season_stats` and `venue_live_season_stats`
- the poller is read-only with respect to these tables
- poller must re-read season live stats at deterministic boundaries: process start, before first over of each live match, and after every completed match before monitoring the next one
- do not keep a long-lived mutable in-memory copy as the authoritative source across matches
- if a local cache is used for speed, treat it as disposable and refreshable from the database

### Venue Stats Lookup (Live, with Blending)

Venue stats are loaded from `venue_season_stats.csv`, but also maintain a **season-scoped live venue buffer** that updates as the season progresses. Use `get_venue_stat()` (defined in Section 6) for every lookup — it automatically applies the graduated blending formula.

```python
# Authoritative store: venue_live_season_stats table keyed by (season, venue_code)
def get_venue_stat_live(venue_code, season, stat_col):
    """
    Merge precomputed historical stats with season-specific matches played so far.
    The live buffer adds real in-season data on top of the blended historical estimate.
    """
    historical_val, is_estimated = get_venue_stat(
        venue_code, as_of_season=season, stat_col=stat_col, ...
    )
    live_data = load_venue_live_row(season=season, venue_code=venue_code)
    live_count = live_data.get('matches_count', 0)

    if live_count == 0:
        return historical_val, is_estimated

    live_val = live_data.get(stat_col, historical_val)
    # Weight live 2026 data at 2x — it's the most current surface conditions
    live_alpha = min(1.0, live_count / 5)
    merged = (live_alpha * live_val) + ((1 - live_alpha) * historical_val)
    return merged, (live_count < 5)
```

### Live Venue Stats Update (Post-Match)

After each completed season match, call this function to feed real data back into the
season-scoped live buffer. Sparse venues accumulate actual in-season data as the season progresses,
reducing dependence on the similarity-based prior by the 3rd or 4th match at that ground.

```python
def update_venue_stats_live(season, match_id, venue_code, deliveries_df, matches_df):
    """
    Called after a match completes. Upserts into `venue_live_season_stats`
    keyed by (season, venue_code).
    """
    match_deliveries = deliveries_df[deliveries_df.match_id == match_id]
    match_row = matches_df[matches_df.id == match_id].iloc[0]

    inn1 = match_deliveries[match_deliveries.inning == 1]
    inn2 = match_deliveries[match_deliveries.inning == 2]
    death = inn2[inn2.over >= 17]

    first_innings_total  = inn1['total_runs'].sum()
    second_innings_total = inn2['total_runs'].sum()
    chasing_team = inn2['batting_team'].iloc[0]
    chase_win = 1 if match_row.winner == chasing_team else 0
    death_legal_balls = death['is_legal_ball'].sum()
    death_rpo = (death['total_runs'].sum() / max(1, death_legal_balls)) * 6

    buf = load_or_init_venue_live_row(season=season, venue_code=venue_code)
    n = buf['matches_count']
    # Running mean update: new_avg = (old_avg * n + new_val) / (n + 1)
    buf['avg_first_innings_score']  = (buf['avg_first_innings_score'] * n + first_innings_total) / (n + 1)
    buf['avg_second_innings_score'] = (buf['avg_second_innings_score'] * n + second_innings_total) / (n + 1)
    buf['chase_success_rate']       = (buf['chase_success_rate'] * n + chase_win) / (n + 1)
    buf['avg_death_rpo']            = (buf['avg_death_rpo'] * n + death_rpo) / (n + 1)
    buf['matches_count'] = n + 1

    upsert_venue_live_row(season=season, venue_code=venue_code, values=buf)
    print(f"[venue] Updated {venue_code} for season {season}: now {buf['matches_count']} matches")
```

### Live Player Stats Update (Post-Match)

After each completed season match, update a player-level live buffer for the active batters and bowlers.
Persist it to `player_live_season_stats` keyed by `(season, player)` for restart safety and replay reuse.

Minimum columns to maintain:

```text
player,
season,
bat_legal_balls,
bat_runs,
bat_boundaries,
bat_dots,
death_bat_legal_balls,
death_bat_runs,
bowl_legal_balls,
bowl_runs_conceded,
bowl_dots,
death_bowl_legal_balls,
death_bowl_runs_conceded,
death_bowl_wickets
```

Derive live SR / economy / dot% from these running totals when assembling the prompt. Keep the
online formulas simple and count-based; the offline EWMA priors remain the anchor.

**Effect on sparse venues over the course of IPL 2026:**
- Match 1 at Guwahati: model uses 30% own historical, 70% Eden Gardens prior
- Match 1 completes → `update_venue_stats_live()` → live buffer has 1 real 2026 match
- Match 2 at Guwahati: historical blend + 1 real 2026 data point fused in
- Match 3+: real 2026 pitch behaviour increasingly dominates, prior fades

### Inference Execution

**Local (laptop):** Load model with 4-bit quantization, generate with `max_new_tokens=256`. Takes 5-15 seconds on CPU or 2-3 seconds on GPU. Completely fine for over-boundary polling.

**Modal (serverless):** 
```python
import modal

app = modal.App("ipl-predictor")
model_volume = modal.Volume.from_name("ipl-model-weights")

@app.function(
    gpu="T4",
    timeout=120,
    volumes={"/model": model_volume}
)
def predict(match_state_text: str) -> dict:
    # Load model from volume, run inference, return prediction
    ...
```
Cold start: ~30 seconds (model load). Warm: ~3 seconds. Cost: ~$0.001 per call.

### Match Finalization Job

The live polling script writes predictions during the chase. A separate **match finalization**
step must run once the match is complete.

Responsibilities:
1. Fetch the final winner, margin, and completion status
2. Mark the `matches` row as `completed` or `abandoned`
3. If the match became DLS / reduced-overs, mark `is_dl_match=True` and remove any existing predictions
4. Otherwise compute `model_brier_score` and `baseline_brier_score` for every saved prediction row
5. Update the release-scoped `season_metrics` row for `(season=2026, release_version, run_scope='prod')`
6. Update `player_live_season_stats` for `season=2026`
7. Update `venue_live_season_stats` for `season=2026`

```python
def finalize_match(match_id):
    match = fetch_final_match_result(match_id)
    match_row = supabase.table('matches').select(
        'assigned_release_version,chasing_team'
    ).eq('id', match_id).single().execute().data
    release_version = match_row['assigned_release_version']

    if match.is_dl_match or match.scheduled_overs != 20:
        supabase.table('matches').update({
            'winner': match.winner,
            'winning_margin': match.margin,
            'status': match.status,
            'is_dl_match': True,
        }).eq('id', match_id).execute()
        supabase.table('predictions').delete() \
            .eq('match_id', match_id) \
            .eq('release_version', release_version) \
            .eq('run_scope', 'prod') \
            .execute()
        return

    preds = supabase.table('predictions').select('*') \
        .eq('match_id', match_id) \
        .eq('release_version', release_version) \
        .eq('run_scope', 'prod') \
        .execute().data
    actual = 1.0 if match.winner == match_row['chasing_team'] else 0.0

    for row in preds:
        model_brier = (row['predicted_probability'] - actual) ** 2
        baseline_brier = (row['baseline_probability'] - actual) ** 2
        supabase.table('predictions').update({
            'model_brier_score': model_brier,
            'baseline_brier_score': baseline_brier,
        }).eq('id', row['id']).execute()

    refresh_season_metrics(season=2026, release_version=release_version, run_scope="prod")
    deliveries_df = load_persisted_live_ball_events(match_id)
    matches_df = load_persisted_match_metadata(match_id)
    update_player_stats_live(season=2026, match_id=match_id, deliveries_df=deliveries_df, matches_df=matches_df, ...)
    update_venue_stats_live(season=2026, match_id=match_id, ..., deliveries_df=deliveries_df, matches_df=matches_df)
```

When the poller writes the first production prediction for a live match, it must also set:
```python
supabase.table('matches').update({
    'assigned_release_version': ACTIVE_RELEASE_VERSION,
    'status': 'live',
}).eq('id', match_id).is_('assigned_release_version', None).execute()
```

If `assigned_release_version` is already set, the poller must use that stored value for all
subsequent prediction writes for the same match, even if `ACTIVE_RELEASE_VERSION` changes later.

Without this finalization job, the website cannot show completed-match accuracy, season-wide Brier
averages, or trustworthy baseline comparisons.

### Supabase Schema

Create tables in this order so foreign keys resolve cleanly:
1. `model_releases`
2. `matches`
3. `predictions`
4. `season_metrics`
5. `player_live_season_stats`
6. `venue_live_season_stats`
7. `live_ball_events`

**`matches` table:**
```sql
CREATE TABLE matches (
    id TEXT PRIMARY KEY,           -- canonical internal key, format espn_<objectId>
    espn_object_id TEXT UNIQUE NOT NULL,
    espn_match_slug TEXT,
    espn_series_slug TEXT,
    cricketdata_match_id TEXT UNIQUE,
    assigned_release_version TEXT REFERENCES model_releases(release_version),
    date DATE,
    team1 TEXT,
    team2 TEXT,
    chasing_team TEXT,
    target INTEGER,
    venue TEXT,
    winner TEXT,                   -- NULL until match complete
    winning_margin TEXT,
    status TEXT DEFAULT 'scheduled', -- scheduled | live | completed | abandoned
    is_dl_match BOOLEAN DEFAULT FALSE
);
```

**`predictions` table:**
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    match_id TEXT REFERENCES matches(id),
    release_version TEXT NOT NULL,
    run_scope TEXT NOT NULL DEFAULT 'prod', -- prod | replay | backfill | experiment
    model_version TEXT,
    prompt_version TEXT,
    baseline_version TEXT,
    snapshot_over INTEGER,
    chasing_team TEXT,
    runs_scored INTEGER,
    wickets_fallen INTEGER,
    balls_remaining INTEGER,
    required_run_rate FLOAT,
    current_run_rate FLOAT,
    batter_a TEXT,
    batter_b TEXT,
    last_over_bowler TEXT,
    predicted_probability FLOAT,
    baseline_probability FLOAT,
    reasoning_summary TEXT,        -- Full public <analysis>...</analysis> content
    model_brier_score FLOAT,       -- NULL until match completes
    baseline_brier_score FLOAT,    -- NULL until match completes
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Idempotency constraint within one immutable release/run scope
    CONSTRAINT unique_prediction UNIQUE (match_id, snapshot_over, release_version, run_scope)
);
```

**Always use upsert, not insert:**
```python
match_release_version = get_match_assigned_release_version(match_id) or ACTIVE_RELEASE_VERSION

supabase.table('predictions').upsert(
    {
        'match_id': match_id,
        'snapshot_over': over,
        'release_version': match_release_version,
        'run_scope': 'prod',
        'model_version': MODEL_VERSION,
        'prompt_version': PROMPT_VERSION,
        'baseline_version': BASELINE_VERSION,
        ...
    },
    on_conflict='match_id,snapshot_over,release_version,run_scope'
).execute()
```

Without this, a script crash mid-write + retry creates two rows for the same over. Release-scoped keys also prevent replays or mid-season shadow runs from overwriting the production record.

**`model_releases` table:**
```sql
CREATE TABLE model_releases (
    release_version TEXT PRIMARY KEY,
    model_version TEXT,
    prompt_version TEXT,
    baseline_version TEXT,
    status TEXT NOT NULL, -- active | retired | shadow
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**`season_metrics` table:**
```sql
CREATE TABLE season_metrics (
    season INTEGER NOT NULL,
    release_version TEXT NOT NULL REFERENCES model_releases(release_version),
    run_scope TEXT NOT NULL DEFAULT 'prod',
    total_predictions INTEGER DEFAULT 0,
    avg_model_brier_score FLOAT,
    avg_baseline_brier_score FLOAT,
    vs_baseline_improvement FLOAT,
    format_success_rate FLOAT,
    last_updated TIMESTAMPTZ,

    PRIMARY KEY (season, release_version, run_scope)
);
```

Implementation rule:
- Website production pages query `run_scope='prod'` joined against `matches.assigned_release_version`
- Replay and backfill jobs write to `run_scope='replay'` or `run_scope='backfill'`
- Never aggregate metrics across different `release_version` values

**`player_live_season_stats` table:**
```sql
CREATE TABLE player_live_season_stats (
    season INTEGER NOT NULL,
    player TEXT NOT NULL,
    bat_legal_balls INTEGER DEFAULT 0,
    bat_runs INTEGER DEFAULT 0,
    bat_boundaries INTEGER DEFAULT 0,
    bat_dots INTEGER DEFAULT 0,
    death_bat_legal_balls INTEGER DEFAULT 0,
    death_bat_runs INTEGER DEFAULT 0,
    bowl_legal_balls INTEGER DEFAULT 0,
    bowl_runs_conceded INTEGER DEFAULT 0,
    bowl_dots INTEGER DEFAULT 0,
    death_bowl_legal_balls INTEGER DEFAULT 0,
    death_bowl_runs_conceded INTEGER DEFAULT 0,
    death_bowl_wickets INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (season, player)
);
```

**`venue_live_season_stats` table:**
```sql
CREATE TABLE venue_live_season_stats (
    season INTEGER NOT NULL,
    venue_code TEXT NOT NULL,
    matches_count INTEGER DEFAULT 0,
    avg_first_innings_score FLOAT DEFAULT 0,
    avg_second_innings_score FLOAT DEFAULT 0,
    chase_success_rate FLOAT DEFAULT 0,
    avg_death_rpo FLOAT DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (season, venue_code)
);
```

**`live_ball_events` table (recommended if using Supabase as the persisted event store):**
```sql
CREATE TABLE live_ball_events (
    id BIGSERIAL PRIMARY KEY,
    match_id TEXT REFERENCES matches(id),
    innings INTEGER,
    over_number INTEGER,
    event_index_in_over INTEGER,
    legal_ball_number INTEGER,
    is_legal_ball BOOLEAN,
    batting_team TEXT,
    bowling_team TEXT,
    striker TEXT,
    non_striker TEXT,
    bowler TEXT,
    runs_off_bat INTEGER,
    extras INTEGER,
    wides INTEGER,
    noballs INTEGER,
    byes INTEGER,
    legbyes INTEGER,
    total_runs INTEGER,
    is_wicket BOOLEAN,
    dismissal_kind TEXT,
    player_dismissed TEXT,
    provider_event_id TEXT,
    event_ts TIMESTAMPTZ,

    CONSTRAINT unique_live_ball UNIQUE (match_id, innings, over_number, event_index_in_over)
);
```

---

## 15. Website Architecture

### Stack

- **Frontend:** Next.js (React) hosted on Vercel (free)
- **Database:** Supabase (free tier, 500MB — plenty)
- **Real-time updates:** Supabase real-time subscriptions (WebSocket)
- **Hosting:** Vercel (auto-deploy from GitHub, free)

### Pages

**`/` — Homepage: Season Overview**

Grid of all IPL 2026 matches. Each card shows:
- Match (Team A vs Team B, date, venue)
- Result badge (green = completed, orange = live, grey = upcoming)
- Small sparkline chart: probability evolution during chase (1 point per predicted over, usually up to 19 points)
- Final Brier score (once match complete)
- Color: green if avg Brier < 0.15, orange if 0.15-0.25, red if > 0.25

**`/match/[id]` — Match Detail Page**

For a completed or live match:
- Large probability timeline chart (x = snapshot over: 1 through 19; y = probability 0-1)
  - Each point is clickable
  - Horizontal reference line at 0.5
  - Actual outcome shown as colored background (green = chasing team won)
- Clicking any point expands the reasoning summary panel below:
  ```
  🏏 Over 16 • RCB need 32 off 24 balls • RRR: 8.00
  
  💭 Reasoning Summary:
  [full <analysis> content rendered here]
  
  📊 Prediction: 73% chase success
  ✅ Outcome: RCB won — Brier score: 0.073 (good)
  ```
- Match summary box: teams, target, result, overall Brier score for this match

**`/stats` — Season Accuracy Dashboard**

- Calibration chart: predicted probability bucket vs actual win rate (should be ~diagonal)
- Brier score vs. baseline comparison chart over the season
- Best predictions (model was most confident and was right)
- Worst predictions (biggest mispredictions — interesting to read)
- Rolling average Brier score by match number

### Real-Time Updates During a Live Match

Supabase supports WebSocket subscriptions on table changes:
```javascript
const subscription = supabase
  .channel('predictions')
  .on('postgres_changes', {
    event: '*',
    schema: 'public',
    table: 'predictions',
    filter: `match_id=eq.${matchId}`
  }, (payload) => {
    // Insert or correction arrived — ignore rows from replay/shadow releases
    if (payload.new.run_scope !== 'prod') return
    if (payload.new.release_version !== assignedReleaseVersion) return
    updateProbabilityChart(payload.new)
  })
  .subscribe()
```

Website visitors see new predictions appear without refreshing. The reasoning summary expands automatically when a new over's prediction arrives.

Implementation rule:
- all production website queries must filter to `run_scope='prod'` and the `assigned_release_version` of the match being displayed
- replay or shadow-release rows must never leak into the public UI
- if the season ever contains more than one production `release_version`, the `/stats` page must segment results by release rather than presenting a merged season line

---

## 16. Failure Modes and Fixes

| Failure Mode | Symptoms | Fix |
|---|---|---|
| **Model always outputs ~0.5** | `reward_std < 0.05`, accuracy reward not improving after step 200 | Check baseline_bonus is active; increase confidence bonus weight; verify temperature is 0.7+ |
| **Mode A reward hacking** | Calibration chart shows predicted-0.85 wins only 60% — model is overconfident | Reduce confidence_bonus coefficient 0.1 → 0.05; cap bonus to [0.15, 0.85] range |
| **Mode B reward hacking** | `reward_std < 0.05`, predictions clustered at 0.5, short summaries | baseline_bonus not working — verify diverged_from_baseline threshold; increase bonus weight |
| **Format never learned** | `reward/format_reward < 0.5` after step 100 | Check system prompt clarity; increase format reward weight temporarily to 0.5 for first 50 steps |
| **KL explosion** | `kl` metric rising past 0.5 | Increase `beta` from 0.04 to 0.1; reduce learning rate |
| **Entropy collapse** | `entropy < 1.0` | Model found a shortcut — inspect completion summaries immediately. Likely outputting same probability regardless of input. Increase beta significantly. |
| **Reasoning summary is gibberish** | High format reward, low accuracy reward, summaries say nonsense | Model learned format without cricket knowledge. Add more cricket-specific SFT examples. Verify player stats are in the prompt. |
| **Completion length growing** | `completion_length` trending upward past 160 | Add length penalty: `-0.001 * max(0, len(completion) - 160)` in reward function |
| **Player names not matching in training** | `is_estimated=True` for >5% of known players | Team name normalization bug or Cricsheet format variant. Print `player_season_stats` player list and diff against delivery player names. |
| **Player names not matching in inference** | Live predictions always show "(career data limited)" | `player_name_map.csv` missing entries. Print unmatched names from `cricdata` scorecard response and add to map. |
| **Wrong target live** | RRR looks impossible or trivial mid-match | DLS revision occurred. Check `api_response['target_runs']` vs `first_innings_total + 1`, mark `is_dl_match=True`, and skip further predictions for that match. |
| **Duplicate predictions** | Release-scoped `season_metrics` drifting unexpectedly | Missing UNIQUE constraint on `(match_id, snapshot_over, release_version, run_scope)`. Add the constraint and dedup within each release/run scope before recomputing aggregates. |
| **Impact Player bowler confusion** | Model rates fresh over-18 bowler as "saving overs" in reasoning summary | Add `is_impact_player` flag to prompt if API provides it; add to known limitations note if it doesn't |
| **Temporal leakage discovered** | Suspicious model performance on known historical matches | Re-check season cutoff logic in `player_season_stats` and `venue_season_stats`. For a season S prior, only matches completed before the season-start cutoff should be included. |
| **Same-match GRPO batches** | Low `reward_std`, slow convergence despite good format learning | Cooldown scheduler not applied or validation check skipped. Rebuild training order with the Section 8 scheduler and re-run the adjacency/window checks. |
| **Over-boundary state mismatch** | Predictions feel off immediately after over breaks | Verify you are using end-of-over state labels only: unordered on-field batters, `NEW_BATTER_PENDING` where needed, and `last_over_bowler` rather than "current bowler". |

---

## 17. Project Timeline

Given that IPL 2026 started on **March 28, 2026** and only the **first phase through April 12, 2026**
is officially announced as of **April 1, 2026**, here is a realistic schedule:

### Week 1 (April 1-7): Data Pipeline
- [ ] Download Kaggle IPL 2008-2025 dataset
- [ ] Run data cleaning: exclusions, name normalization, over indexing fix
- [ ] Build `merged_deliveries` with dates attached
- [ ] Compute `player_season_stats.csv` and `venue_season_stats.csv` through `as_of_season=2026`
- [ ] Materialize and check in `venue_metadata.csv` covering every venue used in offline training/replay/live inference
- [ ] Validate point-in-time correctness with spot checks (e.g., V Kohli's as_of_season=2015 stats should show no 2016 data)
- [ ] Build snapshot generator, validate on 5 full chases manually
- [ ] Compute `run_rate_baseline_prob` for all snapshots using logistic regression
- [ ] Serialize the canonical logistic baseline artifact and metadata (`baseline_version`, coefficients, feature order)
- [ ] Generate final `training_dataset.csv`
- [ ] Validate every-over snapshot generation on a full chase (overs 1-19)

### Week 2 (April 8-14): SFT + Initial GRPO
- [ ] Write/generate 200-300 SFT examples, review 50 manually
- [ ] Run SFT warmup on Colab T4 (~2 hours)
- [ ] Verify SFT output format quality on 20 examples
- [ ] Set up wandb tracking
- [ ] Start first GRPO run (300 steps, monitoring only)
- [ ] Check format reward reaches >0.9 by step 100

### Week 3 (April 15-21): GRPO Tuning
- [ ] Review first GRPO run reasoning summaries in wandb
- [ ] Run qualitative eval suite (20 hand-crafted scenarios)
- [ ] Tune hyperparameters based on metrics
- [ ] Run second GRPO training (500 steps, best config)
- [ ] Evaluate on 2024 validation set: compute Brier score
- [ ] Run 2024 chronological replay evaluation with rolling live buffers
- [ ] Freeze a release bundle: `model_version`, `prompt_version`, `baseline_version` → `release_version`

### Week 4 (April 22-28): Inference Pipeline
- [ ] Build match state assembler (from live API → prompt text)
- [ ] Implement shared live/offline snapshot feature builder from ball-by-ball logs
- [ ] Implement over backfill logic for missed polling intervals
- [ ] Build season-scoped live buffer tables (`player_live_season_stats`, `venue_live_season_stats`) and make finalizer the sole writer
- [ ] Build Supabase schema and Python writer
- [ ] Register active production release in `model_releases`
- [ ] Build `player_name_map.csv` using 2024-2025 history plus 2026 squad / live-feed additions
- [ ] Build match finalization job (winner + Brier + season stats + live buffer updates)
- [ ] Test full pipeline on a 2025 match replay (not live)
- [ ] Deploy inference script on Modal OR set up local cron job
- [ ] **First live prediction on the next officially announced IPL 2026 match after the inference stack is ready**

### Week 5 (April 29 – May 7): Website
- [ ] Build Next.js site with Supabase client
- [ ] Homepage with match cards
- [ ] Match detail page with probability chart and reasoning-summary viewer
- [ ] Real-time subscription for live updates
- [ ] Deploy to Vercel
- [ ] Share on Twitter/blog with first week of predictions

### Weeks 6-8 (May 8 onward, aligned to official fixture announcements): Run and Blog
- [ ] Run live predictions for all remaining officially announced IPL 2026 matches
- [ ] Monitor for pipeline failures after each match
- [ ] Write blog post when enough data (>20 matches predicted)
- [ ] Final frozen-model test evaluation on full 2025 data (static + replay)
- [ ] Push trained model to Hugging Face Hub

---

## 18. Resources and References

### Datasets
- **Primary IPL data:** https://www.kaggle.com/datasets/anukaggle81/ipl-ball-by-ball-dataset-2008-2025
- **Cricsheet (original source, updated):** https://cricsheet.org/matches/ipl/ — download the "CSV" format for all IPL matches
- **Alternative Kaggle dataset:** https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020 (cross-reference for cleaning)

### Model
- **Qwen2.5-1.5B-Instruct:** https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- **Qwen2.5-3B-Instruct (v2):** https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

### TRL Framework
- **TRL v1.0 documentation:** https://huggingface.co/docs/trl/main/en/grpo_trainer
- **TRL docs for the pinned training stack:** use the docs matching your pinned `trl` release only (for v1, pin `trl==0.29.0` and use the matching TRL documentation pages throughout)
- **TRL logging and metrics guide:** https://huggingface.co/docs/trl/logging
- **TRL GitHub:** https://github.com/huggingface/trl
- **TRL v1.0 blog post:** https://huggingface.co/blog/trl-v1

### GRPO Theory
- **DeepSeekMath paper (GRPO origin):** https://arxiv.org/abs/2402.03300
- **DeepSeek-R1 paper (GRPO + reasoning):** https://arxiv.org/abs/2501.12948
- **Cameron Wolfe's GRPO deep dive:** https://cameronrwolfe.substack.com/p/grpo
- **Unsloth GRPO guide (practical):** https://docs.unsloth.ai/basics/reinforcement-learning-guide

### Related Work (Inspiration)
- **AlphaMaze (GRPO for spatial reasoning):** https://arxiv.org/abs/2502.14669 — most similar in spirit: non-math GRPO domain
- **Outcome-based RL for Polymarket forecasting:** https://arxiv.org/abs/2505.17989 — closest work to this project: calibrated probability estimation with binary delayed rewards
- **Reasoning-SQL (GRPO for Text-to-SQL):** https://arxiv.org/html/2503.23157v1 — partial rewards for structured output

### Infrastructure
- **Supabase (database):** https://supabase.com — free tier sufficient
- **Vercel (website hosting):** https://vercel.com — free tier sufficient
- **Modal (serverless inference):** https://modal.com — pay-per-second GPU
- **Hugging Face Inference Endpoints:** https://huggingface.co/docs/inference-endpoints — alternative to Modal
- **Unsloth (VRAM-efficient GRPO):** https://github.com/unslothai/unsloth — 80% less VRAM than vanilla HF, enables 1.5B training on 5GB VRAM

### Cricket Data Sources (for live inference)
- **`cricdata` Python package:** https://pypi.org/project/cricdata/ — unofficial ESPNCricinfo client, ML-native, free
- **CricketData.org free tier:** https://cricketdata.org — 100 hits/day free, structured JSON, backup source

### Calibration Theory
- **Brier Score:** https://en.wikipedia.org/wiki/Brier_score
- **Expected Calibration Error:** Standard metric from forecasting literature; implementation in sklearn as `sklearn.calibration.calibration_curve`

---

## Appendix A: Quick Data Sanity Checks

Before training, verify these invariants hold in your processed data:

**Data correctness:**

1. **No future leakage:** For any training example from season S, all historical player stats in the prompt must come from the pre-season prior row `as_of_season=S`. Spot check: take a 2014 match, find a player in the prompt, look up that player's `as_of_season=2014` row in `player_season_stats.csv`, and verify the stats don't include anything from 2014 matches.

2. **Target computation correct:** Sum deliveries for inning==1 of a known match and verify `target = sum(total_runs) + 1`. Cross-check against Wikipedia or ESPNcricinfo for 5 matches. Confirm `total_runs` includes byes and leg byes (not just `runs_off_bat`).

3. **Legal ball counting correct:** Take a known over that had 1 wide. Verify that `balls_bowled` counts 6 legal balls (not 7). The wide should appear in the `wides` column with a non-null value. **Check this in at least 3 places in the code:** `balls_bowled`, `balls_remaining`, and per-over RPO calculations.

4. **Chasing team derivation correct:** For a known match where the team batting second won, verify `did_chasing_team_win == 1`. For a match where the team batting first defended successfully, verify `did_chasing_team_win == 0`.

5. **Train/test split no overlap:** Verify zero match IDs appear in both training and test sets.

6. **Prompt length:** Check that all prompts are under 512 tokens (the `max_prompt_length` limit). If not, shorten venue context or player stats.

**Team/venue normalization:**

7. **No stray team name variants:** After applying `TEAM_NAME_CANONICAL`, run `assert set(matches['team1'].unique()).issubset(KNOWN_TEAMS)`. Should have exactly 10-14 unique team names across history. If `unique()` shows more, there are unnormalized variants.

8. **Venue code coverage:** Every raw `venue` value in `matches.csv` should map to a known alias in `VENUE_CODES` after normalization. Run the unmapped-venue report after alias resolution — it should return empty. Unmapped venues will silently return no venue stats.

**Feature correctness:**

9. **Partnership runs non-negative:** `partnership_runs` must be >= 0 and <= `runs_scored`. Any negative value means the wicket over was mis-identified. Spot check 10 snapshots manually.

10. **`is_day_match` coverage:** Verify the match time field is populated for at least 90% of 2019+ matches (older data may not have it). For matches with missing time, default to `is_day_match=False` (evening) since most IPL matches are evening.

11. **Same-match cooldown check:** After the cooldown scheduler (Section 8), run a check: in any window of 16 consecutive training rows, no `match_id` should appear more than twice, and adjacent duplicate `match_id`s should be zero in the normal case. If violated, reshuffle with a new seed.

**Brier score baseline check:**

12. **Baseline sanity:** The canonical logistic baseline on `required_run_rate + balls_remaining + wickets_in_hand` should achieve Brier ~0.16-0.18 on the 2024 validation set. If it's above 0.22, your features or target variable are wrong. If it's below 0.13, you may have leakage.

13. **Replay-mode sanity:** On the 2024 chronological replay validation, the deployed inference
stack with rolling live buffers should not materially underperform the static 2024 validation setup.
Then verify the same once on the frozen 2025 final test. If replay Brier degrades sharply, the
live-buffer merge logic is wrong or too unstable.

14. **Versioning present:** Every production prediction row must include `model_version`,
`prompt_version`, `baseline_version`, and `release_version`. If any of these are NULL, season-level
comparisons are not trustworthy.

15. **Baseline artifact consistency:** The baseline probability stored in a live or replay prediction
row must equal the output of the serialized canonical baseline artifact for the same feature vector.
If it differs, the scorer has forked across environments.

16. **Release scoping correct:** For any production UI query, verify that all rows come from the
match's `assigned_release_version` with `run_scope='prod'`. Replay rows must never be visible publicly.

17. **No false corruption exclusions:** Spot-check at least 5 successful short chases and verify
they remain in the dataset. The preprocessing code must never exclude a valid match solely because
the chase ended quickly.

18. **Live event idempotency:** Re-run the same polled live scorecard twice and verify the
`live_ball_events` row count does not change on the second upsert. If it grows, the uniqueness key
is wrong.

19. **Canonical match identity:** Verify the same `match_id` flows through `matches`,
`predictions`, `live_ball_events`, and the provider-map file even when the fallback API is used.

20. **Player-name coverage for live season:** Before the first live prediction, verify every player
in the announced 2026 squads or the first week of live scorecards resolves through `player_name_map.csv`
or a deliberate league-average fallback path. Do not rely on discovering all missing names mid-match.

## Appendix B: Venue Code Mapping

```python
VENUE_CODES = {
    # Rich history — full data from training set
    "Wankhede Stadium":                                     "wankhede",
    "Wankhede Stadium, Mumbai":                             "wankhede",
    "M. Chinnaswamy Stadium":                               "chinnaswamy",
    "M Chinnaswamy Stadium":                                "chinnaswamy",
    "MA Chidambaram Stadium, Chepauk":                      "chepauk",
    "MA Chidambaram Stadium, Chepauk, Chennai":             "chepauk",
    "Eden Gardens":                                         "eden_gardens",
    "Rajiv Gandhi International Stadium":                   "rajiv_gandhi_hyderabad",
    "Rajiv Gandhi International Stadium, Uppal":            "rajiv_gandhi_hyderabad",
    "Sawai Mansingh Stadium":                               "sawai_mansingh",
    "Arun Jaitley Stadium":                                 "arun_jaitley_delhi",
    "Feroz Shah Kotla":                                     "arun_jaitley_delhi",

    # Moderate history
    "Narendra Modi Stadium":                                "narendra_modi_ahmedabad",
    "Narendra Modi Stadium, Ahmedabad":                     "narendra_modi_ahmedabad",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Stadium": "ekana_lucknow",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "ekana_lucknow",
    "Punjab Cricket Association IS Bindra Stadium":          "mohali",   # Pre-2024 PBKS home; used as similarity prior for Mullanpur
    "Punjab Cricket Association Stadium, Mohali":           "mohali",

    # Sparse — active in IPL 2026, use graduated blending
    "HPCA Cricket Stadium":                                 "hpca_dharamshala",  # ~12 matches
    "Himachal Pradesh Cricket Association Stadium":         "hpca_dharamshala",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": "mullanpur",  # ~8 matches; PBKS home from 2024
    "Maharaja Yadavindra Singh International Cricket Stadium": "mullanpur",
    "ACA Cricket Stadium":                                  "aca_guwahati",      # ~6 matches; RR secondary home from 2023
    "Barsapara Cricket Stadium":                            "aca_guwahati",
    "Barsapara Cricket Stadium, Guwahati":                  "aca_guwahati",
    "Nava Raipur Cricket Stadium":                          "raipur",            # ~4 matches; RCB secondary home in 2026
    "Shaheed Veer Narayan Singh International Stadium":     "raipur",            # Alternate name for same ground
    "Shaheed Veer Narayan Singh International Stadium, Raipur": "raipur",

    # In training data but retired from IPL 2026 — never queried during inference
    "Dr. DY Patil Sports Academy":                          "dy_patil",
    "Brabourne Stadium":                                    "brabourne",
    "Maharashtra Cricket Association Stadium":               "pune_mca",
    "MCA International Stadium":                            "pune_mca",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "vizag_vdca",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "vizag_vdca",

    # Historical neutral venues used in the 2013-2025 training window
    "Dubai International Cricket Stadium":                  "dubai",
    "Dubai International Stadium":                          "dubai",
    "Sharjah Cricket Stadium":                              "sharjah",
    "Sheikh Zayed Stadium":                                 "abu_dhabi",
    "Sheikh Zayed Stadium, Abu Dhabi":                      "abu_dhabi",
    "Zayed Cricket Stadium, Abu Dhabi":                     "abu_dhabi",
}

# Similarity pairings for graduated blending (see Section 6)
VENUE_SIMILAR_TO = {
    "mullanpur":        "mohali",                   # Same region (Punjab), similar pitch profile
    "aca_guwahati":     "eden_gardens",             # Similar climate zone, batting-friendly
    "raipur":           "narendra_modi_ahmedabad",  # Flat central India, batting surface
    "hpca_dharamshala": "chinnaswamy",              # Elevated ground, pace assistance early
    "dy_patil":         "wankhede",                 # Mumbai metro, similar conditions
    "brabourne":        "wankhede",                 # Mumbai metro, similar conditions
}
```

Map venue strings from the dataset to these codes. Use the explicit alias table first, and use
fuzzy matching only as a warning-producing fallback for truly unseen variants.

---

---

## Appendix C: Venue Metadata Seed Reference

Used to materialize `venue_metadata.csv`, reason about similarity pairings, and understand why each
pairing was chosen. Also useful for manual qualitative eval (Group 3 scenarios in Section 13).

| venue_code | pitch_type | boundary_size | dew_factor | climate | spin_assist | pace_assist | notes |
|---|---|---|---|---|---|---|---|
| wankhede | batting | medium | high | coastal_mumbai | low | medium | Sea breeze → swing early; dew heavy under lights; strong chase record |
| chinnaswamy | batting | short | medium | elevated_blr | medium | low | Short square boundaries → 200+ totals common; bowling nightmare |
| chepauk | spin | medium | medium | coastal_humid | high | low | Turns square; high humidity; historically bat-first ground |
| eden_gardens | balanced | medium | medium | eastern_humid | medium | medium | 68,000-seat atmosphere; good pace+spin balance; moderate dew |
| rajiv_gandhi_hyderabad | balanced | medium | low | deccan | medium | medium | Even bat/ball contest; chase win rate exactly 50% historically |
| sawai_mansingh | batting | medium | low | dry_jaipur | medium | low | Flat; batsman-friendly; Royals' fortress |
| arun_jaitley_delhi | batting | medium | low | northern_dry | low | medium | Hot conditions; pace gets early nip; flat later |
| narendra_modi_ahmedabad | batting | large | low | central_dry | low | low | World's largest stadium; large boundaries dampen scoring slightly |
| ekana_lucknow | batting | medium | medium | northern_plains | medium | medium | Relatively new (2022); producing high-scoring games |
| mohali | balanced | medium | low | punjab | medium | medium | PCA ground; PBKS old home; good all-round conditions |
| hpca_dharamshala | pace | medium | low | elevated_himachal | low | high | Highest IPL ground; mountain air → swing; smallest capacity |
| mullanpur | balanced | medium | low | punjab | medium | medium | New ground (2024); similar climate to Mohali; limited data |
| aca_guwahati | batting | medium | medium | northeast_humid | medium | medium | Batting-friendly; avg 1st innings ~174; local hero effect (Parag) |
| raipur | batting | medium | low | central_india | low | low | Flat surface; good carry; neutral venue feel |
| dy_patil | batting | medium | high | coastal_mumbai | low | medium | Mumbai outskirts; similar to Wankhede; neutral venue |
| brabourne | batting | medium | medium | coastal_mumbai | low | medium | Mumbai; compact ground; rarely used |

---

## Appendix D: Known Limitations (Won't Fix)

These are inherent constraints of the current design. Document them so future debugging
doesn't chase them as bugs, and so the blog post can honestly describe what the model
can and cannot see.

**1. Playing XI is unknown beyond on-field players.**
The prompt includes the two batters currently on the field and the last-over bowler. It has no
visibility into the quality of batsmen yet to come. The model uses `wickets_in_hand` as
a proxy — 6 wickets in hand usually implies 4-5 capable batsmen remaining — but it can't
distinguish "Dhoni + 3 tail-enders" from "5 quality middle-order hitters." This is
inherent to ball-by-ball data without ball-by-ball batting order.

**2. Bowling resources remaining are estimated, not known.**
We track `last_over_bowler_overs_used` for the most recent over but not for all other bowlers. The
model can't reason about "Bumrah has 2 overs left while every other quality bowler is
done." The over number + last-over bowler is a reasonable proxy but imperfect.

**3. Season-level scoring environment shift.**
Average death over RPO rose from ~8.5 (2013) to ~10.5 (2023). EWMA handles individual
player form, but not the structural drift in the overall game. A model trained on
2013-2023 data will predict slightly conservatively vs. 2026 scoring realities. The
SFT recency bias (prefer 2019-2023 examples) partially mitigates this.

**4. Impact Player rule not in training data cleanly.**
Introduced partway through the training window. Pre-2023 examples don't encode it at
all. Even 2023+ examples don't have an explicit flag. The model learns a confused
representation of "fresh over-18 bowler" that conflates impact substitutes with regular
depth bowlers. Live inference adds a prompt note when the API exposes this, but it is
absent from all training data.

**5. Discrete over-boundary snapshots only.**
The inference pipeline produces one prediction per completed over — not ball-by-ball.
A collapse where 3 wickets fall in a single over is only visible at the next boundary.
The website shows discrete probability jumps, not a continuous probability curve.

---

*Last updated: April 1, 2026*
*Project: IPL Win Probability Reasoner — GRPO + TRL + IPL 2026 Live Predictions*
