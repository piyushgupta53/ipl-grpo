"""SFT candidate selection and draft warmup dataset generation."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ipl_reasoner.paths import ProjectPaths
from ipl_reasoner.prompt_format import ensure_assistant_turn

PREFERRED_SFT_SEASONS = {"2019", "2020", "2021", "2022", "2023"}


@dataclass(frozen=True)
class SFTArtifacts:
    candidate_csv_path: Path
    draft_jsonl_path: Path
    review_pack_csv_path: Path
    anchor_review_pack_csv_path: Path
    gold_review_pack_csv_path: Path
    reviewed_jsonl_path: Path
    reviewed_only_jsonl_path: Path
    first_pass_all_jsonl_path: Path
    anchor_v1_jsonl_path: Path
    gold_v1_jsonl_path: Path
    warmup_training_jsonl_path: Path


def build_sft_artifacts(training_dataset: pd.DataFrame, paths: ProjectPaths) -> SFTArtifacts:
    paths.ensure()
    candidates = select_sft_candidates(training_dataset)

    candidate_csv_path = paths.processed / "sft_candidate_examples.csv"
    draft_jsonl_path = paths.processed / "sft_warmup_drafts.jsonl"
    review_pack_csv_path = paths.processed / "sft_review_pack.csv"
    anchor_review_pack_csv_path = paths.manual / "sft_anchor_review_pack_v1.csv"
    gold_review_pack_csv_path = paths.manual / "sft_gold_review_pack_v1.csv"
    reviewed_jsonl_path = paths.processed / "sft_warmup_reviewed.jsonl"
    reviewed_only_jsonl_path = paths.processed / "sft_warmup_reviewed_only.jsonl"
    first_pass_all_jsonl_path = paths.processed / "sft_warmup_first_pass_all.jsonl"
    anchor_v1_jsonl_path = paths.processed / "sft_warmup_anchor_v1.jsonl"
    gold_v1_jsonl_path = paths.processed / "sft_warmup_gold_v1.jsonl"
    warmup_training_jsonl_path = paths.processed / "sft_warmup_training.jsonl"

    candidates.to_csv(candidate_csv_path, index=False)
    _write_sft_jsonl(candidates, draft_jsonl_path)
    review_pack = _write_review_pack(candidates, review_pack_csv_path)
    first_pass_all = _apply_first_pass_review(candidates.copy())
    _write_reviewed_jsonl(review_pack, reviewed_jsonl_path)
    _write_prompt_completion_jsonl(review_pack, reviewed_only_jsonl_path, response_column="approved_response", response_source="reviewed_only")
    _write_prompt_completion_jsonl(first_pass_all, first_pass_all_jsonl_path, response_column="approved_response", response_source="first_pass_all")
    _write_warmup_training_jsonl(first_pass_all, warmup_training_jsonl_path)

    if anchor_review_pack_csv_path.exists():
        anchor_pack = _load_manual_review_pack(
            anchor_review_pack_csv_path,
            id_column="anchor_example_id",
            response_source="anchor_v1",
        )
        if not anchor_pack.empty:
            _write_manual_jsonl(anchor_pack, anchor_v1_jsonl_path, id_column="anchor_example_id", response_source="anchor_v1")
        elif anchor_v1_jsonl_path.exists():
            anchor_v1_jsonl_path.unlink()

    gold_pack = _load_manual_review_pack(
        gold_review_pack_csv_path,
        id_column="gold_example_id",
        response_source="gold_v1",
    )
    _write_manual_jsonl(gold_pack, gold_v1_jsonl_path, id_column="gold_example_id", response_source="gold_v1")

    return SFTArtifacts(
        candidate_csv_path=candidate_csv_path,
        draft_jsonl_path=draft_jsonl_path,
        review_pack_csv_path=review_pack_csv_path,
        anchor_review_pack_csv_path=anchor_review_pack_csv_path,
        gold_review_pack_csv_path=gold_review_pack_csv_path,
        reviewed_jsonl_path=reviewed_jsonl_path,
        reviewed_only_jsonl_path=reviewed_only_jsonl_path,
        first_pass_all_jsonl_path=first_pass_all_jsonl_path,
        anchor_v1_jsonl_path=anchor_v1_jsonl_path,
        gold_v1_jsonl_path=gold_v1_jsonl_path,
        warmup_training_jsonl_path=warmup_training_jsonl_path,
    )


def build_sft_anchor_pack(
    training_dataset: pd.DataFrame,
    paths: ProjectPaths,
    target_examples: int = 40,
    overwrite: bool = False,
) -> Path:
    paths.ensure()
    output_path = paths.manual / "sft_anchor_review_pack_v1.csv"
    backup_path = paths.manual / "sft_anchor_review_pack_v1.blank_backup.csv"

    if output_path.exists() and not overwrite:
        return output_path

    candidates = select_sft_candidates(training_dataset)
    anchor_df = _select_anchor_pack(candidates, target_examples=target_examples)
    anchor_df.to_csv(output_path, index=False)
    anchor_df.to_csv(backup_path, index=False)
    return output_path


def select_sft_candidates(training_dataset: pd.DataFrame, total_examples: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)

    df = training_dataset.copy()
    df["season"] = df["season"].astype(str)
    df = df.loc[(df["split"] == "train") & (df["season"].isin(PREFERRED_SFT_SEASONS))].copy()
    df["baseline_prob"] = df["run_rate_baseline_prob"].astype(float)
    df["required_run_rate"] = df["required_run_rate"].astype(float)
    df["wickets_in_hand"] = df["wickets_in_hand"].astype(int)
    df["partnership_runs"] = df["partnership_runs"].astype(int)

    easy = df.loc[
        (df["baseline_prob"] <= 0.15)
        | (df["baseline_prob"] >= 0.85)
        | (df["required_run_rate"] <= 6.0)
        | (df["required_run_rate"] >= 14.0)
    ].copy()
    interesting = df.loc[
        (df["baseline_prob"].between(0.25, 0.75))
        & (df["required_run_rate"].between(7.0, 12.0))
        & (df["wickets_in_hand"].between(3, 8))
    ].copy()
    tension = df.loc[
        (df["snapshot_over"].astype(int) >= 15)
        & (df["baseline_prob"].between(0.15, 0.85))
    ].copy()

    easy = _sample_diverse(easy, target=80, seed=seed)
    interesting = _sample_diverse(interesting, target=100, seed=seed + 1)
    tension = _sample_diverse(tension, target=60, seed=seed + 2)

    easy["sft_bucket"] = "easy"
    interesting["sft_bucket"] = "interesting"
    tension["sft_bucket"] = "late_tension"

    combined = pd.concat([easy, interesting, tension], ignore_index=True)
    combined = combined.drop_duplicates(subset=["match_id", "snapshot_over"]).reset_index(drop=True)

    if len(combined) > total_examples:
        combined = combined.sample(n=total_examples, random_state=seed).reset_index(drop=True)

    combined["draft_probability"] = combined["baseline_prob"].clip(0.01, 0.99).round(2)
    combined["draft_response"] = combined.apply(_draft_sft_response, axis=1)
    combined["review_priority"] = combined["sft_bucket"].map(
        {"interesting": "high", "late_tension": "high", "easy": "medium"}
    )

    keep_cols = [
        "match_id",
        "season",
        "snapshot_over",
        "sft_bucket",
        "review_priority",
        "required_run_rate",
        "wickets_in_hand",
        "run_rate_baseline_prob",
        "prompt",
        "draft_probability",
        "draft_response",
    ]
    return combined.loc[:, keep_cols].sort_values(["season", "match_id", "snapshot_over"]).reset_index(drop=True)


def _sample_diverse(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df = df.copy()
    df["season"] = df["season"].astype(str)
    season_groups = []
    seasons = sorted(df["season"].unique())
    per_season = max(1, target // max(1, len(seasons)))

    for idx, season in enumerate(seasons):
        group = df.loc[df["season"] == season].copy()
        n = min(len(group), per_season)
        if n > 0:
            season_groups.append(group.sample(n=n, random_state=seed + idx))

    sampled = pd.concat(season_groups, ignore_index=True) if season_groups else df.head(0).copy()
    if len(sampled) < min(target, len(df)):
        remaining = df.loc[~df.index.isin(sampled.index)].copy()
        extra_n = min(target - len(sampled), len(remaining))
        if extra_n > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=extra_n, random_state=seed + 100)],
                ignore_index=True,
            )
    return sampled


def _draft_sft_response(row: pd.Series) -> str:
    prob = float(row["draft_probability"])
    context = _parse_prompt_context(str(row["prompt"]))
    rng = random.Random(f"{row['match_id']}-{row['snapshot_over']}")
    summary = _build_context_aware_summary(context, prob=prob, rng=rng)

    return f"<analysis>\n{summary}\n</analysis>\n<answer>{prob:.2f}</answer>"


def _build_review_summary(context: dict[str, object], prob: float, seed: str) -> str:
    rng = random.Random(seed)
    primary = _pick_primary_signal(context, prob=prob, rng=rng)
    support = _pick_support_signal(context, prob=prob, primary_type=primary[0], rng=rng)
    counter = _pick_counter_signal(context, prob=prob, primary_type=primary[0], rng=rng)

    sentences = [_opening_sentence(context, prob, rng), primary[1]]
    if support is not None:
        sentences.append(support[1])
    if counter is not None and 0.18 <= prob <= 0.82:
        sentences.append(counter[1])

    return " ".join(sentences[:4])


def _build_review_note(context: dict[str, object], prob: float) -> str:
    notes = ["First-pass AI rewrite focused on the dominant match signals."]
    if prob <= 0.15 or prob >= 0.85:
        notes.append("Kept the rationale short because the game state is close to one-sided.")
    else:
        notes.append("Included a counterweight only where the match still looks genuinely live.")
    if context.get("new_batter_pending"):
        notes.append("Preserved the over-break fresh-batter edge case in the reasoning.")
    return " ".join(notes)


def _parse_prompt_context(prompt: str) -> dict[str, object]:
    context: dict[str, object] = {
        "chasing_team": "",
        "target": None,
        "bowling_team": "",
        "venue": "",
        "score": None,
        "wickets_fallen": None,
        "balls_bowled": None,
        "required_runs": None,
        "balls_remaining": None,
        "required_run_rate": None,
        "current_run_rate": None,
        "last_over_runs": None,
        "last_3_overs_runs": None,
        "last_3_overs_wickets": None,
        "dot_ball_pct_last_5": None,
        "boundary_pct_last_5": None,
        "batter_a_name": "",
        "batter_a_chase_sr": None,
        "batter_a_death_sr": None,
        "batter_a_limited": False,
        "batter_b_name": "",
        "batter_b_chase_sr": None,
        "batter_b_death_sr": None,
        "batter_b_limited": False,
        "new_batter_pending": False,
        "partnership_balls": None,
        "partnership_runs": None,
        "last_over_bowler": "",
        "last_over_bowler_overs_used": None,
        "last_over_bowler_death_economy": None,
        "last_over_bowler_death_wpo": None,
        "last_over_bowler_limited": False,
        "known_bowling_resources": [],
        "known_bowling_overs_left": None,
        "venue_limited": False,
        "venue_surface": "",
        "venue_chase_rate": None,
        "venue_avg_second_innings": None,
        "venue_death_rpo": None,
        "venue_dew": "",
        "snapshot_over": None,
    }

    lines = [line.strip() for line in prompt.splitlines()]
    for line in lines:
        if line.startswith("IPL Match"):
            match = re.search(r"IPL Match\s*-\s*(.+?) chasing (\d+) set by (.+)", line)
            if match:
                context["chasing_team"] = match.group(1).strip()
                context["target"] = int(match.group(2))
                context["bowling_team"] = match.group(3).strip()
        elif line.startswith("Venue: "):
            context["venue"] = line.removeprefix("Venue: ").strip()
        elif line.startswith("End of Over "):
            match = re.search(r"End of Over (\d+):", line)
            if match:
                context["snapshot_over"] = int(match.group(1))
        elif line.startswith("Score: "):
            match = re.search(r"Score: (\d+)/(\d+) \((\d+) balls faced\)", line)
            if match:
                context["score"] = int(match.group(1))
                context["wickets_fallen"] = int(match.group(2))
                context["balls_bowled"] = int(match.group(3))
        elif line.startswith("Need: "):
            match = re.search(r"Need: (\d+) off (\d+) \(RRR: ([0-9.]+)\)", line)
            if match:
                context["required_runs"] = int(match.group(1))
                context["balls_remaining"] = int(match.group(2))
                context["required_run_rate"] = float(match.group(3))
        elif line.startswith("Current run rate: "):
            context["current_run_rate"] = float(line.removeprefix("Current run rate: ").strip())
        elif line.startswith("Last over: "):
            context["last_over_runs"] = int(line.removeprefix("Last over: ").strip())
        elif line.startswith("Last 3 overs: "):
            match = re.search(r"Last 3 overs: (\d+) runs, (\d+) wickets", line)
            if match:
                context["last_3_overs_runs"] = int(match.group(1))
                context["last_3_overs_wickets"] = int(match.group(2))
        elif line.startswith("Last 5 overs: "):
            match = re.search(r"Last 5 overs: Dot% (\d+)% \| Boundary% (\d+)%", line)
            if match:
                context["dot_ball_pct_last_5"] = int(match.group(1)) / 100
                context["boundary_pct_last_5"] = int(match.group(2)) / 100
        elif line.startswith("Partnership: "):
            match = re.search(r"Partnership: (\d+)b, (\d+)r", line)
            if match:
                context["partnership_balls"] = int(match.group(1))
                context["partnership_runs"] = int(match.group(2))
        elif line.startswith("Last over bowler: "):
            match = re.search(
                r"Last over bowler: (.+?) \((\d+)/4 overs used, death econ ([0-9.]+), death wkts/ov ([0-9.]+)\)(.*)",
                line,
            )
            if match:
                context["last_over_bowler"] = match.group(1).strip()
                context["last_over_bowler_overs_used"] = int(match.group(2))
                context["last_over_bowler_death_economy"] = float(match.group(3))
                context["last_over_bowler_death_wpo"] = float(match.group(4))
                context["last_over_bowler_limited"] = "career data limited" in match.group(5)
        elif "over left (death econ" in line or "overs left (death econ" in line:
            match = re.search(
                r"(.+?): (\d+) over[s]? left \(death econ ([0-9.]+), death wkts/ov ([0-9.]+)\)(.*)",
                line,
            )
            if match:
                resources = list(context.get("known_bowling_resources") or [])
                resources.append(
                    {
                        "bowler": match.group(1).strip(),
                        "remaining_overs": int(match.group(2)),
                        "death_economy": float(match.group(3)),
                        "death_wpo": float(match.group(4)),
                        "limited": "career data limited" in match.group(5),
                    }
                )
                context["known_bowling_resources"] = resources
        elif line.startswith("Known overs left among used bowlers: "):
            match = re.search(r"Known overs left among used bowlers: (\d+)", line)
            if match:
                context["known_bowling_overs_left"] = int(match.group(1))
        elif line.startswith("Limited IPL history"):
            context["venue_limited"] = True
        elif line.startswith("Surface: ") or line.startswith("Surface prior: "):
            _, value = line.split(":", maxsplit=1)
            context["venue_surface"] = value.strip()
        elif line.startswith("Chase rate: "):
            match = re.search(r"Chase rate: ~?(\d+)%", line)
            if match:
                context["venue_chase_rate"] = int(match.group(1)) / 100
        elif line.startswith("Avg 2nd inns: "):
            match = re.search(r"Avg 2nd inns: ~?(\d+)", line)
            if match:
                context["venue_avg_second_innings"] = int(match.group(1))
        elif line.startswith("Death RPO: "):
            match = re.search(r"Death RPO: ~?([0-9.]+)", line)
            if match:
                context["venue_death_rpo"] = float(match.group(1))
        elif line.startswith("Dew: "):
            context["venue_dew"] = line.removeprefix("Dew: ").strip()

    batter_lines = _extract_batter_lines(lines)
    if batter_lines:
        batter_a = _parse_batter_line(batter_lines[0])
        context.update(
            {
                "batter_a_name": batter_a["name"],
                "batter_a_chase_sr": batter_a["chase_sr"],
                "batter_a_death_sr": batter_a["death_sr"],
                "batter_a_limited": batter_a["limited"],
            }
        )
    if len(batter_lines) >= 2:
        batter_b = _parse_batter_line(batter_lines[1])
        context.update(
            {
                "batter_b_name": batter_b["name"],
                "batter_b_chase_sr": batter_b["chase_sr"],
                "batter_b_death_sr": batter_b["death_sr"],
                "batter_b_limited": batter_b["limited"],
                "new_batter_pending": batter_b["name"] == "NEW_BATTER_PENDING",
            }
        )

    return context


def _extract_batter_lines(lines: list[str]) -> list[str]:
    try:
        start = lines.index("Batters:") + 1
    except ValueError:
        return []

    batter_lines: list[str] = []
    for line in lines[start:]:
        if not line or line.startswith("Partnership:"):
            break
        batter_lines.append(line)
    return batter_lines


def _parse_batter_line(line: str) -> dict[str, object]:
    if line == "NEW_BATTER_PENDING":
        return {
            "name": "NEW_BATTER_PENDING",
            "chase_sr": None,
            "death_sr": None,
            "limited": False,
        }

    match = re.search(
        r"(.+?) \(chase SR (\d+), death SR (\d+)\)(?: \(career data limited\))?$",
        line,
    )
    if not match:
        return {
            "name": line,
            "chase_sr": None,
            "death_sr": None,
            "limited": "career data limited" in line,
        }

    return {
        "name": match.group(1).strip(),
        "chase_sr": int(match.group(2)),
        "death_sr": int(match.group(3)),
        "limited": "career data limited" in line,
    }


def _pick_primary_signal(
    context: dict[str, object],
    prob: float,
    rng: random.Random,
) -> tuple[str, str]:
    preferred_direction = _preferred_direction(prob)
    signals = _build_signal_pool(context, rng)
    directional = [signal for signal in signals if signal[2] == preferred_direction]
    pool = directional or signals
    pool.sort(key=lambda item: item[3], reverse=True)
    return pool[0] if pool else ("fallback", _fallback_sentence(context, prob, rng), "neutral", 0.0)


def _pick_support_signal(
    context: dict[str, object],
    prob: float,
    primary_type: str,
    rng: random.Random,
) -> tuple[str, str, str, float] | None:
    preferred_direction = _preferred_direction(prob)
    signals = [
        signal
        for signal in _build_signal_pool(context, rng)
        if signal[0] != primary_type and signal[2] == preferred_direction
    ]
    if not signals:
        neutral = [
            signal
            for signal in _build_signal_pool(context, rng)
            if signal[0] != primary_type and signal[2] == "neutral"
        ]
        signals = neutral
    if not signals:
        return None
    signals.sort(key=lambda item: item[3], reverse=True)
    return signals[0]


def _pick_counter_signal(
    context: dict[str, object],
    prob: float,
    primary_type: str,
    rng: random.Random,
) -> tuple[str, str, str, float] | None:
    if prob <= 0.15 or prob >= 0.85:
        return None

    opposing_direction = _opposing_direction(prob)
    signals = [
        signal
        for signal in _build_signal_pool(context, rng)
        if signal[0] != primary_type and signal[2] == opposing_direction
    ]
    if not signals:
        return None
    signals.sort(key=lambda item: item[3], reverse=True)
    best = signals[0]
    if best[3] < 1.4:
        return None
    return best


def _preferred_direction(prob: float) -> str:
    if prob >= 0.6:
        return "positive"
    if prob <= 0.4:
        return "negative"
    return "neutral"


def _opposing_direction(prob: float) -> str:
    if prob >= 0.6:
        return "negative"
    if prob <= 0.4:
        return "positive"
    return "positive"


def _build_signal_pool(
    context: dict[str, object],
    rng: random.Random,
) -> list[tuple[str, str, str, float]]:
    signals: list[tuple[str, str, str, float]] = []

    rrr = float(context.get("required_run_rate") or 0.0)
    crr = float(context.get("current_run_rate") or 0.0)
    wickets_left = max(0, 10 - int(context.get("wickets_fallen") or 0))
    balls_remaining = int(context.get("balls_remaining") or 0)
    over = int(context.get("snapshot_over") or 0)
    pressure_gap = rrr - crr

    if pressure_gap >= 3.0:
        signals.append(("pressure", _pressure_negative_sentence(context, rng), "negative", 2.4 + min(1.5, pressure_gap / 2)))
    elif pressure_gap <= -1.5:
        signals.append(("pressure", _pressure_positive_sentence(context, rng), "positive", 2.2 + min(1.2, abs(pressure_gap) / 2)))
    else:
        signals.append(("pressure", _pressure_balanced_sentence(context, rng), "neutral", 1.4 + min(0.8, abs(pressure_gap))))

    if wickets_left <= 3:
        signals.append(("wickets", _wickets_negative_sentence(context, wickets_left, rng), "negative", 2.5))
    elif wickets_left >= 7 and balls_remaining >= 24:
        signals.append(("wickets", _wickets_positive_sentence(context, wickets_left, rng), "positive", 1.8))
    elif wickets_left <= 5 and over >= 14:
        signals.append(("wickets", _wickets_fragile_sentence(context, wickets_left, rng), "negative", 1.7))

    last_3_wickets = int(context.get("last_3_overs_wickets") or 0)
    last_3_runs = int(context.get("last_3_overs_runs") or 0)
    dot_pct = float(context.get("dot_ball_pct_last_5") or 0.0)
    boundary_pct = float(context.get("boundary_pct_last_5") or 0.0)
    if last_3_wickets >= 2 or dot_pct >= 0.55:
        signals.append(("momentum", _momentum_negative_sentence(context, rng), "negative", 1.9))
    elif last_3_runs >= 28 or boundary_pct >= 0.22:
        signals.append(("momentum", _momentum_positive_sentence(context, rng), "positive", 1.7))

    partnership_runs = int(context.get("partnership_runs") or 0)
    partnership_balls = int(context.get("partnership_balls") or 0)
    if context.get("new_batter_pending"):
        signals.append(("batters", _new_batter_sentence(context, rng), "negative", 1.8))
    elif partnership_runs >= 30 and partnership_balls >= 18:
        signals.append(("batters", _set_pair_sentence(context, rng), "positive", 1.6))

    death_srs = [
        value
        for value in [context.get("batter_a_death_sr"), context.get("batter_b_death_sr")]
        if value is not None
    ]
    if death_srs and over >= 15:
        avg_death_sr = sum(float(value) for value in death_srs) / len(death_srs)
        if avg_death_sr >= 160:
            signals.append(("finishers", _finisher_positive_sentence(context, rng), "positive", 1.7))
        elif avg_death_sr <= 125:
            signals.append(("finishers", _finisher_negative_sentence(context, rng), "negative", 1.3))

    chase_rate = context.get("venue_chase_rate")
    dew = str(context.get("venue_dew") or "").lower()
    venue_surface = str(context.get("venue_surface") or "")
    if chase_rate is not None:
        chase_rate = float(chase_rate)
        if chase_rate >= 0.60 and "high" in dew:
            signals.append(("venue", _venue_positive_sentence(context, rng), "positive", 1.2))
        elif chase_rate <= 0.45:
            signals.append(("venue", _venue_negative_sentence(context, rng), "negative", 1.2))
        elif "spin-friendly" in venue_surface and over >= 13:
            signals.append(("venue", _venue_surface_sentence(context, rng), "neutral", 0.9))

    bowler_overs_used = int(context.get("last_over_bowler_overs_used") or 0)
    bowler_death_econ = float(context.get("last_over_bowler_death_economy") or 0.0)
    if bowler_overs_used >= 3 and bowler_death_econ <= 8.5:
        signals.append(("bowler", _bowler_relief_sentence(context, rng), "positive", 1.0))

    bowling_resources = list(context.get("known_bowling_resources") or [])
    known_overs_left = int(context.get("known_bowling_overs_left") or 0)
    if bowling_resources:
        best_resource = sorted(
            bowling_resources,
            key=lambda item: (-int(item["remaining_overs"]), float(item["death_economy"]), -float(item["death_wpo"])),
        )[0]
        if over >= 15 and int(best_resource["remaining_overs"]) >= 2 and (
            float(best_resource["death_economy"]) <= 8.8 or float(best_resource["death_wpo"]) >= 0.4
        ):
            signals.append(("bowling_resources", _bowling_resources_negative_sentence(context, rng), "negative", 1.6))
        elif over >= 16 and known_overs_left <= 2:
            signals.append(("bowling_resources", _bowling_resources_positive_sentence(context, rng), "positive", 1.3))

    return signals


def _build_context_aware_summary(context: dict[str, object], prob: float, rng: random.Random) -> str:
    rrr = float(context.get("required_run_rate") or 0.0)
    crr = float(context.get("current_run_rate") or 0.0)
    wickets_left = max(0, 10 - int(context.get("wickets_fallen") or 0))
    balls_remaining = int(context.get("balls_remaining") or 0)
    over = int(context.get("snapshot_over") or 0)

    factors: list[tuple[float, str, str]] = []

    pressure_gap = rrr - crr
    if pressure_gap >= 3.0:
        factors.append((2.0 + min(2.0, pressure_gap / 2), "pressure_neg", _pressure_negative_sentence(context, rng)))
    elif pressure_gap <= -1.5:
        factors.append((2.0 + min(1.5, abs(pressure_gap) / 2), "pressure_pos", _pressure_positive_sentence(context, rng)))
    else:
        factors.append((1.2 + min(1.0, abs(pressure_gap)), "pressure_bal", _pressure_balanced_sentence(context, rng)))

    if wickets_left <= 3:
        factors.append((2.4, "wickets_neg", _wickets_negative_sentence(context, wickets_left, rng)))
    elif wickets_left >= 7 and balls_remaining >= 24:
        factors.append((1.8, "wickets_pos", _wickets_positive_sentence(context, wickets_left, rng)))
    elif wickets_left <= 5 and over >= 14:
        factors.append((1.6, "wickets_mid", _wickets_fragile_sentence(context, wickets_left, rng)))

    last_3_wickets = int(context.get("last_3_overs_wickets") or 0)
    last_3_runs = int(context.get("last_3_overs_runs") or 0)
    dot_pct = float(context.get("dot_ball_pct_last_5") or 0.0)
    boundary_pct = float(context.get("boundary_pct_last_5") or 0.0)
    if last_3_wickets >= 2 or dot_pct >= 0.55:
        factors.append((1.9, "momentum_neg", _momentum_negative_sentence(context, rng)))
    elif last_3_runs >= 28 or boundary_pct >= 0.22:
        factors.append((1.7, "momentum_pos", _momentum_positive_sentence(context, rng)))

    partnership_runs = int(context.get("partnership_runs") or 0)
    partnership_balls = int(context.get("partnership_balls") or 0)
    if context.get("new_batter_pending"):
        factors.append((1.8, "batters_new", _new_batter_sentence(context, rng)))
    elif partnership_runs >= 30 and partnership_balls >= 18:
        factors.append((1.6, "batters_set", _set_pair_sentence(context, rng)))

    death_srs = [
        value
        for value in [context.get("batter_a_death_sr"), context.get("batter_b_death_sr")]
        if value is not None
    ]
    if death_srs and over >= 15:
        avg_death_sr = sum(float(value) for value in death_srs) / len(death_srs)
        if avg_death_sr >= 160:
            factors.append((1.7, "finishers_pos", _finisher_positive_sentence(context, rng)))
        elif avg_death_sr <= 125:
            factors.append((1.3, "finishers_neg", _finisher_negative_sentence(context, rng)))

    chase_rate = context.get("venue_chase_rate")
    dew = str(context.get("venue_dew") or "").lower()
    venue_surface = str(context.get("venue_surface") or "")
    if chase_rate is not None:
        chase_rate = float(chase_rate)
        if chase_rate >= 0.60 and "high" in dew:
            factors.append((1.2, "venue_pos", _venue_positive_sentence(context, rng)))
        elif chase_rate <= 0.45:
            factors.append((1.2, "venue_neg", _venue_negative_sentence(context, rng)))
        elif "spin-friendly" in venue_surface and over >= 13:
            factors.append((0.9, "venue_surface", _venue_surface_sentence(context, rng)))

    bowler_overs_used = int(context.get("last_over_bowler_overs_used") or 0)
    bowler_death_econ = float(context.get("last_over_bowler_death_economy") or 0.0)
    if bowler_overs_used >= 3 and bowler_death_econ <= 8.5:
        factors.append((1.0, "bowler_relief", _bowler_relief_sentence(context, rng)))

    bowling_resources = list(context.get("known_bowling_resources") or [])
    known_overs_left = int(context.get("known_bowling_overs_left") or 0)
    if bowling_resources:
        best_resource = sorted(
            bowling_resources,
            key=lambda item: (-int(item["remaining_overs"]), float(item["death_economy"]), -float(item["death_wpo"])),
        )[0]
        if over >= 15 and int(best_resource["remaining_overs"]) >= 2 and (
            float(best_resource["death_economy"]) <= 8.8 or float(best_resource["death_wpo"]) >= 0.4
        ):
            factors.append((1.6, "bowling_resources_neg", _bowling_resources_negative_sentence(context, rng)))
        elif over >= 16 and known_overs_left <= 2:
            factors.append((1.3, "bowling_resources_pos", _bowling_resources_positive_sentence(context, rng)))

    factors.sort(key=lambda item: item[0], reverse=True)
    chosen: list[str] = []
    seen_types: set[str] = set()
    for _, factor_type, sentence in factors:
        if factor_type in seen_types:
            continue
        chosen.append(sentence)
        seen_types.add(factor_type)
        if len(chosen) >= 3:
            break

    if not chosen:
        chosen.append(_fallback_sentence(context, prob, rng))

    opener = _opening_sentence(context, prob, rng)
    sentences = [opener]
    for sentence in chosen:
        if sentence != opener:
            sentences.append(sentence)
    if len(sentences) > 4:
        sentences = sentences[:4]
    return " ".join(sentences)


def _opening_sentence(context: dict[str, object], prob: float, rng: random.Random) -> str:
    team = str(context.get("chasing_team") or "The chasing side")
    required_runs = int(context.get("required_runs") or 0)
    balls_remaining = int(context.get("balls_remaining") or 0)
    wickets_left = max(0, 10 - int(context.get("wickets_fallen") or 0))

    if prob >= 0.8:
        options = [
            f"{team} are firmly ahead here: they need {required_runs} from {balls_remaining} with {wickets_left} wickets in hand.",
            f"{team} have the chase under control at this point, with {required_runs} needed from {balls_remaining} and {wickets_left} wickets left.",
        ]
    elif prob >= 0.6:
        options = [
            f"{team} hold the edge here, but it is not done yet: {required_runs} are still needed from {balls_remaining}.",
            f"{team} are slightly in front, although the next couple of overs can still swing this chase.",
        ]
    elif prob > 0.4:
        options = [
            f"This chase is still live for {team}, and the balance is being decided by how quickly they can handle the next few overs.",
            f"{team} are in a genuinely balanced position here rather than clearly ahead or behind.",
        ]
    elif prob > 0.2:
        options = [
            f"{team} are behind the game now and need a clean acceleration soon to stay in touch.",
            f"The chase is slipping away from {team}, although one strong passage can still pull it back.",
        ]
    else:
        options = [
            f"{team} are in deep trouble here and need something extraordinary from this point.",
            f"This has become a very difficult chase for {team} given the state of the innings.",
        ]
    return rng.choice(options)


def _pressure_positive_sentence(context: dict[str, object], rng: random.Random) -> str:
    rrr = float(context.get("required_run_rate") or 0.0)
    crr = float(context.get("current_run_rate") or 0.0)
    return rng.choice(
        [
            f"The asking rate of {rrr:.2f} is below their current rate of {crr:.2f}, so the run-rate pressure is manageable rather than urgent.",
            f"Run rate is not the main problem yet: they are already scoring at {crr:.2f}, above the required {rrr:.2f}.",
        ]
    )


def _pressure_negative_sentence(context: dict[str, object], rng: random.Random) -> str:
    rrr = float(context.get("required_run_rate") or 0.0)
    crr = float(context.get("current_run_rate") or 0.0)
    return rng.choice(
        [
            f"The main squeeze is run rate: they need {rrr:.2f} while currently scoring only {crr:.2f}.",
            f"The equation is demanding now because the required rate has climbed to {rrr:.2f}, well above the current {crr:.2f}.",
        ]
    )


def _pressure_balanced_sentence(context: dict[str, object], rng: random.Random) -> str:
    rrr = float(context.get("required_run_rate") or 0.0)
    return rng.choice(
        [
            f"The asking rate of {rrr:.2f} keeps this chase alive, but it leaves very little room for a quiet over.",
            f"The rate required is workable at {rrr:.2f}, though it is high enough that one stalled over would change the picture quickly.",
        ]
    )


def _wickets_positive_sentence(context: dict[str, object], wickets_left: int, rng: random.Random) -> str:
    return rng.choice(
        [
            f"Having {wickets_left} wickets left gives the batting side room to attack rather than just survive.",
            f"The batting side still has {wickets_left} wickets in reserve, so they can absorb one mistake and keep pressing the chase.",
        ]
    )


def _wickets_negative_sentence(context: dict[str, object], wickets_left: int, rng: random.Random) -> str:
    return rng.choice(
        [
            f"Only {wickets_left} wickets remain, which means every dot-ball stretch carries collapse risk now.",
            f"With just {wickets_left} wickets left, they do not have much batting insurance if another wicket falls.",
        ]
    )


def _wickets_fragile_sentence(context: dict[str, object], wickets_left: int, rng: random.Random) -> str:
    return rng.choice(
        [
            f"{wickets_left} wickets left is enough to keep the chase on, but not enough for another messy over.",
            f"They still have {wickets_left} wickets left, though the innings is fragile enough that another breakthrough would hurt badly.",
        ]
    )


def _momentum_positive_sentence(context: dict[str, object], rng: random.Random) -> str:
    last_3_runs = int(context.get("last_3_overs_runs") or 0)
    boundary_pct = float(context.get("boundary_pct_last_5") or 0.0)
    return rng.choice(
        [
            f"The recent scoring has been healthy too: {last_3_runs} runs in the last three overs with a {boundary_pct:.0%} boundary rate over the last five.",
            f"Momentum favors the batting side right now, with {last_3_runs} from the last three overs and boundaries arriving often enough to keep the fielding side under pressure.",
        ]
    )


def _momentum_negative_sentence(context: dict[str, object], rng: random.Random) -> str:
    last_3_wickets = int(context.get("last_3_overs_wickets") or 0)
    dot_pct = float(context.get("dot_ball_pct_last_5") or 0.0)
    return rng.choice(
        [
            f"Recent momentum is poor: {last_3_wickets} wickets in the last three overs and a {dot_pct:.0%} dot-ball rate over the last five suggest the innings is getting stuck.",
            f"The chase has lost rhythm recently, with {last_3_wickets} wickets in the last three overs and too many dots at {dot_pct:.0%} across the last five.",
        ]
    )


def _set_pair_sentence(context: dict[str, object], rng: random.Random) -> str:
    batter_a = str(context.get("batter_a_name") or "the striker")
    batter_b = str(context.get("batter_b_name") or "the non-striker")
    partnership_runs = int(context.get("partnership_runs") or 0)
    return rng.choice(
        [
            f"{batter_a} and {batter_b} are already set, and their {partnership_runs}-run stand means the chase is not restarting from scratch.",
            f"The current pair have settled enough to matter, with {batter_a} and {batter_b} already putting together a {partnership_runs}-run partnership.",
        ]
    )


def _new_batter_sentence(context: dict[str, object], rng: random.Random) -> str:
    batter_a = str(context.get("batter_a_name") or "the set batter")
    return rng.choice(
        [
            f"There is also some instability here because {batter_a} will have a new partner at the start of the next over.",
            f"The next over starts with a fresh batter alongside {batter_a}, which adds uncertainty right when the chase needs continuity.",
        ]
    )


def _finisher_positive_sentence(context: dict[str, object], rng: random.Random) -> str:
    names = [name for name in [context.get("batter_a_name"), context.get("batter_b_name")] if name and name != "NEW_BATTER_PENDING"]
    pair = " and ".join(names[:2]) if names else "the current batters"
    return rng.choice(
        [
            f"{pair} also profile as credible late-innings hitters, which matters now that the chase is moving into the death overs.",
            f"The batting pair are not just surviving here; {pair} have the kind of death-over scoring profile that can keep a chase moving.",
        ]
    )


def _finisher_negative_sentence(context: dict[str, object], rng: random.Random) -> str:
    names = [name for name in [context.get("batter_a_name"), context.get("batter_b_name")] if name and name != "NEW_BATTER_PENDING"]
    pair = " and ".join(names[:2]) if names else "the current batters"
    return rng.choice(
        [
            f"{pair} are not especially dominant finishers on their historical profile, so this equation may still need an above-average burst.",
            f"The concern is that {pair} do not bring elite death-over hitting numbers, which makes this finish less straightforward than the wicket count alone suggests.",
        ]
    )


def _venue_positive_sentence(context: dict[str, object], rng: random.Random) -> str:
    venue = str(context.get("venue") or "this venue")
    return rng.choice(
        [
            f"{venue} also tends to be kinder to chases, especially with dew in play, so conditions are not working against the batting side.",
            f"The venue trends help a little as well: chasing is generally productive here and the evening conditions are usually friendlier for batting second.",
        ]
    )


def _venue_negative_sentence(context: dict[str, object], rng: random.Random) -> str:
    venue = str(context.get("venue") or "this venue")
    return rng.choice(
        [
            f"{venue} is not a venue where chases routinely cruise, so the batting side cannot assume conditions will bail them out.",
            f"Venue context leans the other way too: historical chasing numbers at {venue} are not especially generous.",
        ]
    )


def _venue_surface_sentence(context: dict[str, object], rng: random.Random) -> str:
    venue_surface = str(context.get("venue_surface") or "the surface")
    return rng.choice(
        [
            f"The {venue_surface} surface can make late acceleration uneven, so the chase still needs good shot selection rather than a blind finish.",
            f"The surface profile matters here too: {venue_surface} tracks can make boundary hitting less automatic once the pressure rises.",
        ]
    )


def _bowler_relief_sentence(context: dict[str, object], rng: random.Random) -> str:
    bowler = str(context.get("last_over_bowler") or "the last bowler")
    return rng.choice(
        [
            f"There may also be a small release next over because {bowler} has already bowled heavily and cannot simply keep applying the same pressure immediately.",
            f"The fielding side must now change from {bowler}, and that matters if he was one of their better death options.",
        ]
    )


def _bowling_resources_negative_sentence(context: dict[str, object], rng: random.Random) -> str:
    resources = list(context.get("known_bowling_resources") or [])
    if not resources:
        return _fallback_sentence(context, 0.5, rng)
    best = sorted(
        resources,
        key=lambda item: (-int(item["remaining_overs"]), float(item["death_economy"]), -float(item["death_wpo"])),
    )[0]
    bowler = str(best["bowler"])
    overs_left = int(best["remaining_overs"])
    return rng.choice(
        [
            f"The fielding side still has a serious known death option in reserve because {bowler} has {overs_left} overs left.",
            f"{bowler} still has {overs_left} overs available among the bowlers already used, which keeps real closing pressure on the batting side.",
        ]
    )


def _bowling_resources_positive_sentence(context: dict[str, object], rng: random.Random) -> str:
    known_overs_left = int(context.get("known_bowling_overs_left") or 0)
    return rng.choice(
        [
            f"Among bowlers already used, only {known_overs_left} overs remain, so the fielding side may be running short on trusted closing options.",
            f"The known bowling resources are thinning now, with only {known_overs_left} overs left among bowlers already used in the innings.",
        ]
    )


def _fallback_sentence(context: dict[str, object], prob: float, rng: random.Random) -> str:
    rrr = float(context.get("required_run_rate") or 0.0)
    wickets_left = max(0, 10 - int(context.get("wickets_fallen") or 0))
    return rng.choice(
        [
            f"The chase is mostly being decided by the combination of a required rate of {rrr:.2f} and {wickets_left} wickets still available.",
            f"The key balance here is straightforward: they need to match a required rate of {rrr:.2f} without burning through the remaining {wickets_left} wickets.",
        ]
    )


def _write_sft_jsonl(candidates: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in candidates.to_dict("records"):
            payload = {
                "prompt": _sft_prompt_text(str(row["prompt"])),
                "completion": _sft_completion_text(str(row["draft_response"])),
                "metadata": {
                    "match_id": row["match_id"],
                    "season": row["season"],
                    "snapshot_over": row["snapshot_over"],
                    "sft_bucket": row["sft_bucket"],
                    "review_priority": row["review_priority"],
                    "response_source": "draft",
                },
            }
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_review_pack(candidates: pd.DataFrame, output_path: Path, target_examples: int = 50) -> pd.DataFrame:
    review_df = _select_review_pack(candidates, target_examples=target_examples).copy()
    review_df = _apply_first_pass_review(review_df)
    keep_cols = [
        "match_id",
        "season",
        "snapshot_over",
        "sft_bucket",
        "review_priority",
        "required_run_rate",
        "wickets_in_hand",
        "run_rate_baseline_prob",
        "prompt",
        "draft_response",
        "review_status",
        "review_notes",
        "approved_response",
    ]
    review_df = review_df.loc[:, keep_cols]
    review_df.to_csv(output_path, index=False)
    return review_df


def _select_review_pack(candidates: pd.DataFrame, target_examples: int) -> pd.DataFrame:
    bucket_targets = [
        ("interesting", 20),
        ("late_tension", 20),
        ("easy", 10),
    ]
    picks: list[pd.DataFrame] = []
    used_keys: set[tuple[object, object]] = set()

    def take_rows(frame: pd.DataFrame, n: int) -> pd.DataFrame:
        if frame.empty or n <= 0:
            return frame.head(0).copy()
        frame = frame.copy()
        frame["season"] = frame["season"].astype(str)
        picked = []
        seasons = sorted(frame["season"].unique())
        per_season = max(1, n // max(1, len(seasons)))

        for season in seasons:
            group = frame.loc[frame["season"] == season].sort_values(
                ["review_priority", "run_rate_baseline_prob", "match_id", "snapshot_over"],
                ascending=[True, True, True, True],
            )
            for _, row in group.iterrows():
                key = (row["match_id"], row["snapshot_over"])
                if key in used_keys:
                    continue
                picked.append(row)
                used_keys.add(key)
                if sum(1 for item in picked if str(item["season"]) == season) >= per_season:
                    break

        if len(picked) < n:
            remainder = frame.sort_values(
                ["review_priority", "season", "match_id", "snapshot_over"],
                ascending=[True, True, True, True],
            )
            for _, row in remainder.iterrows():
                key = (row["match_id"], row["snapshot_over"])
                if key in used_keys:
                    continue
                picked.append(row)
                used_keys.add(key)
                if len(picked) >= n:
                    break

        return pd.DataFrame(picked)

    for bucket, bucket_target in bucket_targets:
        bucket_df = candidates.loc[candidates["sft_bucket"] == bucket]
        picks.append(take_rows(bucket_df, bucket_target))

    combined = pd.concat(picks, ignore_index=True) if picks else candidates.head(0).copy()
    if len(combined) < target_examples:
        remaining = candidates.loc[
            ~candidates.set_index(["match_id", "snapshot_over"]).index.isin(
                combined.set_index(["match_id", "snapshot_over"]).index
            )
        ].sort_values(["review_priority", "season", "match_id", "snapshot_over"])
        extra = remaining.head(target_examples - len(combined))
        combined = pd.concat([combined, extra], ignore_index=True)

    return combined.sort_values(["season", "sft_bucket", "match_id", "snapshot_over"]).reset_index(drop=True)


def _select_anchor_pack(candidates: pd.DataFrame, target_examples: int) -> pd.DataFrame:
    df = candidates.copy()
    df["scenario_tag"] = df.apply(_scenario_tag_for_row, axis=1)

    target_by_tag = {
        "balanced_chase": 12,
        "death_overs_tension": 10,
        "resources_but_work": 8,
        "cruise_control": 4,
        "near_collapse": 4,
        "fragile_middle_overs": 2,
    }

    picks: list[pd.DataFrame] = []
    used_keys: set[tuple[object, object]] = set()

    for idx, (scenario_tag, scenario_target) in enumerate(target_by_tag.items()):
        scenario_df = df.loc[df["scenario_tag"] == scenario_tag].sort_values(
            ["review_priority", "season", "match_id", "snapshot_over"],
            ascending=[True, True, True, True],
        )
        picked_rows: list[pd.Series] = []
        for _, row in scenario_df.iterrows():
            key = (row["match_id"], row["snapshot_over"])
            if key in used_keys:
                continue
            picked_rows.append(row)
            used_keys.add(key)
            if len(picked_rows) >= scenario_target:
                break
        if picked_rows:
            picks.append(pd.DataFrame(picked_rows))

    combined = pd.concat(picks, ignore_index=True) if picks else df.head(0).copy()

    if len(combined) < target_examples:
        remaining = df.loc[
            ~df.set_index(["match_id", "snapshot_over"]).index.isin(
                combined.set_index(["match_id", "snapshot_over"]).index
            )
        ].sort_values(["review_priority", "season", "match_id", "snapshot_over"])
        extra = remaining.head(target_examples - len(combined))
        combined = pd.concat([combined, extra], ignore_index=True)

    combined = combined.sort_values(["scenario_tag", "season", "match_id", "snapshot_over"]).reset_index(drop=True)
    combined["anchor_example_id"] = [f"anchor_{idx:03d}" for idx in range(1, len(combined) + 1)]
    combined["label_status"] = ""
    combined["gold_probability"] = ""
    combined["gold_analysis"] = ""
    combined["reviewer_notes"] = ""

    keep_cols = [
        "anchor_example_id",
        "match_id",
        "season",
        "snapshot_over",
        "sft_bucket",
        "scenario_tag",
        "review_priority",
        "required_run_rate",
        "wickets_in_hand",
        "prompt",
        "label_status",
        "gold_probability",
        "gold_analysis",
        "reviewer_notes",
    ]
    return combined.loc[:, keep_cols]


def _scenario_tag_for_row(row: pd.Series) -> str:
    rrr = float(row["required_run_rate"])
    wickets = int(row["wickets_in_hand"])
    over = int(row["snapshot_over"])
    baseline = float(row["run_rate_baseline_prob"])

    if rrr <= 6.5 and wickets >= 7 and baseline >= 0.75:
        return "cruise_control"
    if rrr >= 13.5 and wickets <= 4:
        return "near_collapse"
    if over >= 16 and 0.15 <= baseline <= 0.85:
        return "death_overs_tension"
    if wickets >= 7 and 7.5 <= rrr <= 11.5:
        return "resources_but_work"
    if 7 <= over <= 14 and wickets <= 5:
        return "fragile_middle_overs"
    return "balanced_chase"


def _apply_first_pass_review(review_df: pd.DataFrame) -> pd.DataFrame:
    reviewed_rows: list[dict[str, object]] = []
    for row in review_df.to_dict("records"):
        prob = float(row["draft_probability"])
        context = _parse_prompt_context(str(row["prompt"]))
        approved_summary = _build_review_summary(
            context=context,
            prob=prob,
            seed=f"review-{row['match_id']}-{row['snapshot_over']}",
        )
        approved_response = f"<analysis>\n{approved_summary}\n</analysis>\n<answer>{prob:.2f}</answer>"
        row["review_status"] = "first_pass_ready"
        row["review_notes"] = _build_review_note(context, prob=prob)
        row["approved_response"] = approved_response
        reviewed_rows.append(row)
    return pd.DataFrame(reviewed_rows)


def _write_reviewed_jsonl(review_pack: pd.DataFrame, output_path: Path) -> None:
    approved = review_pack.loc[review_pack["review_status"].astype(str) != "reject"].copy()
    with output_path.open("w", encoding="utf-8") as f:
        for row in approved.to_dict("records"):
            payload = {
                "prompt": _sft_prompt_text(str(row["prompt"])),
                "completion": _sft_completion_text(str(row["approved_response"])),
                "metadata": {
                    "match_id": row["match_id"],
                    "season": row["season"],
                    "snapshot_over": row["snapshot_over"],
                    "sft_bucket": row["sft_bucket"],
                    "review_priority": row["review_priority"],
                    "review_status": row["review_status"],
                    "response_source": "reviewed_only",
                },
            }
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_warmup_training_jsonl(
    first_pass_all: pd.DataFrame,
    output_path: Path,
) -> None:
    # Default next-run SFT dataset: all examples get first-pass approved responses
    # in prompt/completion format so loss is applied only to the assistant answer.
    with output_path.open("w", encoding="utf-8") as f:
        for row in first_pass_all.to_dict("records"):
            payload = {
                "prompt": _sft_prompt_text(str(row["prompt"])),
                "completion": _sft_completion_text(str(row["approved_response"])),
                "metadata": {
                    "match_id": row["match_id"],
                    "season": row["season"],
                    "snapshot_over": row["snapshot_over"],
                    "sft_bucket": row["sft_bucket"],
                    "review_priority": row["review_priority"],
                    "response_source": "first_pass_all",
                    "review_status": row.get("review_status"),
                },
            }
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _write_prompt_completion_jsonl(
    df: pd.DataFrame,
    output_path: Path,
    response_column: str,
    response_source: str,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in df.to_dict("records"):
            payload = {
                "prompt": _sft_prompt_text(str(row["prompt"])),
                "completion": _sft_completion_text(str(row[response_column])),
                "metadata": {
                    "match_id": row["match_id"],
                    "season": row["season"],
                    "snapshot_over": row["snapshot_over"],
                    "sft_bucket": row["sft_bucket"],
                    "review_priority": row["review_priority"],
                    "review_status": row.get("review_status"),
                    "response_source": response_source,
                },
            }
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _load_manual_review_pack(
    input_path: Path,
    *,
    id_column: str,
    response_source: str,
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing manual SFT review pack at {input_path}. Expected the curated CSV to exist before building SFT artifacts."
        )

    gold_df = pd.read_csv(input_path)
    required_cols = {
        id_column,
        "match_id",
        "season",
        "snapshot_over",
        "sft_bucket",
        "review_priority",
        "prompt",
        "label_status",
        "gold_probability",
        "gold_analysis",
    }
    missing = sorted(required_cols - set(gold_df.columns))
    if missing:
        raise ValueError(f"Manual SFT review pack is missing columns: {', '.join(missing)}")

    approved = gold_df.loc[gold_df["label_status"].astype(str).str.strip().eq("approved")].copy()
    if approved.empty:
        return approved

    approved["gold_probability"] = approved["gold_probability"].astype(float)
    approved["approved_response"] = approved.apply(_build_gold_response, axis=1)
    approved["response_source"] = response_source
    return approved


def _build_gold_response(row: pd.Series) -> str:
    analysis = str(row["gold_analysis"]).strip()
    probability = float(row["gold_probability"])
    return f"<analysis>\n{analysis}\n</analysis>\n<answer>{probability:.2f}</answer>"


def _write_manual_jsonl(
    gold_df: pd.DataFrame,
    output_path: Path,
    *,
    id_column: str,
    response_source: str,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for row in gold_df.to_dict("records"):
            payload = {
                "prompt": _sft_prompt_text(str(row["prompt"])),
                "completion": _sft_completion_text(str(row["approved_response"])),
                "metadata": {
                    id_column: row[id_column],
                    "match_id": row["match_id"],
                    "season": row["season"],
                    "snapshot_over": row["snapshot_over"],
                    "sft_bucket": row["sft_bucket"],
                    "review_priority": row["review_priority"],
                    "review_status": row.get("label_status"),
                    "response_source": response_source,
                },
            }
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _sft_prompt_text(prompt: str) -> str:
    return ensure_assistant_turn(prompt)


def _sft_completion_text(completion: str) -> str:
    return completion.lstrip()
