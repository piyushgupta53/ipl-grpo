"""Final prompt-bearing training dataset assembly."""

from __future__ import annotations

import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ipl_reasoner.constants import TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS, VENUE_SIMILAR_TO
from ipl_reasoner.paths import ProjectPaths

PROMPT_VERSION = "prompt_v3"
EVENING_MATCH_TYPE = "Evening (7:30 PM IST, dew likely)"


@dataclass(frozen=True)
class TrainingDatasetArtifacts:
    training_dataset_path: Path


def build_training_dataset(
    snapshots: pd.DataFrame,
    matches: pd.DataFrame,
    player_stats: pd.DataFrame,
    league_avgs: pd.DataFrame,
    venue_stats: pd.DataFrame,
    venue_metadata: pd.DataFrame,
    paths: ProjectPaths,
) -> TrainingDatasetArtifacts:
    paths.ensure()

    snapshots = snapshots.loc[snapshots["split"].isin(["train", "validation", "test"])].copy()
    matches = matches.copy()
    player_lookup = _build_player_lookup(player_stats)
    league_lookup = league_avgs.set_index("as_of_season").to_dict(orient="index")
    league_lookup = {_season_key(key): value for key, value in league_lookup.items()}
    venue_stats_lookup = {
        (str(row["venue_code"]), _season_key(row["as_of_season"])): row
        for row in venue_stats.to_dict("records")
    }
    venue_meta_lookup = venue_metadata.set_index("venue_code").to_dict(orient="index")
    global_venue_lookup = _build_global_venue_lookup(venue_stats)

    match_cols = [
        "id",
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "venue",
    ]
    merged = snapshots.merge(
        matches.loc[:, match_cols].rename(columns={"id": "match_id"}),
        on=["match_id"],
        how="left",
        suffixes=("", "_match"),
    )

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        season = _season_key(row.season)
        bowling_team = _other_team(row.chasing_team, row.team1, row.team2)

        batter_a_stats = _get_player_stats(row.batter_a, season, player_lookup, league_lookup, role="batting")
        batter_b_stats = None
        if row.batter_b != "NEW_BATTER_PENDING":
            batter_b_stats = _get_player_stats(row.batter_b, season, player_lookup, league_lookup, role="batting")
        bowler_stats = _get_player_stats(
            row.last_over_bowler,
            season,
            player_lookup,
            league_lookup,
            role="bowling",
        )
        known_bowler_state = _load_known_bowler_state(getattr(row, "known_bowler_state_json", ""))
        bowling_resources = _build_known_bowling_resources(
            known_bowler_state=known_bowler_state,
            season=season,
            player_lookup=player_lookup,
            league_lookup=league_lookup,
        )

        venue_block = _build_venue_context_block(
            venue_code=row.venue_code,
            as_of_season=season,
            venue_stats_lookup=venue_stats_lookup,
            venue_meta_lookup=venue_meta_lookup,
            global_venue_lookup=global_venue_lookup,
        )

        batter_a_display = _format_batter_line(row.batter_a, batter_a_stats)
        batter_b_display = (
            "NEW_BATTER_PENDING"
            if row.batter_b == "NEW_BATTER_PENDING"
            else _format_batter_line(row.batter_b, batter_b_stats)
        )
        bowler_suffix = " (career data limited)" if bowler_stats["is_estimated"] else ""

        prompt = (
            "SYSTEM:\n"
            "You are an IPL cricket analyst. Estimate the probability (0.0 to 1.0) "
            "that the batting team wins the match.\n"
            "Write a short public reasoning summary focused only on the most important signals.\n"
            "Do not reveal hidden chain-of-thought.\n\n"
            "Format exactly as:\n"
            "<analysis>\n"
            "[2-5 sentence public reasoning summary here]\n"
            "</analysis>\n"
            "<answer>0.XX</answer>\n\n"
            "Answer must be a decimal between 0.0 and 1.0.\n\n"
            "USER:\n"
            f"IPL Match - {row.chasing_team} chasing {int(row.target)} set by {bowling_team}\n"
            f"Venue: {row.venue}\n"
            f"Match type: {EVENING_MATCH_TYPE}\n"
            f"Toss: {row.toss_winner} chose to {row.toss_decision} first\n\n"
            f"End of Over {int(row.snapshot_over)}:\n"
            f"Score: {int(row.runs_scored)}/{int(row.wickets_fallen)} ({int(row.balls_bowled)} balls faced)\n"
            f"Need: {int(row.required_runs)} off {int(row.balls_remaining)} (RRR: {float(row.required_run_rate):.2f})\n"
            f"Current run rate: {float(row.current_run_rate):.2f}\n\n"
            "Momentum:\n"
            f"Last over: {int(row.last_over_runs)}\n"
            f"Last 3 overs: {int(row.last_3_overs_runs)} runs, {int(row.last_3_overs_wickets)} wickets\n"
            f"Last 5 overs: Dot% {float(row.dot_ball_pct_last_5):.0%} | Boundary% {float(row.boundary_pct_last_5):.0%}\n\n"
            "Batters:\n"
            f"{batter_a_display}\n"
            f"{batter_b_display}\n"
            f"Partnership: {int(row.partnership_balls)}b, {int(row.partnership_runs)}r\n\n"
            f"Last over bowler: {row.last_over_bowler} ({int(row.last_over_bowler_overs_used)}/4 overs used, "
            f"death econ {float(bowler_stats['death_economy']):.2f}, death wkts/ov {float(bowler_stats['death_wickets_per_over']):.2f}){bowler_suffix}\n"
            "Known bowling resources (among bowlers already used):\n"
            f"{_format_known_bowling_resources_block(bowling_resources)}\n"
            "Next over: different bowler.\n\n"
            f"Venue context ({row.venue}):\n"
            f"{venue_block}\n\n"
            f"Win probability for {row.chasing_team}?"
        )

        is_any_player_estimated = bool(
            batter_a_stats["is_estimated"]
            or (batter_b_stats["is_estimated"] if batter_b_stats is not None else False)
            or bowler_stats["is_estimated"]
        )

        rows.append(
            {
                "match_id": row.match_id,
                "season": season,
                "snapshot_over": int(row.snapshot_over),
                "chasing_team": row.chasing_team,
                "venue_code": row.venue_code,
                "prompt": prompt,
                "did_chasing_team_win": int(row.did_chasing_team_win),
                "run_rate_baseline_prob": float(row.run_rate_baseline_prob),
                "is_any_player_estimated": is_any_player_estimated,
                "required_run_rate": float(row.required_run_rate),
                "wickets_in_hand": int(row.wickets_in_hand),
                "partnership_runs": int(row.partnership_runs),
                "is_day_match": False,
                "toss_decision": row.toss_decision,
                "batter_a": row.batter_a,
                "batter_b": row.batter_b,
                "last_over_bowler": row.last_over_bowler,
                "known_bowlers_used_count": int(bowling_resources["known_bowlers_used_count"]),
                "known_bowling_overs_left": int(bowling_resources["known_bowling_overs_left"]),
                "known_best_death_bowler": bowling_resources["best_bowler_name"],
                "known_best_death_bowler_remaining_overs": int(bowling_resources["best_bowler_remaining_overs"]),
                "known_best_death_bowler_death_economy": float(bowling_resources["best_bowler_death_economy"]),
                "known_best_death_bowler_death_wickets_per_over": float(
                    bowling_resources["best_bowler_death_wickets_per_over"]
                ),
                "known_death_overs_left_top2": int(bowling_resources["top2_remaining_overs"]),
                "known_death_resource_score": float(bowling_resources["known_death_resource_score"]),
                "split": row.split,
                "prompt_version": PROMPT_VERSION,
            }
        )

    df = pd.DataFrame(rows)
    train_df = _build_training_order(df.loc[df["split"] == "train"].copy(), cooldown=8, seed=42)
    validation_df = df.loc[df["split"] == "validation"].copy().sort_values(["season", "match_id", "snapshot_over"])
    test_df = df.loc[df["split"] == "test"].copy().sort_values(["season", "match_id", "snapshot_over"])
    final_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)

    output_path = paths.processed / "training_dataset.csv"
    final_df.to_csv(output_path, index=False)
    return TrainingDatasetArtifacts(training_dataset_path=output_path)


def _build_player_lookup(player_stats: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in player_stats.to_dict("records"):
        lookup[(row["player"], str(row["as_of_season"]))] = row
    return lookup


def _get_player_stats(
    player_name: str,
    season: str,
    player_lookup: dict[tuple[str, str], dict[str, object]],
    league_lookup: dict[str, dict[str, object]],
    role: str,
) -> dict[str, object]:
    row = player_lookup.get((player_name, season))
    league = league_lookup.get(season, {})

    if role == "batting":
        if row is None:
            return {
                "chase_sr": float(league.get("batting_chase_sr_avg", np.nan)),
                "death_sr": float(league.get("batting_death_sr_avg", np.nan)),
                "is_estimated": True,
            }
        return {
            "chase_sr": float(row["chase_sr"]),
            "death_sr": float(row["death_sr"]),
            "is_estimated": bool(_coerce_bool(row["batting_is_estimated"])),
        }

    if row is None:
        return {
            "death_economy": float(league.get("bowling_death_economy_avg", np.nan)),
            "death_wickets_per_over": float(league.get("bowling_death_wickets_per_over_avg", np.nan)),
            "is_estimated": True,
        }
    return {
        "death_economy": float(row["death_economy"]),
        "death_wickets_per_over": float(row["death_wickets_per_over"]),
        "is_estimated": bool(_coerce_bool(row["bowling_is_estimated"])),
    }


def _format_batter_line(player_name: str, stats: dict[str, object] | None) -> str:
    assert stats is not None
    suffix = " (career data limited)" if stats["is_estimated"] else ""
    return f"{player_name} (chase SR {float(stats['chase_sr']):.0f}, death SR {float(stats['death_sr']):.0f}){suffix}"


def _load_known_bowler_state(value: object) -> list[dict[str, object]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    normalized: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        bowler = str(item.get("bowler", "")).strip()
        if not bowler:
            continue
        overs_used = int(item.get("overs_used", 0) or 0)
        remaining_overs = int(item.get("remaining_overs", max(0, 4 - overs_used)) or 0)
        normalized.append(
            {
                "bowler": bowler,
                "overs_used": overs_used,
                "remaining_overs": max(0, remaining_overs),
            }
        )
    return normalized


def _build_known_bowling_resources(
    known_bowler_state: list[dict[str, object]],
    season: str,
    player_lookup: dict[tuple[str, str], dict[str, object]],
    league_lookup: dict[str, dict[str, object]],
) -> dict[str, object]:
    league = league_lookup.get(season, {})
    league_death_economy = float(league.get("bowling_death_economy_avg", 10.0) or 10.0)
    league_death_wpo = float(league.get("bowling_death_wickets_per_over_avg", 0.30) or 0.30)

    resources: list[dict[str, object]] = []
    for item in known_bowler_state:
        remaining_overs = int(item["remaining_overs"])
        if remaining_overs <= 0:
            continue

        stats = _get_player_stats(
            str(item["bowler"]),
            season,
            player_lookup,
            league_lookup,
            role="bowling",
        )
        death_economy = float(stats["death_economy"])
        death_wpo = float(stats["death_wickets_per_over"])
        quality_score = (
            (league_death_economy - death_economy)
            + ((death_wpo - league_death_wpo) * 10.0)
            + (remaining_overs * 0.15)
        )

        resources.append(
            {
                "bowler": str(item["bowler"]),
                "overs_used": int(item["overs_used"]),
                "remaining_overs": remaining_overs,
                "death_economy": death_economy,
                "death_wickets_per_over": death_wpo,
                "is_estimated": bool(stats["is_estimated"]),
                "quality_score": quality_score,
            }
        )

    resources.sort(
        key=lambda item: (
            -float(item["quality_score"]),
            -int(item["remaining_overs"]),
            str(item["bowler"]),
        )
    )

    top2_remaining_overs = sum(int(item["remaining_overs"]) for item in resources[:2])
    known_death_resource_score = sum(
        max(-3.0, min(3.0, float(item["quality_score"]))) * int(item["remaining_overs"])
        for item in resources
    )

    if resources:
        best = resources[0]
        best_name = str(best["bowler"])
        best_remaining_overs = int(best["remaining_overs"])
        best_death_economy = float(best["death_economy"])
        best_death_wpo = float(best["death_wickets_per_over"])
    else:
        best_name = ""
        best_remaining_overs = 0
        best_death_economy = league_death_economy
        best_death_wpo = league_death_wpo

    return {
        "resources": resources,
        "known_bowlers_used_count": len(known_bowler_state),
        "known_bowling_overs_left": sum(int(item["remaining_overs"]) for item in resources),
        "best_bowler_name": best_name,
        "best_bowler_remaining_overs": best_remaining_overs,
        "best_bowler_death_economy": best_death_economy,
        "best_bowler_death_wickets_per_over": best_death_wpo,
        "top2_remaining_overs": top2_remaining_overs,
        "known_death_resource_score": known_death_resource_score,
    }


def _format_known_bowling_resources_block(summary: dict[str, object]) -> str:
    resources = list(summary["resources"])
    if not resources:
        return "No used bowlers with overs left yet.\nKnown overs left among used bowlers: 0"

    lines: list[str] = []
    for resource in resources[:3]:
        overs_left = int(resource["remaining_overs"])
        overs_label = "over" if overs_left == 1 else "overs"
        suffix = " (career data limited)" if bool(resource["is_estimated"]) else ""
        lines.append(
            f"{resource['bowler']}: {overs_left} {overs_label} left "
            f"(death econ {float(resource['death_economy']):.2f}, death wkts/ov {float(resource['death_wickets_per_over']):.2f}){suffix}"
        )

    if len(resources) > 3:
        extra_count = len(resources) - 3
        extra_label = "bowler" if extra_count == 1 else "bowlers"
        lines.append(f"Other used {extra_label} with overs left: {extra_count}")

    lines.append(f"Known overs left among used bowlers: {int(summary['known_bowling_overs_left'])}")
    return "\n".join(lines)


def _build_venue_context_block(
    venue_code: str,
    as_of_season: str,
    venue_stats_lookup: dict[tuple[str, str], dict[str, object]],
    venue_meta_lookup: dict[str, dict[str, object]],
    global_venue_lookup: dict[str, dict[str, float]],
) -> str:
    stats = _get_venue_stats_blended(venue_code, as_of_season, venue_stats_lookup, global_venue_lookup)
    meta = venue_meta_lookup[venue_code]
    dew_factor = str(meta["dew_factor"]).capitalize()
    pitch_label = _pitch_type_label(str(meta["pitch_type"]))
    matches_count = int(stats["matches_count"])
    alpha = float(stats["confidence_alpha"])

    if alpha >= 0.75:
        return "\n".join(
            [
                f"Surface: {pitch_label}, {meta['boundary_size']} boundaries",
                f"Chase rate: {float(stats['chase_success_rate']):.0%} ({matches_count} matches)",
                f"Avg 2nd inns: {float(stats['avg_second_innings_score']):.0f}",
                f"Death RPO: {float(stats['avg_death_rpo']):.1f}",
                f"Dew: {dew_factor} evenings",
            ]
        )

    return "\n".join(
        [
            f"Limited IPL history ({matches_count} matches); venue stats are estimated.",
            f"Surface prior: {pitch_label}, {meta['boundary_size']} boundaries",
            f"Chase rate: ~{float(stats['chase_success_rate']):.0%}",
            f"Avg 2nd inns: ~{float(stats['avg_second_innings_score']):.0f}",
            f"Death RPO: ~{float(stats['avg_death_rpo']):.1f}",
            f"Dew: {dew_factor} evenings",
        ]
    )


def _get_venue_stats_blended(
    venue_code: str,
    as_of_season: str,
    venue_stats_lookup: dict[tuple[str, str], dict[str, object]],
    global_venue_lookup: dict[str, dict[str, float]],
) -> dict[str, object]:
    row = dict(venue_stats_lookup[(venue_code, as_of_season)])
    alpha = float(row["confidence_alpha"])
    if alpha >= 1.0:
        return _fill_missing_venue_values(row, global_venue_lookup.get(as_of_season, {}))

    similar_code = VENUE_SIMILAR_TO.get(venue_code)
    if not similar_code:
        return _fill_missing_venue_values(row, global_venue_lookup.get(as_of_season, {}))

    similar_row = venue_stats_lookup.get((similar_code, as_of_season))
    if similar_row is None:
        return _fill_missing_venue_values(row, global_venue_lookup.get(as_of_season, {}))

    blended = dict(row)
    for col in [
        "avg_first_innings_score",
        "avg_second_innings_score",
        "chase_success_rate",
        "avg_death_rpo",
        "avg_par_score_at_over_10",
    ]:
        own_val = float(row[col]) if pd.notna(row[col]) else np.nan
        similar_val = float(similar_row[col]) if pd.notna(similar_row[col]) else np.nan
        if np.isnan(own_val):
            blended[col] = similar_val
        elif np.isnan(similar_val):
            blended[col] = own_val
        else:
            blended[col] = (alpha * own_val) + ((1 - alpha) * similar_val)
    return _fill_missing_venue_values(blended, global_venue_lookup.get(as_of_season, {}))


def _build_global_venue_lookup(venue_stats: pd.DataFrame) -> dict[str, dict[str, float]]:
    value_cols = [
        "avg_first_innings_score",
        "avg_second_innings_score",
        "chase_success_rate",
        "avg_death_rpo",
        "avg_par_score_at_over_10",
    ]
    lookup: dict[str, dict[str, float]] = {}
    for season, group in venue_stats.loc[venue_stats["matches_count"] > 0].groupby("as_of_season"):
        lookup[_season_key(season)] = {col: float(group[col].mean()) for col in value_cols}
    return lookup


def _fill_missing_venue_values(row: dict[str, object], fallback: dict[str, float]) -> dict[str, object]:
    for col, value in fallback.items():
        current = row.get(col)
        if pd.isna(current):
            row[col] = value
    return row


def _pitch_type_label(value: str) -> str:
    mapping = {
        "batting": "batting-friendly",
        "spin": "spin-friendly",
        "pace": "pace-friendly",
        "balanced": "balanced",
    }
    return mapping.get(value, value)


def _other_team(chasing_team: str, team1: str, team2: str) -> str:
    if chasing_team == team1:
        return team2
    return team1


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _season_key(value: object) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()


def _build_training_order(df: pd.DataFrame, cooldown: int, seed: int) -> pd.DataFrame:
    best_df: pd.DataFrame | None = None
    best_violations: int | None = None

    for attempt_seed in range(seed, seed + 25):
        candidate = _build_training_order_once(df, cooldown=cooldown, seed=attempt_seed)
        violations = _count_cooldown_violations(candidate, cooldown=cooldown)
        if best_violations is None or violations < best_violations:
            best_df = candidate
            best_violations = violations
        if violations == 0:
            break

    assert best_df is not None
    return best_df


def _build_training_order_once(df: pd.DataFrame, cooldown: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in df.sample(frac=1.0, random_state=seed).to_dict("records"):
        groups[str(row["match_id"])].append(row)

    recent: deque[str] = deque(maxlen=cooldown)
    ordered: list[dict[str, object]] = []

    while groups:
        eligible = [mid for mid in groups if mid not in recent]
        if not eligible:
            eligible = list(groups.keys())

        eligible.sort(key=lambda mid: (-len(groups[mid]), mid))
        top_k = eligible[: min(5, len(eligible))]
        match_id = rng.choice(top_k)
        ordered.append(groups[match_id].pop())
        recent.append(match_id)

        if not groups[match_id]:
            del groups[match_id]

    return pd.DataFrame(ordered)


def _count_cooldown_violations(df: pd.DataFrame, cooldown: int) -> int:
    match_ids = df["match_id"].astype(str).tolist()
    violations = 0
    for idx, match_id in enumerate(match_ids):
        if match_id in match_ids[max(0, idx - cooldown) : idx]:
            violations += 1
    return violations
