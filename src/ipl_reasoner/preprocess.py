"""Canonical cleaning, exclusions, and merged delivery construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ipl_reasoner.constants import STATS_SOURCE_SEASONS
from ipl_reasoner.paths import ProjectPaths

WICKET_KINDS_CONSUMING_BATTER = {
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
    "run out",
    "obstructing the field",
    "timed out",
    "handled the ball",
    "hit the ball twice",
}


@dataclass(frozen=True)
class CleaningArtifacts:
    matches_clean_path: Path
    deliveries_clean_path: Path
    merged_deliveries_path: Path
    exclusions_path: Path


def build_clean_datasets(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    paths: ProjectPaths,
) -> CleaningArtifacts:
    paths.ensure()

    exclusions = build_match_exclusion_report(matches, deliveries)
    excluded_match_ids = set(exclusions.loc[exclusions["is_excluded"], "match_id"].astype(str))

    matches_clean = matches.loc[~matches["id"].astype(str).isin(excluded_match_ids)].copy()
    deliveries_clean = deliveries.loc[
        ~deliveries["match_id"].astype(str).isin(excluded_match_ids)
    ].copy()

    merged_deliveries = build_merged_deliveries(matches_clean, deliveries_clean)

    matches_clean["is_stats_source_only"] = matches_clean["season"].isin(STATS_SOURCE_SEASONS)
    matches_clean["is_train_example_eligible"] = ~matches_clean["is_stats_source_only"]

    matches_clean_path = paths.processed / "matches_clean.csv"
    deliveries_clean_path = paths.processed / "deliveries_clean.csv"
    merged_deliveries_path = paths.processed / "merged_deliveries.csv"
    exclusions_path = paths.reports / "match_exclusions.csv"

    matches_clean.to_csv(matches_clean_path, index=False)
    deliveries_clean.to_csv(deliveries_clean_path, index=False)
    merged_deliveries.to_csv(merged_deliveries_path, index=False)
    exclusions.to_csv(exclusions_path, index=False)

    return CleaningArtifacts(
        matches_clean_path=matches_clean_path,
        deliveries_clean_path=deliveries_clean_path,
        merged_deliveries_path=merged_deliveries_path,
        exclusions_path=exclusions_path,
    )


def build_match_exclusion_report(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    matches_index = matches.copy()
    matches_index["match_id"] = matches_index["id"].astype(str)

    deliveries_index = deliveries.copy()
    deliveries_index["match_id"] = deliveries_index["match_id"].astype(str)

    innings_by_match = (
        deliveries_index.groupby("match_id")["inning"].agg(lambda s: {int(x) for x in s.dropna().astype(int)})
    )

    match_rows: list[dict[str, object]] = []
    for row in matches_index.itertuples(index=False):
        reason_codes: list[str] = []
        innings_present = innings_by_match.get(row.match_id, set())

        if int(getattr(row, "dl_applied", 0) or 0) == 1 or _nonnull_string(getattr(row, "method", pd.NA)):
            reason_codes.append("dls_or_method")

        if any(inning >= 3 for inning in innings_present):
            reason_codes.append("super_over")

        if str(getattr(row, "result", "")).strip().lower() == "no result" or not _nonnull_string(
            getattr(row, "winner", pd.NA)
        ):
            reason_codes.append("no_result_or_missing_winner")

        if 2 not in innings_present:
            reason_codes.append("missing_second_innings")
        elif str(getattr(row, "result", "")).strip().lower() == "normal":
            second_innings = deliveries_index[
                (deliveries_index["match_id"] == row.match_id) & (deliveries_index["inning"] == 2)
            ].copy()
            if not is_reconstructable_second_innings(second_innings, row):
                reason_codes.append("unreconstructable_second_innings")

        match_rows.append(
            {
                "match_id": row.match_id,
                "date": getattr(row, "date", pd.NaT),
                "season": getattr(row, "season", pd.NA),
                "team1": getattr(row, "team1", pd.NA),
                "team2": getattr(row, "team2", pd.NA),
                "winner": getattr(row, "winner", pd.NA),
                "result": getattr(row, "result", pd.NA),
                "method": getattr(row, "method", pd.NA),
                "is_excluded": bool(reason_codes),
                "exclusion_reasons": "|".join(reason_codes),
                "is_stats_source_only": getattr(row, "season", None) in STATS_SOURCE_SEASONS,
            }
        )

    report = pd.DataFrame(match_rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    return report


def build_merged_deliveries(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    merged = deliveries.merge(
        matches[["id", "date", "season", "venue"]].rename(columns={"id": "match_id"}),
        on="match_id",
        how="left",
        suffixes=("", "_match"),
    )
    merged["legal_ball"] = ((merged["wides"] == 0) & (merged["noballs"] == 0)).astype(int)
    merged["consumes_wicket"] = merged["dismissal_kind"].apply(_dismissal_consumes_wicket).astype(int)
    merged = merged.sort_values(["date", "match_id", "inning", "over", "ball"]).reset_index(drop=True)
    return merged


def is_reconstructable_second_innings(second_innings: pd.DataFrame, match_row: object) -> bool:
    if second_innings.empty:
        return False

    balls_per_over = int(getattr(match_row, "balls_per_over", 6) or 6)
    legal_balls = int(((second_innings["wides"] == 0) & (second_innings["noballs"] == 0)).sum())
    wickets_fallen = int(second_innings["dismissal_kind"].apply(_dismissal_consumes_wicket).sum())
    runs_scored = int(second_innings["total_runs"].fillna(0).sum())

    target = _infer_target_from_match_context(second_innings, match_row)
    overs_exhausted = legal_balls >= (20 * balls_per_over)
    reached_target = target is not None and runs_scored >= target
    all_out = wickets_fallen >= 10
    return bool(reached_target or all_out or overs_exhausted)


def _infer_target_from_match_context(second_innings: pd.DataFrame, match_row: object) -> int | None:
    if hasattr(match_row, "win_by_runs") and pd.notna(getattr(match_row, "win_by_runs", pd.NA)):
        try:
            win_by_runs = int(getattr(match_row, "win_by_runs", 0) or 0)
        except (TypeError, ValueError):
            win_by_runs = 0
    else:
        win_by_runs = 0

    if hasattr(match_row, "win_by_wickets") and pd.notna(getattr(match_row, "win_by_wickets", pd.NA)):
        try:
            win_by_wickets = int(getattr(match_row, "win_by_wickets", 0) or 0)
        except (TypeError, ValueError):
            win_by_wickets = 0
    else:
        win_by_wickets = 0

    second_innings_runs = int(second_innings["total_runs"].fillna(0).sum())

    if win_by_wickets > 0:
        return second_innings_runs
    if win_by_runs > 0:
        return second_innings_runs + win_by_runs + 1
    return None


def _dismissal_consumes_wicket(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return text in WICKET_KINDS_CONSUMING_BATTER


def _nonnull_string(value: object) -> bool:
    return bool(str(value).strip()) and not pd.isna(value)
