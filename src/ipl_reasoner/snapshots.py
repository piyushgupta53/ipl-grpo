"""Snapshot generation and canonical baseline training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ipl_reasoner.constants import BASELINE_FEATURE_ORDER, TEST_SEASONS, TRAIN_SEASONS, VALIDATION_SEASONS
from ipl_reasoner.paths import ProjectPaths
from ipl_reasoner.venue_data import apply_venue_codes

NEW_BATTER_PENDING = "NEW_BATTER_PENDING"
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
class SnapshotArtifacts:
    snapshot_dataset_path: Path
    baseline_model_path: Path
    baseline_metadata_path: Path


def build_snapshot_dataset_and_baseline(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    paths: ProjectPaths,
) -> SnapshotArtifacts:
    paths.ensure()
    matches_with_codes, deliveries_with_codes = apply_venue_codes(matches, deliveries)

    snapshots = _build_snapshots(matches_with_codes, deliveries_with_codes)
    model, metadata = _fit_baseline(snapshots)
    snapshots["run_rate_baseline_prob"] = score_baseline_dataframe(snapshots, model)
    snapshots["split"] = snapshots["season"].map(_season_to_split)

    snapshot_dataset_path = paths.processed / "snapshot_dataset.csv"
    baseline_model_path = paths.baseline_artifacts / "baseline_v1.joblib"
    baseline_metadata_path = paths.baseline_artifacts / "baseline_v1.metadata.json"

    snapshots.to_csv(snapshot_dataset_path, index=False)
    joblib.dump(model, baseline_model_path)
    baseline_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return SnapshotArtifacts(
        snapshot_dataset_path=snapshot_dataset_path,
        baseline_model_path=baseline_model_path,
        baseline_metadata_path=baseline_metadata_path,
    )


def score_baseline_dataframe(snapshots: pd.DataFrame, model: LogisticRegression) -> np.ndarray:
    x = snapshots.loc[:, list(BASELINE_FEATURE_ORDER)].to_numpy(dtype=float)
    return model.predict_proba(x)[:, 1]


def _build_snapshots(matches: pd.DataFrame, deliveries: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    deliveries = deliveries.copy()
    deliveries["date"] = pd.to_datetime(deliveries["date"], errors="coerce")
    deliveries["match_id"] = deliveries["match_id"].astype(str)
    deliveries = deliveries.sort_values(["match_id", "inning", "over", "ball"]).reset_index(drop=True)
    deliveries_by_match = {match_id: frame.copy() for match_id, frame in deliveries.groupby("match_id", sort=False)}

    for match in matches.itertuples(index=False):
        match_id = str(match.id)
        match_deliveries = deliveries_by_match.get(match_id)
        if match_deliveries is None:
            continue
        inning1 = match_deliveries.loc[match_deliveries["inning"] == 1].copy()
        inning2 = match_deliveries.loc[match_deliveries["inning"] == 2].copy()
        if inning1.empty or inning2.empty:
            continue

        target = int(inning1["total_runs"].sum()) + 1
        chasing_team = _derive_chasing_team(match)
        did_chasing_team_win = int(str(match.winner).strip() == chasing_team)

        next_over_lookup = _first_ball_of_next_over_lookup(inning2)

        for snapshot_over in range(1, 20):
            over_rows = inning2.loc[inning2["over"] == snapshot_over].copy()
            if over_rows.empty:
                break
            over_legal_balls = int(over_rows["legal_ball"].sum())
            if over_legal_balls < 6:
                break

            state = inning2.loc[inning2["over"] <= snapshot_over].copy()
            if state.empty:
                continue

            runs_scored = int(state["total_runs"].sum())
            wickets_fallen = int(state["consumes_wicket"].sum())
            balls_bowled = int(state["legal_ball"].sum())
            balls_remaining = 120 - balls_bowled
            required_runs = target - runs_scored

            if required_runs <= 0 or wickets_fallen >= 10 or balls_remaining <= 0:
                continue

            last_over_runs = int(over_rows["total_runs"].sum())
            recent_3 = inning2.loc[inning2["over"].between(max(1, snapshot_over - 2), snapshot_over)].copy()
            recent_5 = inning2.loc[inning2["over"].between(max(1, snapshot_over - 4), snapshot_over)].copy()
            last_3_overs_runs = int(recent_3["total_runs"].sum())
            last_3_overs_wickets = int(recent_3["consumes_wicket"].sum())
            recent_5_legal = int(recent_5["legal_ball"].sum())
            dot_ball_pct_last_5 = (
                float(((recent_5["legal_ball"] == 1) & (recent_5["total_runs"] == 0)).sum()) / recent_5_legal
                if recent_5_legal > 0
                else 0.0
            )
            boundary_pct_last_5 = (
                float(((recent_5["legal_ball"] == 1) & (recent_5["runs_off_bat"].isin([4, 6]))).sum()) / recent_5_legal
                if recent_5_legal > 0
                else 0.0
            )

            over_last_row = over_rows.iloc[-1]
            last_over_bowler = over_last_row["bowler"]
            last_over_bowler_overs_used = int(
                inning2.loc[
                    (inning2["over"] <= snapshot_over) & (inning2["bowler"] == last_over_bowler),
                    ["over"],
                ]
                .drop_duplicates()
                .shape[0]
            )

            last_wicket_over = _last_wicket_over(state)
            partnership_runs, partnership_balls = _partnership_metrics(state)
            batter_a, batter_b = _batters_at_over_break(
                over_rows=over_rows,
                next_over_row=next_over_lookup.get(snapshot_over + 1),
            )

            rows.append(
                {
                    "match_id": match_id,
                    "season": str(match.season),
                    "date": match.date,
                    "snapshot_over": snapshot_over,
                    "chasing_team": chasing_team,
                    "venue": match.venue,
                    "venue_code": match.venue_code,
                    "target": target,
                    "runs_scored": runs_scored,
                    "wickets_fallen": wickets_fallen,
                    "wickets_in_hand": 10 - wickets_fallen,
                    "balls_bowled": balls_bowled,
                    "balls_remaining": balls_remaining,
                    "required_runs": required_runs,
                    "current_run_rate": (runs_scored / balls_bowled) * 6 if balls_bowled > 0 else 0.0,
                    "required_run_rate": (required_runs / balls_remaining) * 6 if balls_remaining > 0 else 999.0,
                    "last_over_runs": last_over_runs,
                    "last_3_overs_runs": last_3_overs_runs,
                    "last_3_overs_wickets": last_3_overs_wickets,
                    "dot_ball_pct_last_5": dot_ball_pct_last_5,
                    "boundary_pct_last_5": boundary_pct_last_5,
                    "batter_a": batter_a,
                    "batter_b": batter_b,
                    "last_over_bowler": last_over_bowler,
                    "last_over_bowler_overs_used": last_over_bowler_overs_used,
                    "last_wicket_over": last_wicket_over,
                    "partnership_balls": partnership_balls,
                    "partnership_runs": partnership_runs,
                    "did_chasing_team_win": did_chasing_team_win,
                }
            )

    return pd.DataFrame(rows).sort_values(["date", "match_id", "snapshot_over"]).reset_index(drop=True)


def _fit_baseline(snapshots: pd.DataFrame) -> tuple[LogisticRegression, dict[str, object]]:
    train = snapshots.loc[snapshots["season"].isin(TRAIN_SEASONS)].copy()
    x_train = train.loc[:, list(BASELINE_FEATURE_ORDER)].to_numpy(dtype=float)
    y_train = train["did_chasing_team_win"].to_numpy(dtype=int)

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    metadata = {
        "baseline_version": "baseline_v1",
        "trained_on_seasons": list(TRAIN_SEASONS),
        "validation_seasons": list(VALIDATION_SEASONS),
        "test_seasons": list(TEST_SEASONS),
        "feature_order": list(BASELINE_FEATURE_ORDER),
        "coefficients": model.coef_[0].tolist(),
        "intercept": model.intercept_[0].item(),
        "model_type": "sklearn.linear_model.LogisticRegression",
    }
    return model, metadata


def _derive_chasing_team(match: object) -> str:
    toss_decision = str(match.toss_decision).strip().lower()
    toss_winner = str(match.toss_winner).strip()
    team1 = str(match.team1).strip()
    team2 = str(match.team2).strip()
    if toss_decision == "field":
        return toss_winner
    if toss_winner == team1:
        return team2
    if toss_winner == team2:
        return team1
    return team2


def _first_ball_of_next_over_lookup(inning2: pd.DataFrame) -> dict[int, pd.Series]:
    lookup: dict[int, pd.Series] = {}
    for over, group in inning2.groupby("over"):
        lookup[int(over)] = group.sort_values(["ball"]).iloc[0]
    return lookup


def _last_wicket_over(state: pd.DataFrame) -> int:
    wickets = state.loc[state["consumes_wicket"] == 1]
    if wickets.empty:
        return 0
    return int(wickets["over"].max())


def _partnership_metrics(state: pd.DataFrame) -> tuple[int, int]:
    wicket_rows = state.loc[state["consumes_wicket"] == 1]
    if wicket_rows.empty:
        return int(state["total_runs"].sum()), int(state["legal_ball"].sum())
    last_wicket_index = wicket_rows.index.max()
    after_wicket = state.loc[state.index > last_wicket_index]
    return int(after_wicket["total_runs"].sum()), int(after_wicket["legal_ball"].sum())


def _batters_at_over_break(over_rows: pd.DataFrame, next_over_row: pd.Series | None) -> tuple[str, str]:
    last_row = over_rows.sort_values(["ball"]).iloc[-1]
    dismissal_kind = str(last_row["dismissal_kind"]).strip().lower()
    dismissal_consumes = dismissal_kind in WICKET_KINDS_CONSUMING_BATTER
    if dismissal_consumes:
        dismissed = str(last_row["player_dismissed"]).strip()
        batters = [str(last_row["striker"]).strip(), str(last_row["non_striker"]).strip()]
        survivors = [b for b in batters if b and b != dismissed]
        if next_over_row is not None:
            next_batters = [str(next_over_row["striker"]).strip(), str(next_over_row["non_striker"]).strip()]
            if dismissed in next_batters:
                pass
        if len(survivors) == 1:
            return survivors[0], NEW_BATTER_PENDING
        if len(survivors) >= 2:
            return survivors[0], survivors[1]
        return NEW_BATTER_PENDING, NEW_BATTER_PENDING

    if next_over_row is not None:
        return str(next_over_row["striker"]).strip(), str(next_over_row["non_striker"]).strip()

    return str(last_row["striker"]).strip(), str(last_row["non_striker"]).strip()


def _season_to_split(season: str) -> str:
    if season in TRAIN_SEASONS:
        return "train"
    if season in VALIDATION_SEASONS:
        return "validation"
    if season in TEST_SEASONS:
        return "test"
    return "other"
