"""Build pre-season player prior tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ipl_reasoner.constants import AS_OF_SEASONS
from ipl_reasoner.paths import ProjectPaths


@dataclass(frozen=True)
class PlayerStatsArtifacts:
    player_stats_path: Path
    league_averages_path: Path


def build_player_season_stats(merged_deliveries: pd.DataFrame, paths: ProjectPaths) -> PlayerStatsArtifacts:
    paths.ensure()
    merged = merged_deliveries.copy()
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["is_boundary"] = merged["runs_off_bat"].isin([4, 6]).astype(int)
    merged["is_dot_legal"] = ((merged["legal_ball"] == 1) & (merged["total_runs"] == 0)).astype(int)

    player_tables: list[pd.DataFrame] = []
    league_tables: list[dict[str, object]] = []

    for season in AS_OF_SEASONS:
        season_year = int(season)
        cutoff = pd.Timestamp(f"{season_year}-04-01")
        history = merged.loc[merged["date"] < cutoff].copy()

        batting = _build_batting_stats(history, season_year)
        bowling = _build_bowling_stats(history, season_year)
        league_row = _build_league_averages(season, batting, bowling)
        combined = _combine_and_fill_player_stats(season, batting, bowling, league_row)

        player_tables.append(combined)
        league_tables.append(league_row)

    players = pd.concat(player_tables, ignore_index=True).sort_values(["as_of_season", "player"])
    league = pd.DataFrame(league_tables).sort_values(["as_of_season"]).reset_index(drop=True)

    player_stats_path = paths.artifacts / "metadata" / "player_season_stats.csv"
    league_averages_path = paths.artifacts / "metadata" / "league_avg_by_season.csv"
    players.to_csv(player_stats_path, index=False)
    league.to_csv(league_averages_path, index=False)

    return PlayerStatsArtifacts(
        player_stats_path=player_stats_path,
        league_averages_path=league_averages_path,
    )


def _build_batting_stats(history: pd.DataFrame, season_year: int) -> pd.DataFrame:
    chase = history.loc[history["inning"] == 2].copy()
    if chase.empty:
        return pd.DataFrame(columns=_batting_columns())

    chase["weight"] = _ewma_weight(chase["date"], season_year)
    legal = chase.loc[chase["legal_ball"] == 1].copy()
    if legal.empty:
        return pd.DataFrame(columns=_batting_columns())

    innings_count = (
        chase.groupby("striker")["match_id"].nunique().rename("innings_count").reset_index()
    )

    chase_stats = _weighted_group_metrics(
        legal,
        group_col="striker",
        prefix="",
        runs_col="runs_off_bat",
        denom_col="legal_ball",
        boundary_col="is_boundary",
        dot_col="is_dot_legal",
    ).rename(
        columns={
            "strike_rate": "chase_sr",
            "boundary_pct": "chase_boundary_pct",
            "dot_pct": "chase_dot_pct",
        }
    )

    death = legal.loc[legal["over"] >= 17].copy()
    if death.empty:
        death_stats = pd.DataFrame(columns=["striker", "death_sr", "death_boundary_pct"])
    else:
        death_stats = _weighted_group_metrics(
            death,
            group_col="striker",
            prefix="",
            runs_col="runs_off_bat",
            denom_col="legal_ball",
            boundary_col="is_boundary",
            dot_col="is_dot_legal",
        ).rename(
            columns={
                "strike_rate": "death_sr",
                "boundary_pct": "death_boundary_pct",
            }
        )[["striker", "death_sr", "death_boundary_pct"]]

    recent = legal.loc[legal["date"].dt.year == (season_year - 1)].copy()
    if recent.empty:
        recent_stats = pd.DataFrame(columns=["striker", "recent_form_sr"])
    else:
        grouped = recent.groupby("striker", as_index=False).agg(
            runs=("runs_off_bat", "sum"),
            balls=("legal_ball", "sum"),
        )
        grouped["recent_form_sr"] = np.where(grouped["balls"] > 0, grouped["runs"] / grouped["balls"] * 100, np.nan)
        recent_stats = grouped[["striker", "recent_form_sr"]]

    batting = innings_count.merge(chase_stats, on="striker", how="left")
    batting = batting.merge(death_stats, on="striker", how="left")
    batting = batting.merge(recent_stats, on="striker", how="left")
    batting = batting.rename(columns={"striker": "player"})
    return batting


def _build_bowling_stats(history: pd.DataFrame, season_year: int) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=_bowling_columns())

    history = history.copy()
    history["weight"] = _ewma_weight(history["date"], season_year)
    legal = history.loc[history["legal_ball"] == 1].copy()
    if legal.empty:
        return pd.DataFrame(columns=_bowling_columns())

    overs_bowled = (
        history[["bowler", "match_id", "inning", "over"]]
        .drop_duplicates()
        .groupby("bowler")
        .size()
        .rename("overs_bowled")
        .reset_index()
    )

    overall = _weighted_bowler_metrics(legal).rename(columns={"economy": "overall_economy"})
    death = _weighted_bowler_metrics(legal.loc[legal["over"] >= 17].copy()).rename(
        columns={
            "economy": "death_economy",
            "wickets_per_over": "death_wickets_per_over",
            "dot_pct": "death_dot_pct",
        }
    )

    bowling = overs_bowled.merge(overall, on="bowler", how="left")
    bowling = bowling.merge(
        death[["bowler", "death_economy", "death_wickets_per_over", "death_dot_pct"]],
        on="bowler",
        how="left",
    )
    bowling = bowling.rename(columns={"bowler": "player"})
    return bowling


def _build_league_averages(season: str, batting: pd.DataFrame, bowling: pd.DataFrame) -> dict[str, object]:
    eligible_batting = batting.loc[batting["innings_count"] >= 30]
    eligible_bowling = bowling.loc[bowling["overs_bowled"] >= 20]

    row: dict[str, object] = {"as_of_season": season}
    batting_metric_cols = [
        "chase_sr",
        "death_sr",
        "chase_boundary_pct",
        "chase_dot_pct",
        "death_boundary_pct",
        "recent_form_sr",
    ]
    bowling_metric_cols = [
        "death_economy",
        "death_wickets_per_over",
        "overall_economy",
        "death_dot_pct",
    ]

    for col in batting_metric_cols:
        row[f"batting_{col}_avg"] = eligible_batting[col].mean() if col in eligible_batting else np.nan
    for col in bowling_metric_cols:
        row[f"bowling_{col}_avg"] = eligible_bowling[col].mean() if col in eligible_bowling else np.nan

    row["eligible_batters"] = len(eligible_batting)
    row["eligible_bowlers"] = len(eligible_bowling)
    return row


def _combine_and_fill_player_stats(
    season: str,
    batting: pd.DataFrame,
    bowling: pd.DataFrame,
    league_row: dict[str, object],
) -> pd.DataFrame:
    players = pd.DataFrame({"player": sorted(set(batting["player"]).union(set(bowling["player"])))})
    combined = players.merge(batting, on="player", how="left").merge(bowling, on="player", how="left")
    combined["as_of_season"] = season

    combined["innings_count"] = combined["innings_count"].fillna(0).astype(int)
    combined["overs_bowled"] = combined["overs_bowled"].fillna(0).astype(int)
    combined["batting_is_estimated"] = combined["innings_count"] < 30
    combined["bowling_is_estimated"] = combined["overs_bowled"] < 20

    batting_fill_map = {
        "chase_sr": "batting_chase_sr_avg",
        "death_sr": "batting_death_sr_avg",
        "chase_boundary_pct": "batting_chase_boundary_pct_avg",
        "chase_dot_pct": "batting_chase_dot_pct_avg",
        "death_boundary_pct": "batting_death_boundary_pct_avg",
        "recent_form_sr": "batting_recent_form_sr_avg",
    }
    bowling_fill_map = {
        "death_economy": "bowling_death_economy_avg",
        "death_wickets_per_over": "bowling_death_wickets_per_over_avg",
        "overall_economy": "bowling_overall_economy_avg",
        "death_dot_pct": "bowling_death_dot_pct_avg",
    }

    for col, league_key in batting_fill_map.items():
        combined[col] = combined[col].astype(float)
        combined.loc[combined["batting_is_estimated"], col] = league_row.get(league_key, np.nan)
        combined[col] = combined[col].fillna(league_row.get(league_key, np.nan))

    for col, league_key in bowling_fill_map.items():
        combined[col] = combined[col].astype(float)
        combined.loc[combined["bowling_is_estimated"], col] = league_row.get(league_key, np.nan)
        combined[col] = combined[col].fillna(league_row.get(league_key, np.nan))

    return combined[
        [
            "player",
            "as_of_season",
            "innings_count",
            "chase_sr",
            "death_sr",
            "chase_boundary_pct",
            "chase_dot_pct",
            "death_boundary_pct",
            "recent_form_sr",
            "batting_is_estimated",
            "overs_bowled",
            "death_economy",
            "death_wickets_per_over",
            "overall_economy",
            "death_dot_pct",
            "bowling_is_estimated",
        ]
    ]


def _weighted_group_metrics(
    frame: pd.DataFrame,
    group_col: str,
    prefix: str,
    runs_col: str,
    denom_col: str,
    boundary_col: str,
    dot_col: str,
) -> pd.DataFrame:
    grouped = frame.groupby(group_col, as_index=False).apply(
        lambda g: pd.Series(
            {
                "weighted_runs": (g[runs_col] * g["weight"]).sum(),
                "weighted_balls": (g[denom_col] * g["weight"]).sum(),
                "weighted_boundaries": (g[boundary_col] * g["weight"]).sum(),
                "weighted_dots": (g[dot_col] * g["weight"]).sum(),
            }
        ),
        include_groups=False,
    )
    grouped["strike_rate"] = np.where(
        grouped["weighted_balls"] > 0,
        grouped["weighted_runs"] / grouped["weighted_balls"] * 100,
        np.nan,
    )
    grouped["boundary_pct"] = np.where(
        grouped["weighted_balls"] > 0,
        grouped["weighted_boundaries"] / grouped["weighted_balls"],
        np.nan,
    )
    grouped["dot_pct"] = np.where(
        grouped["weighted_balls"] > 0,
        grouped["weighted_dots"] / grouped["weighted_balls"],
        np.nan,
    )
    return grouped[[group_col, "strike_rate", "boundary_pct", "dot_pct"]]


def _weighted_bowler_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["bowler", "economy", "wickets_per_over", "dot_pct"])

    grouped = frame.groupby("bowler", as_index=False).apply(
        lambda g: pd.Series(
            {
                "weighted_runs_conceded": (g["total_runs"] * g["weight"]).sum(),
                "weighted_legal_balls": (g["legal_ball"] * g["weight"]).sum(),
                "weighted_wickets": (g["consumes_wicket"] * g["weight"]).sum(),
                "weighted_dots": (g["is_dot_legal"] * g["weight"]).sum(),
            }
        ),
        include_groups=False,
    )
    grouped["economy"] = np.where(
        grouped["weighted_legal_balls"] > 0,
        grouped["weighted_runs_conceded"] / grouped["weighted_legal_balls"] * 6,
        np.nan,
    )
    grouped["wickets_per_over"] = np.where(
        grouped["weighted_legal_balls"] > 0,
        grouped["weighted_wickets"] / (grouped["weighted_legal_balls"] / 6),
        np.nan,
    )
    grouped["dot_pct"] = np.where(
        grouped["weighted_legal_balls"] > 0,
        grouped["weighted_dots"] / grouped["weighted_legal_balls"],
        np.nan,
    )
    return grouped[["bowler", "economy", "wickets_per_over", "dot_pct"]]


def _ewma_weight(dates: pd.Series, season_year: int) -> pd.Series:
    reference = pd.Timestamp(f"{season_year}-04-01")
    days_ago = (reference - pd.to_datetime(dates, errors="coerce")).dt.days
    return np.exp(-0.003 * days_ago)


def _batting_columns() -> list[str]:
    return [
        "player",
        "innings_count",
        "chase_sr",
        "death_sr",
        "chase_boundary_pct",
        "chase_dot_pct",
        "death_boundary_pct",
        "recent_form_sr",
    ]


def _bowling_columns() -> list[str]:
    return [
        "player",
        "overs_bowled",
        "death_economy",
        "death_wickets_per_over",
        "overall_economy",
        "death_dot_pct",
    ]
