"""Raw data loading, schema detection, and canonical normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from ipl_reasoner.constants import RAW_DELIVERIES_FILENAME, RAW_MATCHES_FILENAME, TEAM_NAME_CANONICAL
from ipl_reasoner.paths import ProjectPaths


@dataclass(frozen=True)
class SchemaReport:
    dataset_name: str
    row_count: int
    canonical_columns: tuple[str, ...]
    alias_columns_used: dict[str, str]
    missing_required_columns: tuple[str, ...]
    missing_optional_columns: tuple[str, ...]
    over_index_base: int | None = None

    @property
    def is_valid(self) -> bool:
        return not self.missing_required_columns


MATCHES_REQUIRED_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "matchId"),
    "season": ("season",),
    "date": ("date",),
    "team1": ("team1",),
    "team2": ("team2",),
    "toss_winner": ("toss_winner",),
    "toss_decision": ("toss_decision",),
    "winner": ("winner",),
    "venue": ("venue",),
}

MATCHES_OPTIONAL_ALIASES: dict[str, tuple[str, ...]] = {
    "city": ("city",),
    "dl_applied": ("dl_applied",),
    "method": ("method",),
    "win_by_runs": ("win_by_runs", "winner_runs"),
    "win_by_wickets": ("win_by_wickets", "winner_wickets"),
    "result_margin": ("result_margin",),
    "player_of_match": ("player_of_match",),
    "neutral_venue": ("neutral_venue", "neutralvenue"),
    "eliminator": ("eliminator",),
    "umpire1": ("umpire1",),
    "umpire2": ("umpire2",),
    "result": ("result", "outcome"),
    "outcome": ("outcome",),
    "balls_per_over": ("balls_per_over",),
}

DELIVERIES_REQUIRED_ALIASES: dict[str, tuple[str, ...]] = {
    "match_id": ("match_id", "id", "matchId"),
    "inning": ("inning", "innings"),
    "batting_team": ("batting_team",),
    "bowling_team": ("bowling_team",),
    "over": ("over",),
    "ball": ("ball",),
    "striker": ("striker", "batter", "batsman"),
    "non_striker": ("non_striker",),
    "bowler": ("bowler",),
    "runs_off_bat": ("runs_off_bat", "batsman_runs", "batsman_run"),
    "extras": ("extras", "extra_runs"),
}

DELIVERIES_OPTIONAL_ALIASES: dict[str, tuple[str, ...]] = {
    "total_runs": ("total_runs",),
    "is_wicket": ("is_wicket",),
    "wides": ("wides", "isWide"),
    "noballs": ("noballs", "isNoBall"),
    "byes": ("byes", "Byes"),
    "legbyes": ("legbyes", "LegByes"),
    "penalty": ("penalty", "Penalty"),
    "dismissal_kind": ("dismissal_kind", "wicket_type"),
    "player_dismissed": ("player_dismissed",),
    "fielder": ("fielder",),
    "extras_type": ("extras_type",),
    "date": ("date", "start_date"),
    "over_ball": ("over_ball",),
}

MATCHES_OPTIONAL_DEFAULTS: dict[str, object] = {
    "city": pd.NA,
    "dl_applied": 0,
    "method": pd.NA,
    "win_by_runs": 0,
    "win_by_wickets": 0,
    "result_margin": pd.NA,
    "player_of_match": pd.NA,
    "neutral_venue": pd.NA,
    "eliminator": pd.NA,
    "umpire1": pd.NA,
    "umpire2": pd.NA,
    "result": pd.NA,
    "outcome": pd.NA,
    "balls_per_over": 6,
}

DELIVERIES_OPTIONAL_DEFAULTS: dict[str, object] = {
    "total_runs": pd.NA,
    "is_wicket": pd.NA,
    "wides": 0,
    "noballs": 0,
    "byes": 0,
    "legbyes": 0,
    "penalty": 0,
    "dismissal_kind": pd.NA,
    "player_dismissed": pd.NA,
    "fielder": pd.NA,
    "extras_type": pd.NA,
    "date": pd.NA,
    "over_ball": pd.NA,
}


def load_raw_matches(paths: ProjectPaths) -> pd.DataFrame:
    return pd.read_csv(paths.raw / RAW_MATCHES_FILENAME)


def load_raw_deliveries(paths: ProjectPaths) -> pd.DataFrame:
    return pd.read_csv(paths.raw / RAW_DELIVERIES_FILENAME)


def validate_raw_tables(matches: pd.DataFrame, deliveries: pd.DataFrame) -> tuple[SchemaReport, SchemaReport]:
    matches_report = _detect_schema(
        dataset_name="matches",
        frame=matches,
        required_aliases=MATCHES_REQUIRED_ALIASES,
        optional_aliases=MATCHES_OPTIONAL_ALIASES,
    )
    deliveries_report = _detect_schema(
        dataset_name="deliveries",
        frame=deliveries,
        required_aliases=DELIVERIES_REQUIRED_ALIASES,
        optional_aliases=DELIVERIES_OPTIONAL_ALIASES,
        over_index_base=detect_over_index_base(deliveries),
    )
    return matches_report, deliveries_report


def normalize_raw_tables(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[SchemaReport, SchemaReport]]:
    matches_report, deliveries_report = validate_raw_tables(matches, deliveries)

    if not matches_report.is_valid or not deliveries_report.is_valid:
        raise ValueError(build_validation_summary((matches_report, deliveries_report)))

    canonical_matches = _rename_columns(
        matches,
        required_aliases=MATCHES_REQUIRED_ALIASES,
        optional_aliases=MATCHES_OPTIONAL_ALIASES,
        optional_defaults=MATCHES_OPTIONAL_DEFAULTS,
    )
    canonical_deliveries = _rename_columns(
        deliveries,
        required_aliases=DELIVERIES_REQUIRED_ALIASES,
        optional_aliases=DELIVERIES_OPTIONAL_ALIASES,
        optional_defaults=DELIVERIES_OPTIONAL_DEFAULTS,
    )

    canonical_matches["date"] = pd.to_datetime(canonical_matches["date"], errors="coerce")
    canonical_matches["season"] = canonical_matches.apply(
        lambda row: canonicalize_season_label(row.get("season"), row.get("date")),
        axis=1,
    )
    canonical_matches["result"] = canonical_matches.apply(_derive_match_result, axis=1)
    canonical_matches["dl_applied"] = (
        canonical_matches["method"].fillna("").astype(str).str.strip().ne("").astype(int)
    )
    canonical_matches = _coerce_numeric_columns(
        canonical_matches,
        columns=["win_by_runs", "win_by_wickets", "dl_applied", "balls_per_over"],
    )

    for column in ("team1", "team2", "toss_winner", "winner"):
        canonical_matches[column] = canonical_matches[column].replace(TEAM_NAME_CANONICAL)

    for column in ("batting_team", "bowling_team"):
        canonical_deliveries[column] = canonical_deliveries[column].replace(TEAM_NAME_CANONICAL)

    canonical_deliveries = _coerce_numeric_columns(
        canonical_deliveries,
        columns=[
            "inning",
            "over",
            "ball",
            "runs_off_bat",
            "extras",
            "wides",
            "noballs",
            "byes",
            "legbyes",
            "penalty",
        ],
    )
    if canonical_deliveries["total_runs"].isna().all():
        canonical_deliveries["total_runs"] = canonical_deliveries["runs_off_bat"] + canonical_deliveries["extras"]
    else:
        canonical_deliveries["total_runs"] = pd.to_numeric(
            canonical_deliveries["total_runs"], errors="coerce"
        ).fillna(canonical_deliveries["runs_off_bat"] + canonical_deliveries["extras"])

    if canonical_deliveries["is_wicket"].isna().all():
        canonical_deliveries["is_wicket"] = (
            canonical_deliveries["dismissal_kind"].fillna("").astype(str).str.strip().ne("").astype(int)
        )
    else:
        canonical_deliveries["is_wicket"] = pd.to_numeric(
            canonical_deliveries["is_wicket"], errors="coerce"
        ).fillna(0).astype(int)

    over_index_base = deliveries_report.over_index_base
    if over_index_base == 0:
        canonical_deliveries["over"] = canonical_deliveries["over"] + 1

    return canonical_matches, canonical_deliveries, (matches_report, deliveries_report)


def write_canonical_raw_outputs(
    matches: pd.DataFrame,
    deliveries: pd.DataFrame,
    paths: ProjectPaths,
) -> tuple[Path, Path, tuple[SchemaReport, SchemaReport]]:
    paths.ensure()
    canonical_matches, canonical_deliveries, reports = normalize_raw_tables(matches, deliveries)

    matches_out = paths.interim / "matches_canonical.csv"
    deliveries_out = paths.interim / "deliveries_canonical.csv"

    canonical_matches.to_csv(matches_out, index=False)
    canonical_deliveries.to_csv(deliveries_out, index=False)
    return matches_out, deliveries_out, reports


def build_validation_summary(reports: Iterable[SchemaReport]) -> str:
    lines: list[str] = []
    for report in reports:
        lines.append(f"{report.dataset_name}: {'OK' if report.is_valid else 'INVALID'}")
        lines.append(f"  rows={report.row_count}")
        lines.append(f"  canonical_columns={len(report.canonical_columns)}")
        if report.alias_columns_used:
            alias_summary = ", ".join(
                f"{canonical}<-{source}" for canonical, source in sorted(report.alias_columns_used.items())
            )
            lines.append(f"  aliases={alias_summary}")
        if report.missing_required_columns:
            lines.append(
                "  missing_required=" + ", ".join(report.missing_required_columns)
            )
        if report.missing_optional_columns:
            lines.append(
                "  missing_optional=" + ", ".join(report.missing_optional_columns)
            )
        if report.over_index_base is not None:
            lines.append(f"  over_index_base={report.over_index_base}")
    return "\n".join(lines)


def detect_over_index_base(deliveries: pd.DataFrame) -> int | None:
    if "over" not in deliveries.columns or "inning" not in deliveries.columns:
        return None

    regular_overs = deliveries.loc[deliveries["inning"].isin([1, 2]), "over"].dropna()
    if regular_overs.empty:
        return None

    min_over = int(regular_overs.min())
    max_over = int(regular_overs.max())

    if min_over == 0 and max_over <= 19:
        return 0
    if min_over == 1 and max_over <= 20:
        return 1
    if min_over == 0:
        return 0
    if min_over == 1:
        return 1
    return None


def canonicalize_season_label(raw_season: object, match_date: object) -> str:
    if pd.notna(match_date):
        timestamp = pd.Timestamp(match_date)
        return str(timestamp.year)

    if pd.isna(raw_season):
        return "UNKNOWN"

    text = str(raw_season).strip()
    if text.isdigit():
        return str(int(text))

    if "/" in text:
        left, _, right = text.partition("/")
        left = left.strip()
        right = right.strip()
        if left.isdigit() and right.isdigit():
            if len(right) == 2:
                return str(int(f"{left[:2]}{right}"))
            if len(right) == 4:
                return right
    return text


def _derive_match_result(row: pd.Series) -> str:
    outcome = _safe_text(row.get("outcome", "")).lower()
    winner = _safe_text(row.get("winner", ""))
    if outcome == "tie":
        return "tie"
    if outcome == "no result":
        return "no result"
    if winner:
        return "normal"
    return "unknown"


def _coerce_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return frame


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _detect_schema(
    dataset_name: str,
    frame: pd.DataFrame,
    required_aliases: dict[str, tuple[str, ...]],
    optional_aliases: dict[str, tuple[str, ...]],
    over_index_base: int | None = None,
) -> SchemaReport:
    columns = set(frame.columns)
    alias_columns_used: dict[str, str] = {}
    missing_required: list[str] = []
    missing_optional: list[str] = []
    canonical_columns: list[str] = []

    for canonical_name, aliases in required_aliases.items():
        source = _first_present_alias(columns, aliases)
        if source is None:
            missing_required.append(canonical_name)
            continue
        canonical_columns.append(canonical_name)
        if source != canonical_name:
            alias_columns_used[canonical_name] = source

    for canonical_name, aliases in optional_aliases.items():
        source = _first_present_alias(columns, aliases)
        if source is None:
            missing_optional.append(canonical_name)
            continue
        canonical_columns.append(canonical_name)
        if source != canonical_name:
            alias_columns_used[canonical_name] = source

    return SchemaReport(
        dataset_name=dataset_name,
        row_count=len(frame),
        canonical_columns=tuple(canonical_columns),
        alias_columns_used=alias_columns_used,
        missing_required_columns=tuple(missing_required),
        missing_optional_columns=tuple(missing_optional),
        over_index_base=over_index_base,
    )


def _rename_columns(
    frame: pd.DataFrame,
    required_aliases: dict[str, tuple[str, ...]],
    optional_aliases: dict[str, tuple[str, ...]],
    optional_defaults: dict[str, object],
) -> pd.DataFrame:
    renamed = frame.copy()
    rename_map: dict[str, str] = {}

    for alias_group in (required_aliases, optional_aliases):
        for canonical_name, aliases in alias_group.items():
            source = _first_present_alias(set(renamed.columns), aliases)
            if source is not None and source != canonical_name:
                rename_map[source] = canonical_name

    renamed = renamed.rename(columns=rename_map)

    for canonical_name, default in optional_defaults.items():
        if canonical_name not in renamed.columns:
            renamed[canonical_name] = default

    return renamed


def _first_present_alias(columns: set[str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return alias
    return None
