"""Venue aliasing, metadata seeds, and venue prior generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ipl_reasoner.constants import AS_OF_SEASONS
from ipl_reasoner.paths import ProjectPaths

VENUE_CODE_BY_NAME = {
    "Arun Jaitley Stadium": "arun_jaitley_delhi",
    "Arun Jaitley Stadium, Delhi": "arun_jaitley_delhi",
    "Barabati Stadium": "barabati",
    "Barsapara Cricket Stadium, Guwahati": "aca_guwahati",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "ekana_lucknow",
    "Brabourne Stadium": "brabourne",
    "Brabourne Stadium, Mumbai": "brabourne",
    "Buffalo Park": "buffalo_park",
    "De Beers Diamond Oval": "de_beers_diamond_oval",
    "Dubai International Cricket Stadium": "dubai_international",
    "Dr DY Patil Sports Academy": "dy_patil",
    "Dr DY Patil Sports Academy, Mumbai": "dy_patil",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "aca_vdca_visakhapatnam",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "aca_vdca_visakhapatnam",
    "Eden Gardens": "eden_gardens",
    "Eden Gardens, Kolkata": "eden_gardens",
    "Feroz Shah Kotla": "arun_jaitley_delhi",
    "Green Park": "green_park",
    "Himachal Pradesh Cricket Association Stadium": "hpca_dharamshala",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "hpca_dharamshala",
    "Holkar Cricket Stadium": "holkar",
    "JSCA International Stadium Complex": "jsca_ranchi",
    "Kingsmead": "kingsmead",
    "M Chinnaswamy Stadium": "chinnaswamy",
    "M Chinnaswamy Stadium, Bengaluru": "chinnaswamy",
    "M.Chinnaswamy Stadium": "chinnaswamy",
    "MA Chidambaram Stadium": "chepauk",
    "MA Chidambaram Stadium, Chepauk": "chepauk",
    "MA Chidambaram Stadium, Chepauk, Chennai": "chepauk",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": "mullanpur",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": "mullanpur",
    "Maharashtra Cricket Association Stadium": "pune_mca",
    "Maharashtra Cricket Association Stadium, Pune": "pune_mca",
    "Narendra Modi Stadium, Ahmedabad": "narendra_modi_ahmedabad",
    "Nehru Stadium": "nehru_stadium",
    "New Wanderers Stadium": "new_wanderers",
    "Newlands": "newlands",
    "OUTsurance Oval": "outsurance_oval",
    "Punjab Cricket Association IS Bindra Stadium": "mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "mohali",
    "Punjab Cricket Association Stadium, Mohali": "mohali",
    "Rajiv Gandhi International Stadium": "rajiv_gandhi_hyderabad",
    "Rajiv Gandhi International Stadium, Uppal": "rajiv_gandhi_hyderabad",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "rajiv_gandhi_hyderabad",
    "Sardar Patel Stadium, Motera": "narendra_modi_ahmedabad",
    "Saurashtra Cricket Association Stadium": "saurashtra_cricket_association",
    "Sawai Mansingh Stadium": "sawai_mansingh",
    "Sawai Mansingh Stadium, Jaipur": "sawai_mansingh",
    "Shaheed Veer Narayan Singh International Stadium": "raipur",
    "Sharjah Cricket Stadium": "sharjah",
    "Sheikh Zayed Stadium": "sheikh_zayed",
    "St George's Park": "st_georges_park",
    "Subrata Roy Sahara Stadium": "subrata_roy_sahara",
    "SuperSport Park": "supersport_park",
    "Vidarbha Cricket Association Stadium, Jamtha": "vidarbha_jamtha",
    "Wankhede Stadium": "wankhede",
    "Wankhede Stadium, Mumbai": "wankhede",
    "Zayed Cricket Stadium, Abu Dhabi": "sheikh_zayed",
}

VENUE_METADATA_ROWS = [
    {"venue_code": "wankhede", "pitch_type": "batting", "boundary_size": "medium", "dew_factor": "high", "climate": "coastal_mumbai", "spin_assist": "low", "pace_assist": "medium", "surface_descriptor": "Sea-breeze aided batting track with chase-friendly dew", "notes": "Sea breeze can create early swing; heavy evening dew and strong chase bias."},
    {"venue_code": "chinnaswamy", "pitch_type": "batting", "boundary_size": "short", "dew_factor": "medium", "climate": "elevated_blr", "spin_assist": "medium", "pace_assist": "low", "surface_descriptor": "Short-boundary high-scoring venue", "notes": "Short square boundaries and altitude support 200-plus totals."},
    {"venue_code": "chepauk", "pitch_type": "spin", "boundary_size": "medium", "dew_factor": "medium", "climate": "coastal_humid", "spin_assist": "high", "pace_assist": "low", "surface_descriptor": "Classic tacky Chennai surface", "notes": "Historically rewards spin and can favor bat-first setups."},
    {"venue_code": "eden_gardens", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "medium", "climate": "eastern_humid", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Balanced Kolkata surface with crowd-pressure factor", "notes": "Usually offers a fair bat-ball contest with moderate dew."},
    {"venue_code": "rajiv_gandhi_hyderabad", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "low", "climate": "deccan", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Even contest venue in Hyderabad", "notes": "Typically neutral conditions without an extreme toss skew."},
    {"venue_code": "sawai_mansingh", "pitch_type": "batting", "boundary_size": "medium", "dew_factor": "low", "climate": "dry_jaipur", "spin_assist": "medium", "pace_assist": "low", "surface_descriptor": "Dry Jaipur batting surface", "notes": "Can be flat with some grip later in the innings."},
    {"venue_code": "arun_jaitley_delhi", "pitch_type": "batting", "boundary_size": "medium", "dew_factor": "low", "climate": "northern_dry", "spin_assist": "low", "pace_assist": "medium", "surface_descriptor": "Delhi surface with hot, dry conditions", "notes": "Often starts with some seam movement before flattening out."},
    {"venue_code": "narendra_modi_ahmedabad", "pitch_type": "batting", "boundary_size": "large", "dew_factor": "low", "climate": "central_dry", "spin_assist": "low", "pace_assist": "low", "surface_descriptor": "Large-ground Ahmedabad batting venue", "notes": "Big boundaries can slightly suppress boundary rates despite true bounce."},
    {"venue_code": "ekana_lucknow", "pitch_type": "batting", "boundary_size": "medium", "dew_factor": "medium", "climate": "northern_plains", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Newer Lucknow venue with mixed scoring profile", "notes": "Limited long-run history; recent seasons have been more batting-friendly."},
    {"venue_code": "mohali", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "low", "climate": "punjab", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Traditional Punjab venue with even conditions", "notes": "Usually offers true bounce and a balanced contest."},
    {"venue_code": "hpca_dharamshala", "pitch_type": "pace", "boundary_size": "medium", "dew_factor": "low", "climate": "elevated_himachal", "spin_assist": "low", "pace_assist": "high", "surface_descriptor": "High-altitude seam-friendly ground", "notes": "Mountain air and elevation help pace and swing; limited IPL sample."},
    {"venue_code": "mullanpur", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "low", "climate": "punjab", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "New Punjab venue with moderate evidence", "notes": "Limited data; use Mohali as the similarity prior."},
    {"venue_code": "aca_guwahati", "pitch_type": "batting", "boundary_size": "medium", "dew_factor": "medium", "climate": "northeast_humid", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Humid Guwahati batting venue", "notes": "Small historical sample; scoring has been healthy so far."},
    {"venue_code": "raipur", "pitch_type": "batting", "boundary_size": "medium", "dew_factor": "low", "climate": "central_india", "spin_assist": "low", "pace_assist": "low", "surface_descriptor": "Flat Raipur neutral-venue style surface", "notes": "Sparse history; use Ahmedabad-like prior characteristics."},
    {"venue_code": "dy_patil", "pitch_type": "batting", "boundary_size": "medium", "dew_factor": "high", "climate": "coastal_mumbai", "spin_assist": "low", "pace_assist": "medium", "surface_descriptor": "Mumbai-metro venue similar to Wankhede", "notes": "Neutral-venue usage but broadly similar conditions to Mumbai grounds."},
    {"venue_code": "brabourne", "pitch_type": "batting", "boundary_size": "medium", "dew_factor": "medium", "climate": "coastal_mumbai", "spin_assist": "low", "pace_assist": "medium", "surface_descriptor": "Compact Mumbai venue with true bounce", "notes": "Often high scoring with a coastal evening pattern."},
    {"venue_code": "pune_mca", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "low", "climate": "deccan", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Pune venue with balanced conditions", "notes": "Historical venue used by Pune franchises and neutral-season matches."},
    {"venue_code": "aca_vdca_visakhapatnam", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "medium", "climate": "coastal_andhra", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Coastal Vizag venue with moderate scoring", "notes": "Has served as both home and neutral venue in multiple seasons."},
    {"venue_code": "barabati", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "medium", "climate": "eastern_humid", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Occasionally used eastern venue", "notes": "Small historical IPL sample only."},
    {"venue_code": "green_park", "pitch_type": "spin", "boundary_size": "medium", "dew_factor": "low", "climate": "northern_plains", "spin_assist": "high", "pace_assist": "low", "surface_descriptor": "Kanpur surface with grip", "notes": "Limited IPL appearances; tends to assist slower bowling."},
    {"venue_code": "holkar", "pitch_type": "batting", "boundary_size": "medium", "dew_factor": "low", "climate": "central_india", "spin_assist": "low", "pace_assist": "low", "surface_descriptor": "Indore batting venue", "notes": "Generally quick-scoring with true bounce."},
    {"venue_code": "jsca_ranchi", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "medium", "climate": "eastern_plateau", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Ranchi venue with mixed evidence", "notes": "Short IPL history with fairly balanced outcomes."},
    {"venue_code": "saurashtra_cricket_association", "pitch_type": "batting", "boundary_size": "medium", "dew_factor": "low", "climate": "western_dry", "spin_assist": "low", "pace_assist": "medium", "surface_descriptor": "Rajkot batting venue", "notes": "Often fast outfield and good batting conditions."},
    {"venue_code": "subrata_roy_sahara", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "low", "climate": "northern_plains", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Retired Pune venue", "notes": "Historical only; moderate all-round conditions."},
    {"venue_code": "vidarbha_jamtha", "pitch_type": "balanced", "boundary_size": "large", "dew_factor": "low", "climate": "central_india", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Nagpur venue with large boundaries", "notes": "Very small historical IPL sample."},
    {"venue_code": "kingsmead", "pitch_type": "pace", "boundary_size": "medium", "dew_factor": "medium", "climate": "coastal_sa", "spin_assist": "low", "pace_assist": "high", "surface_descriptor": "South African coastal pace venue", "notes": "2009-only venue with seam movement likely."},
    {"venue_code": "new_wanderers", "pitch_type": "pace", "boundary_size": "medium", "dew_factor": "low", "climate": "highveld", "spin_assist": "low", "pace_assist": "high", "surface_descriptor": "Johannesburg altitude venue", "notes": "Bounce and carry help seamers; used in 2009."},
    {"venue_code": "newlands", "pitch_type": "pace", "boundary_size": "medium", "dew_factor": "low", "climate": "coastal_sa", "spin_assist": "low", "pace_assist": "high", "surface_descriptor": "Cape Town pace-friendly venue", "notes": "Limited IPL use during the South Africa season."},
    {"venue_code": "st_georges_park", "pitch_type": "pace", "boundary_size": "medium", "dew_factor": "low", "climate": "coastal_sa", "spin_assist": "low", "pace_assist": "high", "surface_descriptor": "Port Elizabeth seam-leaning venue", "notes": "Historical-only IPL venue from 2009."},
    {"venue_code": "supersport_park", "pitch_type": "pace", "boundary_size": "medium", "dew_factor": "low", "climate": "highveld", "spin_assist": "low", "pace_assist": "high", "surface_descriptor": "Centurion pace venue", "notes": "Altitude and carry favor quicks."},
    {"venue_code": "buffalo_park", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "low", "climate": "coastal_sa", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "East London venue with tiny IPL sample", "notes": "Historical-only and very sparse."},
    {"venue_code": "de_beers_diamond_oval", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "low", "climate": "interior_sa", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Kimberley venue with minimal IPL history", "notes": "Historical-only sparse venue."},
    {"venue_code": "outsurance_oval", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "low", "climate": "south_africa", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Bloemfontein venue with extremely small sample", "notes": "Historical-only sparse venue."},
    {"venue_code": "sheikh_zayed", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "low", "climate": "uae_desert", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Abu Dhabi UAE venue", "notes": "Used in multiple UAE seasons; slower than Sharjah."},
    {"venue_code": "dubai_international", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "low", "climate": "uae_desert", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Dubai UAE venue with balanced scoring", "notes": "Used heavily in UAE seasons; generally more balanced than Sharjah."},
    {"venue_code": "sharjah", "pitch_type": "batting", "boundary_size": "short", "dew_factor": "low", "climate": "uae_desert", "spin_assist": "low", "pace_assist": "low", "surface_descriptor": "Sharjah small-ground batting venue", "notes": "Short dimensions can create extreme scoring spikes."},
    {"venue_code": "nehru_stadium", "pitch_type": "balanced", "boundary_size": "medium", "dew_factor": "medium", "climate": "coastal_humid", "spin_assist": "medium", "pace_assist": "medium", "surface_descriptor": "Kochi venue with tiny IPL sample", "notes": "Historical-only venue from the Kochi season."},
]


@dataclass(frozen=True)
class VenueArtifacts:
    venue_metadata_path: Path
    venue_season_stats_path: Path
    venue_alias_report_path: Path


def apply_venue_codes(matches: pd.DataFrame, deliveries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = matches.copy()
    deliveries = deliveries.copy()

    matches["venue_code"] = matches["venue"].map(VENUE_CODE_BY_NAME)
    deliveries["venue_code"] = deliveries["venue"].map(VENUE_CODE_BY_NAME)

    unknown_matches = sorted(matches.loc[matches["venue_code"].isna(), "venue"].dropna().unique())
    unknown_deliveries = sorted(deliveries.loc[deliveries["venue_code"].isna(), "venue"].dropna().unique())
    unknown = sorted(set(unknown_matches).union(unknown_deliveries))
    if unknown:
        raise ValueError(f"Unmapped venue names: {unknown}")

    return matches, deliveries


def build_venue_artifacts(matches: pd.DataFrame, merged_deliveries: pd.DataFrame, paths: ProjectPaths) -> VenueArtifacts:
    paths.ensure()
    matches_with_codes, deliveries_with_codes = apply_venue_codes(matches, merged_deliveries)

    venue_metadata = pd.DataFrame(VENUE_METADATA_ROWS).sort_values("venue_code").reset_index(drop=True)
    used_codes = set(matches_with_codes["venue_code"].dropna().unique())
    metadata_codes = set(venue_metadata["venue_code"])
    missing_metadata = sorted(used_codes - metadata_codes)
    if missing_metadata:
        raise ValueError(f"Missing venue metadata rows for codes: {missing_metadata}")

    alias_report = (
        matches_with_codes.groupby(["venue", "venue_code"], as_index=False)
        .size()
        .rename(columns={"size": "match_count"})
        .sort_values(["venue_code", "venue"])
        .reset_index(drop=True)
    )

    venue_stats = _build_venue_season_stats(matches_with_codes, deliveries_with_codes)

    venue_metadata_path = paths.artifacts / "metadata" / "venue_metadata.csv"
    venue_season_stats_path = paths.artifacts / "metadata" / "venue_season_stats.csv"
    venue_alias_report_path = paths.reports / "venue_alias_report.csv"

    venue_metadata.to_csv(venue_metadata_path, index=False)
    venue_stats.to_csv(venue_season_stats_path, index=False)
    alias_report.to_csv(venue_alias_report_path, index=False)

    return VenueArtifacts(
        venue_metadata_path=venue_metadata_path,
        venue_season_stats_path=venue_season_stats_path,
        venue_alias_report_path=venue_alias_report_path,
    )


def _build_venue_season_stats(matches: pd.DataFrame, merged_deliveries: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    deliveries = merged_deliveries.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    deliveries["date"] = pd.to_datetime(deliveries["date"], errors="coerce")

    innings_totals = (
        deliveries.groupby(["match_id", "inning", "venue_code"], as_index=False)
        .agg(total_runs=("total_runs", "sum"))
    )

    inning1 = innings_totals.loc[innings_totals["inning"] == 1, ["match_id", "venue_code", "total_runs"]].rename(
        columns={"total_runs": "first_innings_total"}
    )
    inning2 = innings_totals.loc[innings_totals["inning"] == 2, ["match_id", "venue_code", "total_runs"]].rename(
        columns={"total_runs": "second_innings_total"}
    )
    match_totals = inning1.merge(inning2, on=["match_id", "venue_code"], how="inner")

    death = deliveries.loc[(deliveries["inning"] == 2) & (deliveries["over"] >= 17)].copy()
    death_by_match = death.groupby(["match_id", "venue_code"], as_index=False).agg(
        death_runs=("total_runs", "sum"),
        death_legal_balls=("legal_ball", "sum"),
    )
    death_by_match["death_rpo"] = (
        death_by_match["death_runs"] / death_by_match["death_legal_balls"].replace(0, pd.NA) * 6
    )

    over10 = deliveries.loc[(deliveries["inning"] == 2) & (deliveries["over"] <= 10)].copy()
    over10_by_match = over10.groupby(["match_id", "venue_code"], as_index=False).agg(
        score_at_over_10=("total_runs", "sum")
    )

    enriched = matches[["id", "date", "season", "venue_code", "winner", "team1", "team2", "toss_winner", "toss_decision"]].rename(
        columns={"id": "match_id"}
    )
    enriched = enriched.merge(match_totals, on=["match_id", "venue_code"], how="left")
    enriched = enriched.merge(death_by_match[["match_id", "venue_code", "death_rpo"]], on=["match_id", "venue_code"], how="left")
    enriched = enriched.merge(over10_by_match, on=["match_id", "venue_code"], how="left")
    enriched["chasing_team"] = enriched.apply(_derive_chasing_team, axis=1)
    enriched["chasing_team_won"] = (enriched["winner"] == enriched["chasing_team"]).astype(int)
    all_venue_codes = sorted(matches["venue_code"].dropna().unique())

    rows: list[dict[str, object]] = []
    for season in AS_OF_SEASONS:
        cutoff = pd.Timestamp(f"{season}-04-01")
        history = enriched.loc[enriched["date"] < cutoff].copy()
        for venue_code in all_venue_codes:
            group = history.loc[history["venue_code"] == venue_code].copy()
            matches_count = len(group)
            rows.append(
                {
                    "venue_code": venue_code,
                    "as_of_season": season,
                    "matches_count": matches_count,
                    "avg_first_innings_score": group["first_innings_total"].mean(),
                    "avg_second_innings_score": group["second_innings_total"].mean(),
                    "chase_success_rate": group["chasing_team_won"].mean(),
                    "avg_death_rpo": group["death_rpo"].mean(),
                    "avg_par_score_at_over_10": group["score_at_over_10"].mean(),
                    "confidence_alpha": min(1.0, matches_count / 20.0),
                }
            )

    return pd.DataFrame(rows).sort_values(["as_of_season", "venue_code"]).reset_index(drop=True)


def _derive_chasing_team(row: pd.Series) -> str:
    toss_decision = str(row.get("toss_decision", "")).strip().lower()
    toss_winner = str(row.get("toss_winner", "")).strip()
    team1 = str(row.get("team1", "")).strip()
    team2 = str(row.get("team2", "")).strip()
    if toss_decision == "field":
        return toss_winner
    if toss_winner == team1:
        return team2
    if toss_winner == team2:
        return team1
    return team2
