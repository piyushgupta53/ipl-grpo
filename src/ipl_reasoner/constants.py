"""Project constants derived from PLAN.md."""

from __future__ import annotations

TEAM_NAME_CANONICAL = {
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Pune Warriors": "Pune Warriors",
    "Kochi Tuskers Kerala": "Kochi Tuskers Kerala",
    "Gujarat Lions": "Gujarat Lions",
}

VENUE_SIMILAR_TO = {
    "mullanpur": "mohali",
    "aca_guwahati": "eden_gardens",
    "raipur": "narendra_modi_ahmedabad",
    "hpca_dharamshala": "chinnaswamy",
    "dy_patil": "wankhede",
    "brabourne": "wankhede",
}

STATS_SOURCE_SEASONS = ("2008", "2009", "2010", "2011", "2012")
TRAIN_SEASONS = tuple(str(year) for year in range(2013, 2024))
VALIDATION_SEASONS = ("2024",)
TEST_SEASONS = ("2025",)
LIVE_SEASONS = ("2026",)
AS_OF_SEASONS = tuple(str(year) for year in range(2013, 2027))

RAW_MATCHES_FILENAME = "matches_updated_ipl_upto_2025.csv"
RAW_DELIVERIES_FILENAME = "deliveries_updated_ipl_upto_2025.csv"

BASELINE_FEATURE_ORDER = (
    "required_run_rate",
    "balls_remaining",
    "wickets_in_hand",
)
