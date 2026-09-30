"""
This file was made to match the formatting of wc_matches.csv to the full international match history data

"""

import pandas as pd

wc_matches = pd.read_csv("data/wc_matches_old.csv")
blank = pd.read_csv("data/blank.csv")

wc_matches["date"] = pd.to_datetime(wc_matches["date"])
blank["Date"] = pd.to_datetime(blank["Date"])

# Match column naming

home_scores = blank[["Date", "Team", "Opponent", "Goals"]].rename(columns={
    "Date": "date",
    "Team": "home_team",
    "Opponent": "away_team",
    "Goals": "home_score"
})

away_scores = blank[["Date", "Team", "Opponent", "Goals"]].rename(columns={
    "Date": "date",
    "Team": "away_team",
    "Opponent": "home_team",
    "Goals": "away_score"
})

# Merge so both team scores are included

wc_matches = wc_matches.merge(home_scores, on=["date", "home_team", "away_team"], how="left")
wc_matches = wc_matches.merge(away_scores, on=["date", "home_team", "away_team"], how="left")

wc_matches.to_csv("data/wc_matches.csv", index=False)