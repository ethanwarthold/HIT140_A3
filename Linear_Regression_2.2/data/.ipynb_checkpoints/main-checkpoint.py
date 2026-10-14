import pandas as pd

# FIFA POINTS DATA

df = pd.read_csv("data/blank.csv")
points = pd.read_csv("data/points.csv")

points_lookup = points.set_index("Team")["FIFA points"]
df["FIFA points"] = df["Team"].map(points_lookup)

opp_lookup = (df[["Team", "FIFA points"]].drop_duplicates(subset="Team").set_index("Team")["FIFA points"])
df["Opp FIFA points"] = df["Opponent"].map(opp_lookup)

# GOALS SCORED AND CONCEDED DATA

# Using data from a GitHub repository to extract goals scored/conceded data in the past 12 months
results = pd.read_csv("https://raw.githubusercontent.com/martj42/international_results/master/results.csv")

results["date"] = pd.to_datetime(results["date"])

# Use data up to 1 year before the world cup
start = pd.Timestamp("2025-06-11")
end = pd.Timestamp("2026-06-11")

results_past_year = results[(results["date"] >= start) & (results["date"] < end)].copy()

home = results_past_year[["date", "home_team", "away_team", "home_score", "away_score"]].rename(
    columns = {
        "home_team": "team",
        "away_team": "opponent",
        "home_score": "goals_scored",
        "away_score": "goals_conceded"
    }
)

away = results_past_year[["date", "away_team", "home_team", "away_score", "home_score"]].rename(
    columns = {
        "away_team": "team",
        "home_team": "opponent",
        "away_score": "goals_scored",
        "home_score": "goals_conceded"
    }
)

team_data = pd.concat([home, away], ignore_index=True)

# Fix inconsistent naming between datasets
name_map = {
    "United States": "USA",
    "South Korea": "Korea Republic",
    "DR Congo": "Congo DR",
    "Iran": "IR Iran",
    "Ivory Coast": "Côte d'Ivoire",
    "Bosnia and Herzegovina": "Bosnia-Herz",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
    "Cape Verde": "Cabo Verde"
}

team_data["team"] = team_data["team"].replace(name_map)

team_data["result"] = team_data.apply(
    lambda row:
    "w" if row["goals_scored"] > row["goals_conceded"] else
    "l" if row["goals_scored"] < row["goals_conceded"] else
    "d",
    axis = 1
)

wins_data = team_data.groupby("team").agg(
    wins=("result", lambda x: (x == "w").sum()),
    matches=("result", "count")
).reset_index()

wins_data["win_rate"] = wins_data["wins"] / wins_data["matches"]

goals_data = team_data.groupby("team").agg(
    average_scored=("goals_scored", "mean"),
    average_conceded=("goals_conceded", "mean"),
    total_scored=("goals_scored", "sum"),
    total_conceded=("goals_conceded", "sum"),
    matches=("goals_scored", "count")
).reset_index()

# goals_data.to_csv("data/goals_data.csv", index=False)
# wins_data.to_csv("data/wins_data.csv", index=False)

print("=== GOALS DATA ===")
print(goals_data.info())
print("=== WINS DATA ===")
print(wins_data.info())

scored_lookup = goals_data.set_index("team")["average_scored"]
conceded_lookup = goals_data.set_index("team")["average_conceded"]
winrate_lookup = wins_data.set_index("team")["win_rate"]

df["Avg Scored"] = df["Team"].map(scored_lookup)
df["Opp Avg Conceded"] = df["Opponent"].map(conceded_lookup)
df["Win Rate"] = df["Team"].map(winrate_lookup)
df["Opp Win Rate"] = df["Opponent"].map(winrate_lookup)

# REST DAYS DATA

df["Date"] = pd.to_datetime(df["Date"])

wc_matches = pd.read_csv("data/wc_matches.csv")
wc_matches["date"] = pd.to_datetime(wc_matches["date"])

# Information for all matches in the world cup and 1 year before the world cup
stuff = pd.concat([results_past_year, wc_matches], ignore_index=True)

home = stuff[["date", "home_team"]].rename(columns={"home_team": "Team"})
away = stuff[["date", "away_team"]].rename(columns={"away_team": "Team"})

matches = pd.concat([home, away], ignore_index=True)
matches = matches.rename(columns={"date": "last_match_date"}).sort_values(["last_match_date", "Team"])

matches["Team"] = matches["Team"].replace(name_map)

df = df.sort_values(["Date", "Team"])

df = pd.merge_asof(
    df,
    matches,
    left_on="Date",
    right_on="last_match_date",
    by="Team",
    direction="backward",
    allow_exact_matches=False # look for row with an earlier date
)

df["Rest Days"] = (df["Date"] - df["last_match_date"]).dt.days

# Restore the original ordering and remove unecessary columns
df = df.sort_values("index").drop(["index", "last_match_date"], axis=1)

# Get the opponent rest days
opp_rest_days = df[["Date", "Team", "Rest Days"]].rename(columns={
    "Team": "Opponent",
    "Rest Days": "Opp Rest Days"
})

df = df.merge(
    opp_rest_days,
    on=["Date", "Opponent"],
    how="left"
)

# RECENT MATCHES DATA

results_f = results[(results["tournament"] != "FIFA World Cup")].copy() # results formatted, remove world cup matches to prevent duplicates
results_f["home_team"] = results_f["home_team"].replace(name_map)
results_f["away_team"] = results_f["away_team"].replace(name_map)

# All matches before and during the world cup
results_f = pd.concat([results_f, wc_matches], ignore_index=True)

# Format columns to include the date, team name, goals scored and conceded
# Matches will be duplicated, one row per team
# This makes it simpler to identify which matches to use later

home = results_f[["date", "home_team", "home_score", "away_score"]].rename(columns={
    "home_team": "team",
    "home_score": "goals_scored",
    "away_score": "goals_conceded"
})

away = results_f[["date", "away_team", "away_score", "home_score"]].rename(columns={
    "away_team": "team",
    "away_score": "goals_scored",
    "home_score": "goals_conceded"
})

matches = pd.concat([home, away], ignore_index=True).sort_values("date")

def get_last_n_avg(team, date, n, type):
    previous = matches[(matches["team"] == team) & (matches["date"] < date)].sort_values("date").tail(n)

    if len(previous) < n: # check if any teams have less than the desired number of matches
        print(f"Less than {n} previous matches for {team}, only {len(previous)} matches")

    return previous[f"goals_{type}"].mean()

n = 5 # Using scoring data for the previous 5 matches

df[f"Last {n} Avg Scored"] = df.apply(lambda row: get_last_n_avg(row["Team"], row["Date"], n, "scored"), axis=1)
df[f"Last {n} Opp Avg Conceded"] = df.apply(lambda row: get_last_n_avg(row["Opponent"], row["Date"], n, "conceded"), axis=1)

# xG DATA - Unused

team_ids = pd.read_csv("https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/teams.csv")
matches_data = pd.read_csv("https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main/matches.csv")

team_lookup = team_ids.set_index("team_id")["team_name"]

matches_data["date"] = pd.to_datetime(matches_data["date"])
matches_data["home_team"] = matches_data["home_team_id"].map(team_lookup)
matches_data["away_team"] = matches_data["away_team_id"].map(team_lookup)

home = matches_data[["date", "home_team", "home_xg"]].rename(columns={
    "home_team": "team",
    "home_xg": "xg"
})

away = matches_data[["date", "away_team", "away_xg"]].rename(columns={
    "away_team": "team",
    "away_xg": "xg"
})

xg_data = pd.concat([home, away], ignore_index=True).sort_values("date")

xg_data["team"] = xg_data["team"].replace(name_map)

xg_data.to_csv("data/xg_data.csv", index=False)

# for the xG column for the first match, perhaps use the xG from the most recent match played by the team
# and for later matches, average the xG across previous matches

# SAVE TO CSV

# Order columns and remove unecessary columns
# Opponent Rest Days and Opponent Win Rate have been removed so the total number of variables is 8

df = df[
    [
        "Team",
        "Opponent",
        "FIFA points",
        "Opp FIFA points",
        "Avg Scored",
        "Opp Avg Conceded",
        "Rest Days",
        # "Opp Rest Days",
        f"Last {n} Avg Scored",
        f"Last {n} Opp Avg Conceded",
        "Win Rate",
        # "Opp Win Rate",
        "Goals"
    ]
]

print("=== FINAL DATA ===")
print(df.info()) # Final verification that each column has 208 values

df.to_csv("data/Data2.2.csv", index=False)
