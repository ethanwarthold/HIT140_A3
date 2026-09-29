import pandas as pd

# FIFA POINTS DATA

df = pd.read_csv("data/blank.csv")
points = pd.read_csv("data/points.csv")

points_lookup = points.set_index("Team")["FIFA points"]
df["FIFA points"] = df["Team"].map(points_lookup)

opp_lookup = (df[["Team", "FIFA points"]].drop_duplicates(subset="Team").set_index("Team")["FIFA points"])
df["Opp FIFA points"] = df["Opponent"].map(opp_lookup)

print(df.describe())
print(df.info())

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

matches = pd.concat([home, away], ignore_index=True)

goals = matches.groupby("team").agg(
    average_scored=("goals_scored", "mean"),
    average_conceded=("goals_conceded", "mean"),
    total_scored=("goals_scored", "sum"),
    total_conceded=("goals_conceded", "sum"),
    matches=("goals_scored", "count")
).reset_index()

# goals.to_csv("data/goals_data.csv")
print(goals.describe())
print(goals.info())

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

goals["team"] = goals["team"].replace(name_map)

scored_lookup = goals.set_index("team")["average_scored"]
conceded_lookup = goals.set_index("team")["average_conceded"]

df["Avg Goals Scored"] = df["Team"].map(scored_lookup)
df["Opp Avg Conceded"] = df["Opponent"].map(conceded_lookup)

# Check if any values in the average goals colum are null
print(df.describe())
print(df.info())

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

# Put the opponent rest days column after the rest days column
cols = list(df.columns)
cols.remove("Opp Rest Days")
cols.insert(cols.index("Rest Days") + 1, "Opp Rest Days")
df = df[cols]

df = df.drop("Date", axis=1)

# SAVE TO CSV

df.to_csv("data/Data2.2.csv", index=False)