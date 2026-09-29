import pandas as pd

# FIFA POINTS DATA:

df = pd.read_csv("data/blank.csv")
points = pd.read_csv("data/points.csv")

points_lookup = points.set_index("Team")["FIFA points"]
df["FIFA points"] = df["Team"].map(points_lookup)

opp_lookup = (df[["Team", "FIFA points"]].drop_duplicates(subset="Team").set_index("Team")["FIFA points"])
df["Opp FIFA points"] = df["Opponent"].map(opp_lookup)

print(df.describe())
print(df.info())

# GOALS SCORED AND CONCEDED DATA:

# Using data from a GitHub repository to extract goals scored/conceded data in the past 12 months
results = pd.read_csv("https://raw.githubusercontent.com/martj42/international_results/master/results.csv")

results["date"] = pd.to_datetime(results["date"])

# Use data up to 1 year before the world cup
start = pd.Timestamp("2025-06-11")
end = pd.Timestamp("2026-06-11")

results_past_year = results[(results["date"] >= start) & (results["date"] < end)].copy()

home = results_past_year[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
home.columns = ["date", "team", "opponent", "goals_scored", "goals_conceded"]

away = results_past_year[["date", "away_team", "home_team", "away_score", "home_score"]].copy()
away.columns = ["date", "team", "opponent", "goals_scored", "goals_conceded"]

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

df.to_csv("data/Data2.2.csv", index=False)