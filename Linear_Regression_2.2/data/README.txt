blank.csv:
  An empty dataset containing the explanatory variable names in the columns, and team and opponent names for each world cup game (one row for each team)
  The team match-ups and goals scored data is taken from: https://fbref.com/en/comps/1/schedule/World-Cup-Scores-and-Fixtures#all_sched_all
  The data was formatted partially manually, and partially using the find and replace feature in Visual Studio Code (with regular expressions)

Data2.2.csv:
  The full data CSV that will be used in linear regression
  It is constructed by populating values into blank.csv and saving it to Data2.2.csv

points.csv:
  FIFA points for each team in the FIFA World Cup, before the World Cup Started (11th June 2026)
  Taken from https://inside.fifa.com/fifa-rankings/world-ranking/men?dateId=FRS_Male_Football_20260401
  Pasted into Google Sheets, formatted to only include the team name and points columns, then exported to points.csv