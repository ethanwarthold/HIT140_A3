# Linear Regression 2.1 — FIFA World Cup 2026 Goal Difference

## Overview

This project develops and evaluates a multiple linear regression model to predict the goal difference between two teams in a FIFA World Cup 2026 match. The dataset contains **104 matches**, with **eight explanatory variables** per match. Goal difference is defined as the home team's goals minus the away team's goals; a positive value favours the home team.

The work is recorded in `main.ipynb`, which prepares the data, explores the predictors, fits regression models, and evaluates predictions against actual match results.

## Python version and dependencies

Recommended Python version: Python 3.11 or 3.12 (64-bit)

## Project Structure
```
Linear_Regression_2.1/
├── main.ipynb
├── main.py
├── README.md
└── data/
    ├── 2025_games.xlsx
    ├── country_variables_2025.csv
    ├── dataset.csv
    ├── fifa_points.csv
    ├── match_teams_goal_difference.csv
    ├── sportsref_download-2.csv
    └── top_5_league_player.csv
```

## Files

| Path | Description |
| --- | --- |
| `main.ipynb` | Notebook for data preparation, exploratory analysis, training, and evaluation. |
| `data/dataset.csv` | Final modelling dataset (104 matches). |
| `data/sportsref_download-2.csv` | Source match fixtures and scores. |
| `data/match_teams_goal_difference.csv` | Home/away fixtures and derived goal differences. |
| `data/2025_games.xlsx` | Manually collected 2025 international team statistics. |
| `data/country_variables_2025.csv` | Derived win/draw rates and scoring/conceding averages. |
| `data/top_5_league_player.csv` | Supporting league-player data. |
| `data/fifa_points.csv` | Additional FIFA points data supplied with the project; not among the eight predictors in `dataset.csv`. |

The notebook also reads **`../Linear_Regression_2.2/data/data2.2.csv`** to obtain two recent-match predictors.


The notebook:

1. Cleans the match fixture data and derives goal differences from scores (excluding penalty-shootout figures).
2. Calculates 2025 team-level rates and home-minus-away differences.
3. Adds the top-five-league player count difference.
4. Merges two recent-match statistics from Regression 2.2.
5. Performs exploratory analysis using a correlation heatmap, histograms, and scatterplots.
6. Uses a **70/30 train/test split** (`random_state=0`).
7. Fits linear regression with scikit-learn and an OLS model with statsmodels.
8. Uses five-fold cross-validation on training data and compares ordinary linear regression with a Yeo–Johnson transformation pipeline.
9. Evaluates test predictions using MAE, MSE, RMSE, R², and a mean-prediction baseline.

## Sources

- FIFA team data centre: <https://inside.fifa.com/data-centre/teams/cabo-verde?year=2025> (example page cited in the notebook for manually collected 2025 statistics).
- FBref World Cup fixtures/results: <https://fbref.com/en/comps/1/schedule/World-Cup-Scores-and-Fixtures>.
- The notebook comments identify FBref as the source for top-five-league player counts.
- Two recent-match predictors are imported from the group's separate Linear Regression 2.2 project.
