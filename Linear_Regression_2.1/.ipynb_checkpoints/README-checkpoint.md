# Linear Regression 2.1 — FIFA World Cup 2026 Goal Difference

## Overview

This project develops and evaluates a multiple linear regression model to predict the goal difference between two teams in a FIFA World Cup 2026 match. The dataset contains **104 matches**, with **eight explanatory variables** per match. Goal difference is defined as the home team's goals minus the away team's goals; a positive value favours the home team.

The work is recorded in `main.ipynb`, which prepares the data, explores the predictors, fits regression models, and evaluates predictions against match results.

## requirements
Python 3.10+

## install dependencies
```
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels openpyxl notebook ipykernel
```

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


# The notebook:

## Data collection — data_collection.ipynb

Cleans match fixtures and calculates Home-minus-Away goal differences, excluding penalty-shootout figures.

Calculates 2025 team rates, averages, and Home-minus-Away differences.

Adds the top-five-European-league player-count difference.

Merges two recent-match predictors from Linear Regression 2.2.

Saves the final dataset as data/dataset.csv.

## Analysis — main.ipynb

Checks for 104 matches, eight predictors, and missing values.

Displays descriptive statistics, correlations, scatterplots, and training-data histograms.

Fits full-data statsmodels OLS and examines coefficients, R², adjusted R², residuals, and a Q–Q plot.

Uses a 70/30 training/test split (random_state=0).

Compares original linear regression, Yeo–Johnson linear regression, Random Forest, SVR, and a mean baseline using identical five-fold training splits.

Fits transformations and scaling inside each cross-validation fold.

Evaluates test predictions using MAE, MSE, RMSE, normalised RMSE, and R².

Plots actual versus predicted test goal differences for the selected linear regression approach.

Normalised RMSE is calculated using the test goal-difference range. Full-data OLS results describe model fit; they are distinct from test prediction performance.

## Sources

- FIFA team data centre: <https://inside.fifa.com/data-centre/teams/cabo-verde?year=2025> (example page cited in the notebook for manually collected 2025 statistics).
- FBref World Cup fixtures/results: <https://fbref.com/en/comps/1/schedule/World-Cup-Scores-and-Fixtures>.
- The notebook comments identify FBref as the source for top-five-league player counts.
- Two recent-match predictors are imported from the group's separate Linear Regression 2.2 project.
