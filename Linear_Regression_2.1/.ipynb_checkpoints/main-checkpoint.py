#!/usr/bin/env python
# coding: utf-8

# In[2]:


#This notebook includes the import of csv file and then use only the necessary columns
#some cleaning is also done like removing country codes
import pandas as pd

df = pd.read_csv("data/sportsref_download-2.csv")

df = df[['Home', 'Away', 'Score']].dropna()
df = df[df['Home'] != 'Home'].copy()

# Remove country codes
df['Home'] = df['Home'].str.rsplit(' ', n=1).str[0]
df['Away'] = df['Away'].str.split(' ', n=1).str[1]

df = df.replace({
    'USA': 'United States',
    'Bosnia–Herz': 'Bosnia and Herzegovina',
    'Korea Republic': 'South Korea',
    'IR Iran': 'Iran',
    'Congo DR': 'Democratic Republic of the Congo'
})

# Extract goals, excluding penalty shoot-out numbers
goals = df['Score'].str.extract(r'(\d+)[–-](\d+)').astype(int)
df['Goal_Difference'] = goals[0] - goals[1]

df = df[['Home', 'Away', 'Goal_Difference']]
df.to_csv('data/match_teams_goal_difference.csv', index=False)

df.head()


# In[3]:


#2025 international games were taken manually from "https://inside.fifa.com/data-centre/teams/cabo-verde?year=2025" website
#from that website, total matches, wins, draws, defeats, goal scored, goal conceded were taken

import pandas as pd

teams = pd.read_excel("data/2025_games.xlsx")
#calculation of variables
teams["Win_Rate"] = teams["Wins"] / teams["Total matches"]
teams["Draw_Rate"] = teams["Draws"] / teams["Total matches"]
teams["Avg_Scored"] = teams["Goals scored"] / teams["Total matches"]
teams["Avg_Conceded"] = teams["Goals conceded"] / teams["Total matches"]

teams.to_csv("data/country_variables_2025.csv", index=False)

teams.head()


# In[4]:


#adding different explanatory variables in the main dataset
import pandas as pd

matches = pd.read_csv("data/dataset.csv")
teams = pd.read_csv("data/country_variables_2025.csv").set_index("Team")

variables = {
    "Total matches": "Matches_Played_Difference",
    "Win_Rate": "Win_Rate_Difference",
    "Draw_Rate": "Draw_Rate_Difference",
    "Avg_Scored": "Avg_Scored_Difference",
    "Avg_Conceded": "Avg_Conceded_Difference"
}

for variable, column in variables.items():
    home = matches["Home"].map(teams[variable])
    away = matches["Away"].map(teams[variable])

    if home.isna().any() or away.isna().any():
        raise ValueError("Check that team names match in both files.")

    matches[column] = home - away

goal_difference = matches.pop("Goal_Difference")
matches["Goal_Difference"] = goal_difference

matches.to_csv("data/dataset.csv", index=False, float_format="%.4f")

matches.head()


# In[5]:


#this is the data from the fbref where player played in top 5 league
#if the team doesnot have a player in top 5 league, it returns 0
import pandas as pd

players = {
    "Algeria": 11, "Argentina": 21, "Australia": 5,
    "Austria": 17, "Belgium": 20, "Bosnia and Herzegovina": 6,
    "Brazil": 15, "Cabo Verde": 1, "Canada": 9,
    "Colombia": 8, "Croatia": 17, "Curaçao": 0,
    "Czechia": 7, "Côte d'Ivoire": 13,
    "Democratic Republic of the Congo": 10,
    "Ecuador": 6, "Egypt": 4, "England": 26,
    "France": 24, "Germany": 26, "Ghana": 12,
    "Haiti": 4, "Iran": 0, "Iraq": 0,
    "Japan": 13, "Jordan": 1, "Mexico": 6,
    "Morocco": 17, "Netherlands": 23, "New Zealand": 1,
    "Norway": 17, "Panama": 0, "Paraguay": 4,
    "Portugal": 17, "Qatar": 0, "Saudi Arabia": 1,
    "Scotland": 12, "Senegal": 20, "South Africa": 1,
    "South Korea": 5, "Spain": 26, "Sweden": 15,
    "Switzerland": 23, "Tunisia": 7, "Türkiye": 9,
    "United States": 13, "Uruguay": 8, "Uzbekistan": 1
}

matches = pd.read_csv("data/dataset.csv")

home = matches["Home"].str.strip().map(players)
away = matches["Away"].str.strip().map(players)

if home.isna().any() or away.isna().any():
    raise ValueError("Some team names do not match. Check their spelling.")

matches["Top_5_League_Players_Difference"] = home - away

# Keep goal difference last.
goal_difference = matches.pop("Goal_Difference")
matches["Goal_Difference"] = goal_difference

matches.to_csv("data/dataset.csv", index=False, float_format="%.4f")

matches.head()


# In[7]:


#using data from Linear regression 2.2 file
import pandas as pd

matches = pd.read_csv("data/dataset.csv")

other = pd.read_csv(
    "../Linear_Regression_2.2/data/data2.2.csv"
)

print(other.columns.tolist())
print(other.head())


# In[8]:


#using last 5 avg scored and Last 5 opp avvg conceded
aliases = {
    "Korea Republic": "South Korea",
    "USA": "United States",
    "Bosnia-Herz": "Bosnia and Herzegovina",
    "IR Iran": "Iran",
    "Congo DR": "Democratic Republic of the Congo"
}

data = other[
    ["Team", "Opponent", "Last 5 Avg Scored", "Last 5 Opp Avg Conceded"]
].copy()

data = data.rename(columns={"Team": "Home", "Opponent": "Away"})

for column in ["Home", "Away"]:
    data[column] = data[column].str.strip().replace(aliases)

columns = ["Last 5 Avg Scored", "Last 5 Opp Avg Conceded"]
matches = matches.drop(columns=columns, errors="ignore")

matches = matches.merge(
    data, on=["Home", "Away"], how="left", validate="many_to_one"
)

if matches[columns].isna().any().any():
    raise ValueError("Some team names did not match.")

matches.to_csv("data/dataset.csv", index=False, float_format="%.4f")


# In[10]:


#loading data
#X has all the 8 explanatory variables
#Y has the predictive value
import pandas as pd

data = pd.read_csv("data/dataset.csv")

X = data.drop(columns=["Home", "Away", "Goal_Difference"])
y = data["Goal_Difference"]

print("Matches:", len(data))
print("Explanatory variables:", X.shape[1])
print("Missing values:", data.isna().sum().sum())

data.head()


# In[11]:


#data split into training and testing set
#70% training and 30% testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))


# In[12]:


#displaying the correlation between predictors and goal difference
#using training data to find out strong related variables
import matplotlib.pyplot as plt
import seaborn as sns

training_data = X_train.copy()
training_data["Goal_Difference"] = y_train

sns.heatmap(training_data.corr(),
            cmap="coolwarm",
            vmin=-1,
            vmax=1)

plt.show()


# In[13]:


#identify skewed distribution
X_train.hist(figsize=(14, 14), layout=(4, 2))
plt.tight_layout(pad=3)

plt.show()


# In[14]:


# Examining each predictor's relationship with goal difference.
fig, axes = plt.subplots(4, 2, figsize=(14, 14))

for ax, column in zip(axes.flat, X_train.columns):
    ax.scatter(X_train[column], y_train)
    ax.set_xlabel(column)
    ax.set_ylabel("Goal difference")

plt.tight_layout(pad=3)
plt.show()


# In[15]:


# Training linear regression by using all eight explanatory variables
# And displaying the intercept and coefficients
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Display the intercept and coefficients.
print("Intercept:", round(model.intercept_, 4))

coefficients = pd.DataFrame({
    "Explanatory variable": X_train.columns,
    "Coefficient": model.coef_
})

coefficients.round(4)


# In[16]:


# Includeing an intercept and fit OLS by using training data.
import statsmodels.api as sm

X_train_ols = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_ols).fit()

print(ols_model.summary())


# In[17]:


# Calculating the training residuals: actual minus predicted.
import matplotlib.pyplot as plt

training_predictions = model.predict(X_train)
residuals = y_train - training_predictions

plt.scatter(training_predictions, residuals)
plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Predicted goal difference")
plt.ylabel("Residual (actual − predicted)")
plt.title("Training residuals")
plt.show()


# In[18]:


# Evaluateing the linear regression across five training-data folds
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    LinearRegression(),
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error"
)

# lets convert each fold's error into RMSE.
rmse_scores = (-scores) ** 0.5

print("RMSE for each fold:", rmse_scores.round(4))
print("Average RMSE:", round(rmse_scores.mean(), 4))
print("RMSE standard deviation:", round(rmse_scores.std(), 4))


# In[19]:


#comparing it with Yeo–Johnson plus linear regression while using the same folds
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer

# Fitting the transformation separately inside each fold.
yj_model = make_pipeline(
    PowerTransformer(method="yeo-johnson"),
    LinearRegression()
)

yj_scores = cross_val_score(
    yj_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error"
)

yj_rmse = (-yj_scores) ** 0.5

comparison = pd.DataFrame({
    "Model": ["Original", "Yeo–Johnson"],
    "Average CV RMSE": [rmse_scores.mean(), yj_rmse.mean()],
    "RMSE standard deviation": [rmse_scores.std(), yj_rmse.std()]
})

comparison.round(4)


# In[20]:


#fitting both approaches on the full training set and comparing their test predictions
from sklearn import metrics

# Fitted the transformed model using all training matches.
yj_model.fit(X_train, y_train)

# Evaluating both approaches on the same test matches.
predictions = model.predict(X_test)
predictions_yj = yj_model.predict(X_test)

test_comparison = pd.DataFrame({
    "Original": [
        metrics.mean_absolute_error(y_test, predictions),
        metrics.mean_squared_error(y_test, predictions) ** 0.5,
        metrics.r2_score(y_test, predictions)
    ],
    "Yeo–Johnson": [
        metrics.mean_absolute_error(y_test, predictions_yj),
        metrics.mean_squared_error(y_test, predictions_yj) ** 0.5,
        metrics.r2_score(y_test, predictions_yj)
    ]
}, index=["MAE", "RMSE", "R²"])

test_comparison.round(4)


# In[21]:


# Predicting the training mean for every test match.
baseline = [y_train.mean()] * len(y_test)

test_comparison["Mean baseline"] = [
    metrics.mean_absolute_error(y_test, baseline),
    metrics.mean_squared_error(y_test, baseline) ** 0.5,
    metrics.r2_score(y_test, baseline)
]

test_comparison.round(4)


# In[26]:


# Completing the evaluation table.
test_comparison.loc["MSE"] = [
    metrics.mean_squared_error(y_test, predictions),
    metrics.mean_squared_error(y_test, predictions_yj),
    metrics.mean_squared_error(y_test, baseline)
]

# Normalising RMSE using the range of test goal differences.
test_comparison.loc["Normalised RMSE"] = (
    test_comparison.loc["RMSE"] / (y_test.max() - y_test.min())
)

test_comparison.loc[
    ["MAE", "MSE", "RMSE", "Normalised RMSE", "R²"]
].round(4)


# In[24]:


# Comparing actual outcomes with Yeo–Johnson model predictions.
#points closer to the red line represent more accurate prediction
plt.scatter(y_test, predictions_yj)

low = min(y_test.min(), predictions_yj.min())
high = max(y_test.max(), predictions_yj.max())

plt.plot([low, high], [low, high], "r--")

plt.xlabel("Actual goal difference")
plt.ylabel("Predicted goal difference")
plt.title("Actual versus predicted: Yeo–Johnson regression")
plt.tight_layout()
plt.show()


# In[49]:





# In[ ]:




