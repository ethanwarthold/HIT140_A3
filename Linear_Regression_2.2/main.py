import pandas as pd

import transform as t
import visualise as v

from linreg_sm import linear_reg_sm
from regression import linear_reg_sk, adaboost_reg, bagging_reg, gradientboost_reg, randomforest_reg

def main():
    df = pd.read_csv("data/Data2.2.csv")
    df = df.drop(columns=["Team", "Opponent"]) # Remove team names

    """
    Correlations to Goals, Ordered by magnitude

    FIFA points                0.38
    Opp FIFA points           -0.34
    Avg Scored                 0.33
    Win Rate                   0.31
    Last 5 Opp Avg Conceded    0.25
    Opp Avg Conceded           0.24
    Last 5 Avg Scored          0.22
    Rest Days                 -0.013
    
    Correlated variables

    FIFA points, Win Rate: 0.8
    Avg Scored, Win Rate: 0.74
    FIFA points, Avg Scored: 0.66
    Avg Scored, Last 5 Avg Scored: 0.65
    Opp Avg Conceded, Last 5 Opp Avg Conceded: 0.63
    FIFA points, Last 5 Avg Scored: 0.62
    """

    # between FIFA points and Win Rate, FIFA points has a higher correlation to Goals
    # So Win Rate is more reasonable to drop
    df = df.drop(columns=["Win Rate"])

    x = df.iloc[:,:-1] # Eight explanatory variables
    y = df.iloc[:,-1]

    cols = df.columns[:-1]

    # show correlations between explanatory variables to identify and remove colinearity
    # v.corr(df)
    # v.show()

    # show graph of goals against each explanatory variable
    # useful to see which are actually correlated with the number of goals scored
    # v.scatter(x, y, cols)
    # v.show()

    # show distribution of explanatory variables
    v.histogram(x, cols)

    x = t.power_transform(x) # Yeo-Johnson transformation

    # show distribution of explanatory variables after the Yeo-Johnson transform
    v.histogram(x, cols)

    v.show()

    # statsmodels - used to test if optimisation techniques are working
    linear_reg_sm(x, y)

    # sklearn
    print("\n=== SPLIT: 0.4 | RANDOM SEED: 0 ===\n")
    linear_reg_sk(x, y, 0.4, 0)
    adaboost_reg(x, y, 0.4, 0)
    bagging_reg(x, y, 0.4, 0)
    gradientboost_reg(x, y, 0.4, 0)
    randomforest_reg(x, y, 0.4, 0)

    # testing different train-test splits and random seeds with different models

    print("\n=== SPLIT: 0.35 | RANDOM SEED: 123 ===\n")
    linear_reg_sk(x, y, 0.35, 123)
    adaboost_reg(x, y, 0.35, 123)
    bagging_reg(x, y, 0.35, 123)
    gradientboost_reg(x, y, 0.35, 123)
    randomforest_reg(x, y, 0.35, 123)
    
    print("\n=== SPLIT: 0.3 | RANDOM SEED: 999 ===\n")
    linear_reg_sk(x, y, 0.3, 999)
    adaboost_reg(x, y, 0.3, 999)
    bagging_reg(x, y, 0.3, 999)
    gradientboost_reg(x, y, 0.3, 999)
    randomforest_reg(x, y, 0.3, 999)

if __name__ == "__main__":
    main()