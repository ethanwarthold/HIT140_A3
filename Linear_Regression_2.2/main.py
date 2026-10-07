import pandas as pd

import transform as t
import visualise as v

from linreg_sk import linreg_sk
from linreg_sm import linreg_sm

def main():
    df = pd.read_csv("data/Data2.2.csv")
    df = df.drop(columns=["Team", "Opponent"]) # Remove team names

    """
    Correlations to Goals
    Ordered by magnitude

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
    # df = df.drop(columns=["Win Rate", "Opp Avg Conceded", "Last 5 Avg Scored", "Rest Days"])
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
    v.show()

    x = t.power_transform(x) # Yeo-Johnson transformation

    # show distribution of explanatory variables after the Yeo-Johnson transform
    v.histogram(x, cols)
    v.show()

    # statsmodels
    linreg_sm(x, y)

    # sklearn
    linreg_sk(x, y, 0.4, 0)
    # linreg_sk(x, y, 0.4, 123)
    # linreg_sk(x, y, 0.4, 999)
    # linreg_sk(x, y, 0.4, 60)

if __name__ == "__main__":
    main()