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

    """

    x = df.iloc[:,:-1] # Eight explanatory variables
    y = df.iloc[:,-1]

    cols = df.columns[:-1]

    v.corr(df) # show correlations between variables
    v.show()
    v.scatter(x, y, cols) # show graph of goals against each explanatory variable
    v.show()

    x = t.power_transform(x) # Yeo-Johnson transformation

    # statsmodels
    linreg_sm(x, y)

    # sklearn
    linreg_sk(x, y)

if __name__ == "__main__":
    main()