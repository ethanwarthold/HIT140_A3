import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

import transform as t
import visualise as v

def main():
    df = pd.read_csv("data/Data2.2.csv")
    df = df.drop(columns=["Team", "Opponent"]) # Remove team names

    x = df.iloc[:,:-1] # Eight explanatory variables
    y = df.iloc[:,-1]

    cols = df.columns[:-1]

    # v.corr(df)
    # v.show()
    # v.scatter(x, y, cols)
    # v.show()

    x = t.power_transform(x)

    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    pred = model.predict(x)
    model_details = model.summary()

    print(model_details)

if __name__ == "__main__":
    main()