import pandas as pd

import statsmodels.api as sm

def linreg_sm(x: pd.DataFrame, y: pd.DataFrame):
    x = sm.add_constant(x)
    
    model = sm.OLS(y, x).fit()
    pred = model.predict(x)
    model_details = model.summary()

    print(model_details)