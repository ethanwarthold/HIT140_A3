import math
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def linreg_sk(x: pd.DataFrame, y: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_)

    y_pred = model.predict(X_test)

    df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    print(df_pred)

    y_max = y.max()
    y_min = y.min()

    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    rmse_norm = rmse / (y_max - y_min)
    r_2 = metrics.r2_score(y_test, y_pred)

    # Compare with a base model that uses the mean

    mean = np.mean(y_train)
    y_base = [mean] * len(y_test)

    base_mae = metrics.mean_absolute_error(y_test, y_base)
    base_mse = metrics.mean_squared_error(y_test, y_base)
    base_rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_base))
    base_rmse_norm = base_rmse / (y_max - y_min)
    base_r_2 = metrics.r2_score(y_test, y_base)

    print("Model metrics:")
    print(f"MAE:               {mae:.2f}   |  Baseline: {base_mae:.2f}")
    print(f"MSE:               {mse:.2f}   |  Baseline: {base_mse:.2f}")
    print(f"RMSE:              {rmse:.2f}   |  Baseline: {base_rmse:.2f}")
    print(f"Norm. RMSE:        {rmse_norm:.2f}   |  Baseline: {base_rmse_norm:.2f}")
    print(f"R^2:               {r_2:.3f}  |  Baseline: {base_r_2:.3f}")