import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer

def power_transform(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        PowerTransformer().fit_transform(x.values),
        index=x.index,
        columns=x.columns
    )

def standardise(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        StandardScaler().fit_transform(x.values),
        index=x.index,
        columns=x.columns
    )

def log_transform(x: pd.DataFrame, col: str, new_col: str = None, drop_old = True) -> pd.DataFrame:
    if new_col is None:
        new_col = col
        drop_old = False

    new = x.copy()
    new[new_col] = new[col].apply(np.log)    

    if drop_old:
        new = new.drop(columns=[col])

    return new