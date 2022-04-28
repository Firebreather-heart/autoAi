from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
def impute(data:pd.DataFrame,method:str):
    imr = IterativeImputer(initial_strategy=method)
    return imr.fit_transform(data.values)