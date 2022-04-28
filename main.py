import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

class DataFlow:
    def __init__(self,data:pd.DataFrame,use_filter: bool =True )-> None:
        """
             data is a pandas.Dataframe object, use_filter is used to specify if
             the filter_redundant_object funtion should be used.
        """
        self.data = data 
        self.use_filter = use_filter
        print(self.data)
        self.target = input('please specify the target column correctly')
        return None

    def classOrReg(self) -> tuple(int,str):
        dataTarget = self.data.pop(self.target)
        decider = np.unique(dataTarget)
        if decider > 5:
            return (0,'REG')
        else:
            return (1,'CLS')

    def fixData(self)-> pd.DataFrame:
        from fireAutoML import manual_object_fix,manual_missing_NonObject_fix
        self.data = manual_missing_NonObject_fix(data=self.data,aggresive=False)
        self.data = manual_object_fix(self.data)
        return self.data

    def preprocess(self):

        from fireAutoML import (
            encoding,dataNorm,feature_selector,
            filterRedundantObject
        )

        self.data1,self.data2,self.data3 = dataNorm(self.data)
        db = [self.data1,self.data2,self.data3]
        if self.use_filter == True:
            db = [filterRedundantObject(df) for df in db]
        else:
            pass
        db = [feature_selector(df) for df in db]
        db = [encoding(df) for df in db]
        return db 