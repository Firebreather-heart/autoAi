import pandas as pd
import numpy as np
from tkinter.filedialog import askopenfilename
from modelling import serialize

class DataFlow:
    def __init__(self,data:pd.DataFrame,target:str,use_filter: bool =True )-> None:
        """
             data is a pandas.Dataframe object, use_filter is used to specify if
             the filter_redundant_object funtion should be used.
        """
        self.data = data 
        self.use_filter = use_filter
        print(self.data)
        self.target = target
        self.dataTarget = self.data.pop(self.target)
        return None

    def classOrReg(self) :
        decider = len(np.unique(self.dataTarget))
        print(decider,'classes detected')
        if decider > 10:
            return (0,'reg')
        else:
            return (1,'cls',decider)

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
        db = [feature_selector(df,self.dataTarget)[0] for df in db]
        db = [encoding(df) for df in db]
        return db 
if __name__ =='__main__':
    import os,re,time
    from __init__ import confirmInstallLib,__all__
    print('Please keep the internet connetion active\n')
    print('I will be needing a few libraries, including pandas and sklearn, possibly tensorflow and keras\n If you dont have them I would install them for you \n ')
    deepOrNot = input('\n would you prefer i use a deep Model or not, if so i would have to install tensorflow\n Y or N\n').lower()
    deep = True if deepOrNot == 'y' else False
    exreg = re.compile(r'[.].*')
    app_path = os.getcwd()+r'/fire_automl/'

    if os.path.exists(app_path):
        os.system(f'cd {app_path}')
    else:
        try:
            os.system(f'mkdir {app_path} ')
        except Exception:
            print('Seems permission is needed for me to create directories, run this program as admin!')
            print('closing program......')
            os.system('exit')
    ml = __all__['ml']
    dl = __all__['deep']
    for library in ml:
        confirmInstallLib(library)
    if deep == True:
        confirmInstallLib(dl[0])
    
    import pandas as pd        
    print('csv and xlsx files only\n')
    state = True
    while state == True:
        data_dir = askopenfilename()
        extension = exreg.findall(data_dir)
        print(extension)
        ex = ['.csv', '.xlsx']
        if extension == [ex[0]]:
            dataframe = pd.read_csv(data_dir)
            state=False
        elif extension == [ex[1]]:
            dataframe = pd.read_excel(data_dir)
            state = False
        else:
            print('data is not a csv or excel file and cannot be opened\n please try again')

    print(dataframe.keys())

    tstat = True
    while tstat == True:
        try:
            target = input('which of these is your target column:\t')
            y = dataframe[target]
            tstat = False
        except Exception:
            print('could not find the specified target, try again\n')
    dataflow = DataFlow(dataframe,target)
    dataflow.fixData()
    data_list = dataflow.preprocess()
    from sklearn.model_selection import train_test_split
    from modelling import makeRegressors,makeClassifiers
    for data in data_list:
        cor = dataflow.classOrReg()
        X_train,X_val,y_train,y_val = train_test_split(dataflow.data,dataflow.dataTarget, test_size=0.3)
        if cor[0] == 0:
            modelClass = makeRegressors
        elif cor[0] == 1:
            modelClass = makeClassifiers
            no_classes = cor[2]
        _,_,preferredmodel =modelClass(X_train,y_train,X_val,y_val)[0][0]
        preferredmodel.fit(dataflow.data,dataflow.dataTarget)
        serialize(preferredmodel)
        