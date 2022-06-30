import pandas as pd
import numpy as np
from tkinter.filedialog import askopenfilename
from fireml.modelling import serialize
from fireml.deepModel import DeepModel
from fireml.truth import targetTypechecker

info_string =''
class DataFlow:
    def __init__(self,data:pd.DataFrame,target:str,use_filter: bool =True )-> None:
        """
             data is a pandas.Dataframe object, use_filter is used to specify if
             the filter_redundant_object funtion should be used, default is true.
        """
        self.data = data 
        self.use_filter = use_filter
        print(self.data)
        self.target = target
        self.dataTarget = self.data[self.target]
        return None

    def classOrReg(self) :
        decider = len(np.unique(self.dataTarget))
        print(decider,'classes detected')
        info_string.join('Task is a classfication task\n')
        if decider > 10:
            info_string.join('Task is a regression task\n')
            return (0,'reg')
        else:
            dcd = decider -1
            if targetTypechecker(self.dataTarget) is False:
                from sklearn.preprocessing import LabelEncoder
                self.dataTarget = LabelEncoder().fit_transform(self.dataTarget)
                info_string.join('Encoded the labels on the target\n')
            return (1,'cls',dcd)
            

    def fixData(self)-> pd.DataFrame:
        from fireml.fireAutoML import manual_object_fix,manual_missing_NonObject_fix
        self.data = manual_missing_NonObject_fix(data=self.data,aggresive=False)
        self.data = manual_object_fix(self.data)
        self.data.pop(self.target)
        return self.data

    def preprocess(self):

        from fireml.fireAutoML import (
            encoding,dataNorm,feature_selector,
            filterRedundantObject
        )

        self.data1,self.data2,self.data3 = dataNorm(self.data)
        db = [self.data1,self.data2,self.data3]
        if self.use_filter == True:
            db = [filterRedundantObject(df) for df in db]
            info_string.join('Used filter to remmove columns that contain useless data, check stdout for more info\n')
            
        else:
            pass
        db = [feature_selector(df,self.dataTarget)[0] for df in db]
        db = [encoding(df) for df in db]
       
        return db 
if __name__ =='__main__':
    import os,re,sys,logging
    logging.basicConfig(level=logging.INFO,)
    from __init__ import confirmInstallLib,__all__

    logging.info('Please keep the internet connetion active\n')
    logging.info('I will be needing a few libraries, including pandas and sklearn, possibly tensorflow \n If you dont have them I would install them for you \n ')

    deepOrNot = input('\n would you prefer i use a deep Model or not, if so i would have to install tensorflow\n Y or N\n').lower()
    deep = True if deepOrNot == 'y' else False
    exreg = re.compile(r'[.].*')
    app_path = os.getcwd()+r'/fire_automl/'

    if os.path.exists(app_path):
        os.chdir(app_path)
    else:
        try:
            os.system(f'mkdir {app_path} ')
        except Exception:
            print('Seems permission is needed for me to create directories, run this program as admin!')
            print('closing program......')
            sys.exit("failed creating directory")

    ml = __all__['ml']
    dl = __all__['deep']
    for library in ml:
        confirmInstallLib(library)
    if deep == True:
        confirmInstallLib(dl[0])
    
            
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

    dataflow = DataFlow(dataframe,target,use_filter=True)
    dataflow.fixData()
    data_list = dataflow.preprocess()

    from sklearn.model_selection import train_test_split
    from fireml.modelling import makeRegressors,makeClassifiers

    for data in data_list:
        cor = dataflow.classOrReg()
        logging.info('working.........')
        dataflow.data = pd.get_dummies(dataflow.data)
        X_train,X_val,y_train,y_val = train_test_split(data,dataflow.dataTarget, test_size=0.3)
        if cor[0] == 0:
            modelClass = makeRegressors
        elif cor[0] == 1:
            modelClass = makeClassifiers
            no_classes = cor[2]
        _,_,preferredmodel =modelClass(X_train,y_train,X_val,y_val)

        preferredmodel:dict

        usecase = preferredmodel.keys()
        preferredmodel = list(usecase)[0][1]
        preferredmodel.fit(dataflow.data,dataflow.dataTarget)

        logging.info('Here is the final model, you can delete the other serialized models if you wish\n')
        serialize(preferredmodel)

        if deep == True:
            print('proceeding to fit a deep model\n please wait.....')
            if cor[2] > 1:
                multiclass = True 
            if multiclass == True:
                classNo = cor[2] +1
            neuralModel = DeepModel(X_train,y_train,X_val,y_val,task=cor[1], multiclass=classNo)
            neuralModel.rollOver()
        