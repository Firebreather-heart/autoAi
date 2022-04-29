import os,time
import autoImputer
import pandas as pd 
import numpy as np
import mlmath


classdata = pd.read_csv(r'C:\Users\codeworld\diabetes.csv')

def dataNorm(data:pd.DataFrame, ) -> pd.DataFrame:
    '''
        Returns a list of pandas datframe objects containing three categories of data,
        standard scaled data, MinMax scaled and Normalised data. This would allow the 
        flexibility of choosing models trained on the three data forms 
    '''
    #normalises the given data
    print('Normalising data..............\n')
    try: 
        from sklearn.preprocessing import StandardScaler,MinMaxScaler,Normalizer
    except ModuleNotFoundError:
        try:
            os.system('pip install sklearn')
        except Exception as e:
            print(f'{e}\n could not install sklearn library, try to activate your internet connection and try again\ closing program........')
            time.sleep(3)
            os.system('exit')
    stand = StandardScaler()
    minmax = MinMaxScaler(feature_range=(0,1))
    norm = Normalizer()
    data_to_normalise = data.select_dtypes(exclude=object)
    transformed_array = [stand.fit_transform(data_to_normalise),minmax.fit_transform(data_to_normalise),norm.fit_transform(data_to_normalise)]
    
    return [pd.DataFrame(i,columns=data.select_dtypes(exclude=object).columns) for i in transformed_array]

def encoding(data:pd.DataFrame,):
    '''
    An improvement on the former encoding function
    CAUTION!!! this must be used only after the feature_selector
    has been used, that is if you wish to use the feature selector
    '''
    return pd.get_dummies(data)

def feature_selector(data:pd.DataFrame,target):
    '''
        This will select the best features from the given
        dataset.
        Params:
            data is any pandas Dataframe Object
    '''
    print('...selecting best features.......')
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVR
    dat = data.copy()
    y=target
    X=dat
    model = SVR(kernel='linear')
    rfe = RFE(model,n_features_to_select=int(len(data.columns)*0.5),step=1)
    selections=rfe.fit(X,y)
    selections = selections.get_feature_names_out()
    return X[selections],y


def filterRedundantObject(data:pd.DataFrame,):
    '''
    used to remove columns with too many variable
    the logic is that such a column might contain a lot of irrelevant 
    features
    Params:
            data is any pandas Dataframe Object
    '''
    for column in data.select_dtypes(include=object).columns:
        if set(i for i in data[column].values).__len__() >50:
            data = data.drop(column,axis='columns')
        
    return data
def generateModel(data: pd.DataFrame,target,mode:int):
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=42)
    try:
        import xgboost
    except ModuleNotFoundError:
        try:
            os.system('pip install xgboost')
            import xgboost
        except Exception:
            pass
    if mode == 0:
        from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import KFold
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import classification_report,accuracy_score,roc_auc_score

        booster = xgboost.XGBClassifier()
        rfc = RandomForestClassifier()
        abcf = AdaBoostClassifier()
        gnb = GaussianNB()
        supVect = SVC()
        dtc = DecisionTreeClassifier()
        knc =KNeighborsClassifier()
        kfold = KFold(n_splits=10,)
        models  = [('BO',booster),('RFC',rfc),('ADA',abcf),('GNB',gnb),
                     ('SVC',supVect),
                     ('DC',dtc),('KNC',knc),
                      ]
        results = []
        names =[]
        for name,model in models:
            cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring='accuracy')
            results.append(cv_results.mean())
            names.append(name)
            print(f'{name}: {cv_results.mean()}   |   {cv_results.std()}')
        topModel = results.index(max(results))
        topModel = names[topModel]
        

        print(f'\n{topModel} is the best performing right now in the crossvalidation take note of it')
        print('\n now testing the model .......\n')
        algorithms =[]
        f_s=[]
        for name,model in models:
            alg = model.fit(X_train,y_train)
            algorithms.append(alg)
            pred = alg.predict(X_test)
            f_s.append((name,roc_auc_score(pred,y_test)))
            print(name,classification_report(pred,y_test))
        top2 = algorithms[f_s.index(max(f_s))]
        def serialize(model):
            filename = input('input your desired model name\n')+'.sav'
            try:
                import joblib 
                joblib.dump(model,filename)
            except ModuleNotFoundError:
                try:
                    os.system('pip install joblib')
                    import joblib 
                    joblib.dump(model,filename)
                except Exception:
                    print('failed to install joblib, using pickle instead')
                    import pickle 
                    pickle.dump(model,filename)


        if top2 == topModel:
            print(f'{top2} is the best performing model, you can tune it for better performances \n saving model in current directory\n ')
            serialize(top2)
        else:
            print(f'{top2} and {topModel} are top performers,\n saving the models in the current directory')
            serialize(top2)
            serialize(topModel)


    try:
        print('trying out a deep model...\n')
        from tensorflow.keras import layers 
        from tensorflow.keras import Sequential
        from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

        cs = [X_train.shape[1]]
        deepModel = Sequential([
            layers.Dense(512, activation='relu',input_shape=cs),
            layers.Dropout(0.05, ),

            layers.Dense(256, activation='relu'),
            layers.Dropout(0.05),

            layers.Dense(64,activation='relu'),
            layers.Dropout(0.05),

            layers.Dense(1,activation='sigmoid'),

        ])
        callback = EarlyStopping(
        patience=100,
        min_delta=0.001,
        restore_best_weights=True)

        deepModel.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['binary_accuracy']
               )
        history = deepModel.fit(
        X_train,y_train,
        validation_split=0.3,
        batch_size=512,
        epochs =1000,
        callbacks=[callback,ModelCheckpoint('runtimemodels.sav' , verbose=1, save_best_only=True,mode= max)],
         verbose=1
         )
         
        preds=deepModel.predict(X_test)
        preds = [mlmath.approximator(i) for i in preds]
        print(classification_report(preds,y_test))
        print('this is your deep model\n')
        deepModel.save(input('input model name\n')+'.sav')
    except ModuleNotFoundError:
        print('could not use deep model....\n')
        pass





def manual_missing_NonObject_fix(data: pd.DataFrame,target=None,aggresive: bool=True) -> pd.DataFrame:
    '''
        This will fix the non-categorical datafields by filling the missing values
        if the missing values are more than half of the whole column then the whole 
        column will be forfeited. You can disable this behaviour by setting aggressive to False 
        Params:
            data is any pandas Dataframe Object
    '''
    #fix missing values 
    data = data.drop_duplicates()
    v = []
    for i in data.select_dtypes(exclude=object).columns:
        v.append(data[i].isnull().sum())
    for j in zip(data.select_dtypes(exclude=object).columns,v):
        if j[1] >0:
            if j[1] > 0.5*data[j[0]].shape[0]:
                if aggresive == True:
                    data = data.drop(j[0],axis='columns')
                else:
                    decision = input(f'[.] ..The column {j[0]} has {j[1]} missing values out of {data[j[0]].shape[1]}, its advisable to drop it\n Reply yes or no\n').lower()
                    if decision == 'yes':
                        data = data.drop(j[0],axis='columns')
                    else:
                        print("Should i fill the the null values with the mean, median, or zero..\n or would you like to type in a code for me to evaluate\n")
                        resp = input('input any of the following, mean, median, zero or code:\t').lower()
                        if resp == 'mean':
                            data[j[0]]=autoImputer.impute(data[j[0]],method=resp)
                        elif resp == 'median':
                            data[j[0]]=autoImputer.impute(data[j[0]],method=resp)
                        elif resp == 'zero':
                            data[j[0]]=data[j[0]].fillna(0)
                        elif resp == 'code':
                            try :
                                data[j[0]] = eval(input("type in your code please \n dataframe name in the namespace is data[j[0]] \n "))
                            except Exception as e:
                                print(e)
            else:
                data = data.dropna(axis='rows')
    return data
            
def manual_object_fix(data:pd.DataFrame,target = None) -> pd.DataFrame:
    '''
    This would remove columns or rows containing categorical data that can't be imputed
    avoid it if you have better means of fixing the missing values
    Params:
            data is any pandas Dataframe Object
    '''
    #fix missing values
    print('fixing missing values.....\n')
    data = data.drop_duplicates()
    v=[]
    for i in data.select_dtypes(include=object).columns:
        v.append(data[i].isnull().sum())
    for j in zip(data.columns,v):
         if j[1]>0 and j[1] > 0.3*len(data[j[0]]):
            data =data.drop(j[0],axis ='columns')
         elif j[1]>0 and j[1] < 0.3*len(data[j[0]]):
            data = data.dropna(axis='rows')
    print('..missing values fixed\n')
    return data






class Process:
    def __init__(self,data:pd.DataFrame,target):
        self.target =data.pop(target)
        self.data = data 
    def direction():
        pass
    def rollOver(self):
        self.data =manual_missing_NonObject_fix(self.data)
        self.data=manual_object_fix(self.data)
        data1,data2,data3 =dataNorm(self.data)
        databank =[data1,data2,data3]
        for i in databank:
            i=filterRedundantObject(i)
            i=encoding(i)
            i,targ =feature_selector(i,self.target)
            print(i)
            generateModel(i,targ,mode=0)

Process(classdata,'Outcome').rollOver()      