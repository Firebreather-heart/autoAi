import pandas as pd
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR 
from sklearn.metrics import f1_score,mean_absolute_error
from xgboost import XGBClassifier,XGBRegressor
import os
def serialize(model):
            print(model)
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

def makeClassifiers(data:pd.DataFrame,target,testData:pd.DataFrame,testTarget, voting:str = 'hard'):
    from threading import Thread
    from sklearn.metrics import classification_report
    lr=0.001
    rs = 2000
    nj =-1
    models = [
               (MLPClassifier(early_stopping=False,max_iter=1000,verbose=0,random_state=rs,learning_rate='adaptive',
                            n_iter_no_change=10,hidden_layer_sizes=(100,),
                            warm_start=True),'MLP'),
             ( RandomForestClassifier(n_estimators=500,n_jobs=nj,random_state=rs,warm_start=True,),'RFC'),
              (XGBClassifier(use_label_encoder=False,n_estimators=500,n_jobs=nj,),'XGB'),
           ( AdaBoostClassifier(n_estimators=500,learning_rate=lr,random_state=rs),'ABC'),
              (SVC(probability=True),'SVC'),
            (GaussianNB(),'GNB'),
             ( KNeighborsClassifier(algorithm='brute',n_jobs=nj,leaf_size=60),'KNC'),]
             
    trainedModels = []
    preds=[]
    fs={}
    def runModel(model):
        name =model[1]
        model =model[0].fit(data,target)
        pred = model.predict(testData)
        print(model,'\n',classification_report(pred,testTarget))
        trainedModels.append((name,model))
        preds.append((model,pred))
        fs[(name,model)]=f1_score(pred,testTarget)
        
        
    threads =[]
    for mod in models:
        mod_to_thread = Thread(target=runModel(mod))
        threads.append(mod_to_thread)
        mod_to_thread.start()
    for t in threads:t.join()
    

    fs = dict(sorted(fs.items(),key=lambda x: x[1],reverse=True))

    selectedTrainedSet = list(set(trainedModels).
    intersection(set(
        list(fs.keys())[:3])
        )
    )
    from sklearn.ensemble import VotingClassifier 
    voter = VotingClassifier(selectedTrainedSet,voting=voting,verbose=True,n_jobs=nj)
    voter.fit(data,target)
    trainedModels.append(voter)

    for mo in selectedTrainedSet:serialize(mo)

    voterPred=voter.predict(testData)
    preds.append(('Voting Classifier',voterPred))
    print('voting Classifier','\n',classification_report(voterPred,testTarget))
    fs[('voter',voter)]= f1_score(voterPred,testTarget)
    fs = sorted(fs.items(),key=lambda x: x[1],reverse=True)
    fs = dict(fs)
    print(f'{fs[0][0]} appears to be the best performer with f1 score of {fs[1]}')
    serialize(voter)
    return trainedModels,preds,fs

def makeRegressors(data:pd.DataFrame,target,testData:pd.DataFrame,testTarget,):
    from threading import Thread
    lr=0.001
    rs = 2000
    nj =-1
    models = [
               (MLPRegressor(early_stopping=False,max_iter=1000,verbose=0,random_state=rs,learning_rate='adaptive',
                            n_iter_no_change=10,hidden_layer_sizes=(100,),
                            warm_start=True),'MLP'),
             ( RandomForestRegressor(n_estimators=500,n_jobs=nj,random_state=rs,warm_start=True,),'RFR'),
              (XGBRegressor(use_label_encoder=False,n_estimators=500,n_jobs=nj,),'XGB'),
           ( AdaBoostRegressor(n_estimators=500,learning_rate=lr,random_state=rs),'ABR'),
              (SVR(),'SVR'),
             ( KNeighborsRegressor(algorithm='brute',n_jobs=nj,leaf_size=60),'KNR'),
             ]
             
    trainedModels = []
    preds=[]
    fs={}
    def runModel(model):
        name =model[1]
        model =model[0].fit(data,target)
        pred = model.predict(testData)
        print(pred)
        print(model,'error \n',mean_absolute_error(testTarget,pred))
        trainedModels.append((name,model))
        preds.append((model,pred))
        fs[(name,model)]=mean_absolute_error(pred,testTarget)
        
        
   
    for mod in models:
        mod_to_thread = Thread(target=runModel(mod))
        mod_to_thread.start()

    fs = dict(sorted(fs.items(),key=lambda x: x[1],reverse=True))

    selectedTrainedSet = list(set(trainedModels).
    intersection(set(
        list(fs.keys())[:3])
        )
    )

    from sklearn.ensemble import VotingRegressor
    voter = VotingRegressor(selectedTrainedSet,verbose=True,n_jobs=nj)
    voter.fit(data,target)

    for mo in selectedTrainedSet:serialize(mo)

    voterPred=voter.predict(testData)
    preds.append(('Voting Regressor',voterPred))
    print('voting Regressor error','\n',mean_absolute_error(voter.predict(testData),testTarget))
    fs[('voter',voter)] = mean_absolute_error(voterPred,testTarget)
    fs = sorted(fs.items(),key=lambda x: x[1],)
    print(f'{fs[0][0]} appears to be the best performer with error of {fs[0][1]}')
    serialize(voter)
    fs=dict(fs)
    return trainedModels,preds,fs