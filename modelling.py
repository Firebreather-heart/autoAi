from typing import Any
import pandas as pd
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR 
from sklearn.metrics import classification_report,f1_score
from xgboost import XGBClassifier,XGBRegressor
def makeClassifiers(data:pd.DataFrame,target,testData:pd.DataFrame,testTarget):
    from threading import Thread
    from sklearn.metrics import classification_report
    lr=0.001
    rs = 2000
    nj =-1
    models = [
               (MLPClassifier(early_stopping=False,max_iter=1000,verbose=1,random_state=rs,learning_rate='adaptive',
                            n_iter_no_change=10,hidden_layer_sizes=(100,),
                            warm_start=True),'MLP'),
             ( RandomForestClassifier(n_estimators=500,n_jobs=nj,random_state=rs,warm_start=True,),'RFC'),
              (XGBClassifier(use_label_encoder=False,n_estimators=500,n_jobs=nj,),'XGB'),
           ( AdaBoostClassifier(n_estimators=500,learning_rate=lr,random_state=rs),'ABC'),
              (SVC(probability=True),'SVC'),
            (GaussianNB(),'GNB'),
             ( KNeighborsClassifier(algorithm='brute',n_jobs=nj,leaf_size=60),'KNC'),]
             
    names =[]
    trainedModels = []
    preds=[]
    fs={}
    def runModel(model:tuple(str,object)):
        model =model[1].fit(data,target)
        pred = model[1].predict(testData)
        print(pred)
        print(model[1],'\n',classification_report(pred,testTarget))
        trainedModels.append(model[1])
        preds.append((model[1],pred))
        fs[model[0]]=f1_score(pred,testTarget)
        
        
   
    for mod in models:
        mod_to_thread = Thread(target=runModel(mod))
        mod_to_thread.start()
    fs = sorted(fs.items(),reverse=True)
    from sklearn.ensemble import VotingClassifier 
    voter = VotingClassifier(fs[:3],voting='hard',verbose=True,n_jobs=nj)
    voter.fit(data,target)
    print('voting Classifier','\n',classification_report(voter.predict(testData),testTarget))
    return [trainedModels,preds]