import pandas as pd
from tensorflow.keras import layers,Sequential
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint 
import mlmath
class DeepModel:
    def __init__(self,data:pd.DataFrame,target,task:str='cls',**kwargs:dict(str,int)):
        '''
            DeepModel Class should be fed only with preprocessed data
            the target can be any of pd.series,pd.Dataframe or np.ndarray object
            
            task is a string `cls` for classification tasks, or `reg` for regression tasks
            
            if task is a multiclass classification task, then pass in the arguments
            multiclass, and the number of classes, if not specified, default is a single class

        Example:
        from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
        >>> DeepModel(X,y,task ='cls',multiclass =1)
        '''
        self.data = data 
        self.target = target 
        self.task = task
        self.input_shape = [data.shape[1]]
        if self.task == 'cls':
            self.activation ='sigmoid'
            self.output = kwargs.get('multiclass')
            if self.output is None:
                self.outLayer = 1
            else:
                self.outLayer = self.output
        elif self.task == 'reg':
            self.outLayer =1
            self.activation = 'relu'
        else:
            raise ValueError('invalid argument passed for task %s'%(task))
    def createLayer(self,model:Sequential,num_layers:int,layer_width:int)->Sequential:
        activation_func = 'sigmoid' if self.task == 'cls' else 'relu'
        layer_width = layer_width/2
        for _ in range(num_layers):
            layer_width =int(layer_width)
            model.add(
                layers.Dense(layer_width/2,activation=activation_func),
                layers.Dropout(0.05,)
            )
            layer_width /= 2
            num_layers -=1
            if num_layers <= 1:
                break 
        return model 
    def determinePower(self,data:pd.DataFrame)-> tuple(int,int):
        length = data.shape[0]
        layer_width = mlmath.fastForward(length)
        num_layers = mlmath.breakDown(layer_width)
        return (layer_width,num_layers)
    def makeCompileModel(self,data:pd.DataFrame,):
        layer_width,num_layers = self.determinePower(self.data)
        deepModel = Sequential([
            layers.Dense(layer_width,activation='relu',input_shape=self.input_shape,),
            layers.Dropout(0.05)
        ])
        deepModel = self.createLayer(deepModel,num_layers,layer_width)
        deepModel = deepModel.add(layers.Dense(self.outLayer,activation=self.activation))        
