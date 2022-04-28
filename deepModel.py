import pandas as pd
from tensorflow.keras import layers,Sequential
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint 
class DeepModel:
    def __init__(self,data:pd.DataFrame,target,task:str ='cls'):
        '''
            DeepModel Class should be fed only with preprocessed data
            the target can be any of pd.series,pd.Dataframe or np.ndarray object
        '''
        self.data = data 
        self.target = target 
        self.task = task
        self.input_shape = [data.shape[1]]
    def createLayer(self,model:Sequential,num_layers:int,layer_width:int):
        activation_func = 'sigmoid' if self.task == 'cls' else 'relu'
        layer_width = layer_width/2
        for i in range(num_layers):
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
    def determinePower(self,data:pd.DataFrame):
        length = data.shape[0]
        