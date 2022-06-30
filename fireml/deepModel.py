import pandas as pd
from tensorflow.keras import layers,Sequential
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint 
import fireml.mlmath as mlmath
class DeepModel:
    def __init__(self,data:pd.DataFrame,target:pd.Series,testdata:pd.DataFrame,testTarget:pd.Series,task:str='cls',**kwargs):
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
        self.testdata = testdata
        self.testTarget = testTarget
        self.input_shape = [data.shape[1]]
        if self.task == 'cls':
            self.activation ='sigmoid'
            self.output = kwargs.get('multiclass')
            if self.output is None:
                self.outLayer = 1
            else:
                self.outLayer = self.output
                self.activation = 'softmax'
        elif self.task == 'reg':
            self.outLayer = 1
            self.activation = 'relu'
        else:
            raise ValueError('invalid argument passed for task %s'%(task))

    def createLayer(self,model:Sequential,num_layers:int,layer_width:int)->Sequential:
        activation_func =  'relu'
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

    def determinePower(self,data:pd.DataFrame):
        length = data.shape[0]
        layer_width = mlmath.fastForward(length)
        num_layers = mlmath.breakDown(layer_width)
        return (layer_width,num_layers)

    def makeCompileModel(self,):
        layer_width,num_layers = self.determinePower(self.data)
        deepModel = Sequential([
            layers.Dense(layer_width,activation='relu',input_shape=self.input_shape,),
            layers.Dropout(0.05)
        ])
        metric_used = self.task
        deepModel = self.createLayer(deepModel,num_layers,layer_width)
        deepModel = deepModel.add(layers.Dense(self.outLayer,activation=self.activation))  

        loss = 'binary_crossentropy' if metric_used == 'cls' else 'mse'
        metrics=['binary_accuracy'] if metric_used == 'cls' else 'mae'

        if self.outLayer > 1 :
            metrics = ['accuracy']
            loss = 'categorical_crossentropy'

        deepModel.compile(
            optimizer='adam',
            loss=loss,
            metrics=metrics
               )      
        self.deepModel = deepModel
        return deepModel
        
    def trainModel(self):
        callback = EarlyStopping(
                patience=100,
                min_delta=0.001,
                restore_best_weights=True)
        history = self.deepModel.fit(self.data,self.target,
                        validation_split =0.3,
                         batch_size=50,
                        epochs =1000,
                        callbacks=[callback,ModelCheckpoint('runtimemodels.sav', 
                        verbose=1, 
                        save_best_only=True,mode= max)],
                        verbose=1
                        )
        print('Evaluating deep Model\n')
        self.deepModel.evaluate(self.testdata,self.testTarget)
        self.deepModel.save('savedmodel.sav')
        return history

    def rollOver(self):
        self.makeCompileModel()
        self.trainModel()
        return self.deepModel
    