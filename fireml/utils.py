path = 'c:/users/codeworld/desktop/fraudcsv/'
train,test = 'train.csv', 'test.csv'
import pandas as pd 
df_train = pd.read_csv(path+train)
print(df_train)