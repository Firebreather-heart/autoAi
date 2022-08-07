import logging
import os
import time
import autoImputer
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO,)
classdata = pd.read_csv(r'C:\Users\codeworld\diabetes.csv')


def check_corr(main, scd):
    return main.corr(scd)


def dataNorm(data: pd.DataFrame, ) -> pd.DataFrame:
    '''
        Returns a list of pandas datframe objects containing three categories of data,
        standard scaled data, MinMax scaled and Normalised data. This would allow the 
        flexibility of choosing models trained on the three data forms 
    '''
    # normalises the given data
    print('Normalising data..............\n')
    try:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
    except ModuleNotFoundError:
        try:
            os.system('pip install sklearn')
        except Exception as e:
            print(f'{e}\n could not install sklearn library, try to activate your internet connection and try again\ closing program........')
            time.sleep(3)
            os.system('exit')
    stand = StandardScaler()
    minmax = MinMaxScaler(feature_range=(0, 1))
    norm = Normalizer()
    data_to_normalise = data.select_dtypes(exclude=object)
    transformed_array = [stand.fit_transform(data_to_normalise), minmax.fit_transform(
        data_to_normalise), norm.fit_transform(data_to_normalise)]

    return [pd.DataFrame(i, columns=data.select_dtypes(exclude=object).columns) for i in transformed_array]


def encoding(data: pd.DataFrame,):
    '''
    An improvement on the former encoding function
    CAUTION!!! this must be used only after the feature_selector
    has been used, that is if you wish to use the feature selector
    '''
    data = pd.get_dummies(data)
    print(data)
    return data


def feature_selector(data: pd.DataFrame, target: pd.Series, style='corr_check'):
    '''
        This will select the best features from the given
        dataset.
        Params:
            data is any pandas Dataframe Object
            target is a pandas series object
            stle: defaults to corr_check
            the corr_check style will select the features with high
            negative or positive correlation.
            the other option is the svr_model
    '''
    from utils import split_by_sign, fill_to_threshold
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVR
    dat = data.copy()
    dat = pd.get_dummies(dat)
    y = target
    X = dat
    print('...selecting best features.......')
    if len(data.columns) < 11:
        return data, target
    else:
        if style == 'svr_model':
            model = SVR(kernel='linear')
            rfe = RFE(model, n_features_to_select=int(
                len(data.columns)*0.5), step=1)
            selections = rfe.fit(X, y)
            selections = selections.get_feature_names_out()
            return X[selections], y

        elif style == 'corr_check':
            buc = dict()
            for i in dat.columns:
                a = check_corr(y, dat[i])
                buc[a] = i
            bucp, bucn = split_by_sign(buc)
            whole = fill_to_threshold(
                bucp, bucn, threshold=len(data.columns)*0.5)
            sel_col = []
            for val in whole:
                sel_col.append(buc.get(val))
            return X[sel_col], y


def filterRedundantObject(data: pd.DataFrame,):
    '''
    used to remove columns with too many variable
    the logic is that such a column might contain a lot of irrelevant 
    features
    Params:
            data is any pandas Dataframe Object
    '''
    for column in data.select_dtypes(include=object).columns:
        if set(i for i in data[column].values).__len__() > 50:
            data = data.drop(column, axis='columns')
            print(f'dropping.......{column}')

    return data


def manual_missing_NonObject_fix(data: pd.DataFrame, target=None, aggresive: bool = True) -> pd.DataFrame:
    '''
        This will fix the non-categorical datafields by filling the missing values
        if the missing values are more than half of the whole column then the whole 
        column will be forfeited. You can disable this behaviour by setting aggressive to False 
        Params:
            data is any pandas Dataframe Object
    '''
    # fix missing values
    data = data.drop_duplicates()
    v = []
    for i in data.select_dtypes(exclude=object).columns:
        v.append(data[i].isnull().sum())
    for j in zip(data.select_dtypes(exclude=object).columns, v):
        if j[1] > 0:
            if j[1] > 0.5*data[j[0]].shape[0]:
                if aggresive == True:
                    data = data.drop(j[0], axis='columns')
                    print(f'dropping.......{j[0]}')
                else:
                    decision = 'yes' if aggresive else input(
                        f'[.] ..The column {j[0]} has {j[1]} missing values out of {data[j[0]].shape[1]}, its advisable to drop it\n Reply yes or no\n').lower()
                    if decision == 'yes':
                        data = data.drop(j[0], axis='columns')
                        print(f'dropping.......{j[0]}')
                    else:
                        print(
                            "Should i fill the the null values with the mean, median, or zero..\n or would you like to type in a code for me to evaluate\n")
                        resp = input(
                            'input any of the following, mean, median, zero or code:\t').lower()
                        if resp == 'mean':
                            data[j[0]] = autoImputer.impute(
                                data[j[0]], method=resp)
                        elif resp == 'median':
                            data[j[0]] = autoImputer.impute(
                                data[j[0]], method=resp)
                        elif resp == 'zero':
                            data[j[0]] = data[j[0]].fillna(0)
                        elif resp == 'code':
                            try:
                                data[j[0]] = eval(input(
                                    "type in your code please \n dataframe name in the namespace is data[j[0]] \n "))
                            except Exception as e:
                                print(e)
            else:
                data = data.dropna(axis='rows')
    return data


def manual_object_fix(data: pd.DataFrame, target=None) -> pd.DataFrame:
    '''
    This would remove columns or rows containing categorical data that can't be imputed
    avoid it if you have better means of fixing the missing values
    Params:
            data is any pandas Dataframe Object
    '''
    # fix missing values
    logging.info('fixing missing values.....\n')
    data = data.drop_duplicates()
    v = []
    for i in data.select_dtypes(include=object).columns:
        v.append(data[i].isnull().sum())
    for j in zip(data.columns, v):
        if j[1] > 0 and j[1] > 0.3*len(data[j[0]]):
            data = data.drop(j[0], axis='columns')
            print(f'dropping.......{j[0]}')
        elif j[1] > 0 and j[1] < 0.3*len(data[j[0]]):
            data = data.dropna(axis='rows')
    logging.info('..missing values fixed\n')
    print(data)
    return data
