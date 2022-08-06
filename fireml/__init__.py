import os,sys

#help needed here, if you understand what i'm trying to do help me out
__version__ = '1.00'
__all__={'ml':['pandas','numpy','sklearn','xgboost'],'deep':['tensorflow']}

def confirmInstallLib(libname):
    try:
        assert libname
    except (ModuleNotFoundError,ImportError,AssertionError):
        try:
            os.system(f'pip install {libname}')
            print(f'trying to install {libname}')
        except Exception as e:
            print(e)
            sys.exit()