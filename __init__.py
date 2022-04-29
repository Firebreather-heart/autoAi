import os,sys

__version__ = '1.00'
__all__={'ml':['pandas','numpy','sklearn','xgboost'],'deep':['tensorflow']}
def confirmInstallLib(libname:str):
    try:
        import libname
    except (ModuleNotFoundError,ImportError):
        try:
            os.system(f'pip install {libname}')
            print(f'trying to install {libname}')
        except Exception as e:
            print(e)
            sys.exit()