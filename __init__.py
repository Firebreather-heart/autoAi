import os,sys
try:
    import pandas,numpy,sklearn,xgboost,tensorflow
except:
    pass
#help needed here, if you understand what i'm trying to do help me out
__version__ = '1.00'
__all__={'ml':[pandas,numpy,sklearn,xgboost],'deep':[tensorflow]}
def confirmInstallLib(libname):
    try:
        import libname
    except (ModuleNotFoundError,ImportError):
        try:
            os.system(f'pip install {str(libname)}')
            print(f'trying to install {str(libname)}')
        except Exception as e:
            print(e)
            sys.exit()