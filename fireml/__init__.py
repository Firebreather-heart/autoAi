import os,sys, importlib, subprocess

__version__ = '1.1'
REQUIRED_PACKAGES = {
    'ml': ['pandas', 'numpy', 'scikit-learn', 'xgboost'],
    'deep': ['tensorflow']
}

__all__ = ['confirmInstallLib', 'REQUIRED_PACKAGES']

def confirmInstallLib(libname):
    try:
        import_name = 'sklearn' if libname == 'scikit-learn' else libname
        
        importlib.import_module(import_name)
        print(f"{libname} is already installed")
    except ImportError:
        try:
            print(f"Installing {libname}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", libname])
        except Exception as e:
            print(f"Failed to install {libname}: {e}")
            sys.exit(1)