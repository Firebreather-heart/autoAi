import os, logging, pathlib, random
from settings import Settings
from fireml import confirmInstallLib


NAMES = ['charlie', 'gemini', 'lexie', 'lollie']
settings = Settings()

OUTPUT_DIR = settings.output_directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

os.makedirs(settings.log_directory, exist_ok=True)
LOG_DIR = settings.log_directory
log_file = os.path.join(settings.log_directory, 'op-logs.log')
logger = logging.getLogger(log_file)
logger.setLevel(logging.DEBUG)
formatter_console = logging.Formatter('%(levelname)s: %(message)s')
formatter_file = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Only INFO and above to console
console_handler.setFormatter(formatter_console)
logger.addHandler(console_handler)

LOG_FILE_PATH = os.path.join(LOG_DIR, 'application.log')
file_handler = logging.handlers.RotatingFileHandler( #type:ignore
    LOG_FILE_PATH,
    maxBytes=1024 * 1024, # 1 MB
    backupCount=5,      
    encoding='utf-8'     
)
file_handler.setLevel(logging.DEBUG) # Log all levels to file
file_handler.setFormatter(formatter_file)
logger.addHandler(file_handler)

# --- Optional: Error Log File Handler --
error_file_path = os.path.join(LOG_DIR, 'error.log')
error_handler = logging.handlers.RotatingFileHandler( #type:ignore
    error_file_path,
    maxBytes=5 * 1024 * 1024, # 5 MB
    backupCount=2,
    encoding='utf-8'
)
error_handler.setLevel(logging.ERROR) 
error_handler.setFormatter(formatter_file)
logger.addHandler(error_handler)



def serialize(model):
            filename = os.path.join(OUTPUT_DIR, random.choice(NAMES) + '.sav')
            try:
                confirmInstallLib('joblib')
                import joblib 
                joblib.dump(model,filename)
            except ModuleNotFoundError:
                logger.error('failed to install joblib, using pickle instead')
                confirmInstallLib('pickle') 
                import pickle
                with open(filename, 'wb') as f:
                    pickle.dump(model,f)