import os
import logging
import logging.handlers
from fireml.settings import Settings

def setup_logging():
    """
    Configures logging for the entire application, including separate error logs.
    """
    settings = Settings()
    log_dir = settings.log_directory
    os.makedirs(log_dir, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Formatters ---
    formatter_file = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    formatter_console = logging.Formatter('%(levelname)s: %(message)s')

    # --- Console Handler ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter_console)
    logger.addHandler(console_handler)

    # --- Main Log File Handler (Rotating) ---
    log_file_path = os.path.join(log_dir, 'application.log')
    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter_file)
    logger.addHandler(file_handler)

    # --- Error Log File Handler (Rotating) ---
    error_file_path = os.path.join(log_dir, 'error.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_file_path,
        maxBytes=5 * 1024 * 1024, # 5 MB
        backupCount=2,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter_file)
    logger.addHandler(error_handler)
