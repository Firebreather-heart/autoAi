import logging
from .utils.logging_config import setup_logging

__version__ = "1.0.0"

# Configure logging when the package is imported
setup_logging()

# Set a null handler to prevent "No handler found" warnings
# if the library is used without a configured logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())