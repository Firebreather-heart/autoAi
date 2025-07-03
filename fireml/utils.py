import os
import sys

# Get the absolute path to the project root (the directory containing 'fireml')
# This assumes utils.py is inside 'fireml/' and 'fireml/' is directly under the project root.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fireml.setup import logger 

logger.debug("checking")