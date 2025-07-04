import os
import random
import logging
from fireml.settings import Settings

logger = logging.getLogger(__name__)

def serialize(model, model_name=None, output_dir=None):
    """
    Serialize a model to disk.
    
    Args:
        model: The model to serialize
        model_name: Optional name for the model file
    
    Returns:
        Path to the saved model file
    """
    settings = Settings()
    if output_dir is None:
        output_dir = settings.output_directory
    os.makedirs(output_dir, exist_ok=True)
    
    names = ['charlie', 'gemini', 'lexie', 'lollie']
    if model_name is None:
        model_name = random.choice(names)
    
    filename = os.path.join(output_dir, model_name)
    
    if 'keras' in str(type(model)).lower():
        try:
            import tensorflow as tf
            model_path = filename + '.keras'
            model.save(model_path)
            logger.info(f"Saved Keras model to {model_path}")
            return model_path
        except ImportError:
            logger.error("TensorFlow not found. Please install with 'pip install .[deep_learning]' to save Keras models.")
            raise
        except Exception as e:
            logger.error(f"Failed to save Keras model: {e}")
            raise

    try:
        import joblib
        model_path = filename + '.joblib'
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        return model_path
    except Exception as e:
        logger.warning(f"Failed to save with joblib: {e}, trying pickle.")
        try:
            import pickle
            model_path = filename + '.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model to {model_path}")
            return model_path
        except Exception as e2:
            logger.error(f"Failed to save with pickle: {e2}")
            raise IOError("Could not serialize model with any available method.")

def split_by_sign(iterable:list):
    """
    Takes a given list and splits it into positive and negative values
    """
    s1,s2 = [i for i in iterable if i>0 ],[j for j  in iterable if j<0]
    return s1,s2 

def fill_to_threshold(p:list, n:list, threshold:int):
    """
        return a list filled with values gotten from the highest 
        of the two input lists as dictated by the threshold
    """
    threshold = int(threshold)
    if len(p) + len(n)  < threshold:
        raise ValueError(f"The sum of the lengths of the entries must be equal to or greater than the threshold {threshold}")
    p = sorted(p, reverse=True)
    n = sorted(n, )
    whole = []
    half_t = int(threshold/2)
    if len(p)>= int(threshold/2) and len(n) >= int(threshold/2):
        whole.extend(p[:half_t])
        whole.extend(n[:half_t])
        return whole
    else:
        if len(p) > half_t:
            a = 'p'
        elif len(n) >half_t:
            a = 'n'
        if a == 'p':
            tn = threshold-len(n)
            whole.extend(n)
            whole.extend(p[:tn])
        elif a =='n':
            tp = threshold-len(p)
            whole.extend(p)
            whole.extend(n[:tp])
        assert len(whole) == threshold
        return whole

