"""
Data augmentation techniques for various data types.

This module provides functions to augment different types of data
(tabular, text, images) to improve model performance.
"""
import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

logger = logging.getLogger(__name__)

def augment_tabular_data(
    df: pd.DataFrame, 
    target_column: str,
    minority_class: Any = None,
    strategy: str = 'smote',
    sampling_ratio: float = 0.5,
    random_state: int = 42,
    categorical_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Augment tabular data to handle class imbalance.

    Args:
        df: DataFrame to augment
        target_column: Name of the target column
        minority_class: Value of the minority class to oversample
        strategy: Augmentation strategy ('smote', 'adasyn', 'random_oversample', 'mixup')
        sampling_ratio: Target ratio of minority to majority class
        random_state: Random seed for reproducibility
        categorical_columns: Optional list of categorical feature names (for SMOTENC)
        
    Returns:
        Augmented DataFrame
    """
    try:
        from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN, RandomOverSampler
    except ImportError:
        logger.error("imbalanced-learn package required. Please install it via 'pip install imbalanced-learn'")
        return df

    # Extract features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Infer minority class if not provided
    if minority_class is None:
        minority_class = y.value_counts().idxmin()

    logger.info(f"Augmenting minority class '{minority_class}' using {strategy}")

    # Check if enough samples exist
    n_minority = sum(y == minority_class)
    n_majority = sum(y != minority_class)
    if n_minority < 2 and strategy in ['smote', 'adasyn']:
        logger.warning("Too few samples in minority class for SMOTE/ADASYN. Returning original data.")
        return df

    if strategy == 'smote':
        if categorical_columns:
            cat_indices = [X.columns.get_loc(col) for col in categorical_columns if col in X.columns]
            oversampler = SMOTENC(
                categorical_features=cat_indices,
                # sampling_strategy=sampling_ratio,
                random_state=random_state
            )
        else:
            oversampler = SMOTE(
                # sampling_strategy=sampling_ratio,
                random_state=random_state
            )
        X_resampled, y_resampled , *_= oversampler.fit_resample(X, y) 

    elif strategy == 'adasyn':
        oversampler = ADASYN(
            # sampling_strategy=sampling_ratio,
            random_state=random_state
        )
        X_resampled, y_resampled, *_ = oversampler.fit_resample(X, y)

    elif strategy == 'random_oversample':
        oversampler = RandomOverSampler(
            # sampling_strategy=sampling_ratio,
            random_state=random_state
        )
        X_resampled, y_resampled, *_ = oversampler.fit_resample(X, y)

    elif strategy == 'mixup':
        np.random.seed(random_state)
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y != minority_class)[0]

        n_to_generate = int((n_majority - n_minority) * sampling_ratio)
        if n_to_generate <= 0:
            logger.warning("No new samples required using MixUp based on given ratio.")
            return df

        X_additional = []
        y_additional = []

        for _ in range(n_to_generate):
            idx1, idx2 = np.random.choice(minority_indices, 2, replace=True)
            alpha = np.random.beta(0.4, 0.4)
            x_new = alpha * X.iloc[idx1].values + (1 - alpha) * X.iloc[idx2].values
            X_additional.append(x_new)
            y_additional.append(minority_class)

        X_additional_df = pd.DataFrame(X_additional, columns=X.columns)
        X_resampled = pd.concat([X, X_additional_df], ignore_index=True)
        y_resampled = pd.concat([y, pd.Series(y_additional)], ignore_index=True)

    else:
        logger.warning(f"Unknown strategy '{strategy}', returning original data.")
        return df

    # Recombine X and y
    result_df = pd.DataFrame(X_resampled, columns=X.columns)
    result_df[target_column] = y_resampled.reset_index(drop=True)

    logger.info(f"Class distribution after augmentation: {result_df[target_column].value_counts().to_dict()}")
    return result_df

def augment_text_data(df: pd.DataFrame, 
                     text_column: str,
                     target_column: Optional[str] = None,
                     n_samples: int = 1,
                     strategies: List[str] = ['synonym_replacement', 'random_swap'],
                     random_state: int = 42) -> pd.DataFrame:
    """
    Augment text data using various NLP techniques.
    
    Args:
        df: DataFrame containing text data
        text_column: Column containing text to augment
        target_column: Target column (for balanced augmentation)
        n_samples: Number of augmented samples to generate per original
        strategies: List of augmentation strategies to apply
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with augmented text data
    """
    # Set random seed
    np.random.seed(random_state)
    
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        from nltk.tokenize import word_tokenize
        from nltk.corpus import wordnet
    except ImportError:
        logger.error("NLTK package required for text augmentation. Please install with 'pip install .[text]'")
        return df
    
    def synonym_replacement(text: str, replace_fraction: float = 0.2) -> str:
        """Replace words with synonyms"""
        words = word_tokenize(text)
        if not words:
            return text
            
        n_to_replace = max(1, int(len(words) * replace_fraction))
        indices = np.random.choice(len(words), min(n_to_replace, len(words)), replace=False)
        
        for idx in indices:
            word = words[idx]
            synonyms = []
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas(): #type:ignore
                    synonyms.append(lemma.name())
            
            if synonyms:
                words[idx] = np.random.choice(synonyms)
                
        return ' '.join(words)
    
    def random_swap(text: str, n_swaps: int = 2) -> str:
        """Randomly swap words in the text"""
        words = word_tokenize(text)
        if len(words) < 2:
            return text
            
        for _ in range(min(n_swaps, len(words) // 2)):
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
            
        return ' '.join(words)
    
    def random_deletion(text: str, delete_prob: float = 0.1) -> str:
        """Randomly delete words"""
        words = word_tokenize(text)
        if len(words) <= 3:
            return text
            
        result = []
        for word in words:
            if np.random.random() > delete_prob:
                result.append(word)
                
        if not result:  # Ensure at least one word remains
            return words[0]
            
        return ' '.join(result)
    
    def back_translation(text: str) -> str:
        """Translate text to another language and back"""
        try:
            from googletrans import Translator #type:ignore
            translator = Translator()
            
            # Choose a random language
            languages = ['fr', 'es', 'de', 'it']
            lang = np.random.choice(languages)
            
            # Translate to language and back
            intermediate = translator.translate(text, dest=lang).text
            result = translator.translate(intermediate, dest='en').text
            return result
        except ImportError:
            logger.warning("googletrans not found. Please install with 'pip install .[text]' to use back_translation.")
            return text
        except Exception as e:
            logger.warning(f"Back translation failed: {e}")
            return text
    
    # Map strategy names to functions
    strategy_funcs = {
        'synonym_replacement': synonym_replacement,
        'random_swap': random_swap,
        'random_deletion': random_deletion,
        'back_translation': back_translation
    }
    
    # Validate strategies
    valid_strategies = [s for s in strategies if s in strategy_funcs]
    if not valid_strategies:
        logger.warning("No valid augmentation strategies specified. Using original data.")
        return df
    
    logger.info(f"Augmenting text data using strategies: {valid_strategies}")
    
    # Prepare result DataFrame
    result = df.copy()
    
    # If target column specified, augment each class separately
    if target_column:
        for target_value in df[target_column].unique():
            subset = df[df[target_column] == target_value]
            augmented_rows = []
            
            for _, row in subset.iterrows():
                original_text = row[text_column]
                
                for i in range(n_samples):
                    new_row = row.copy()
                    
                    # Apply a random strategy
                    strategy = np.random.choice(valid_strategies)
                    new_text = strategy_funcs[strategy](original_text)
                    new_row[text_column] = new_text
                    
                    augmented_rows.append(new_row)
            
            if augmented_rows:
                result = pd.concat([result, pd.DataFrame(augmented_rows)])
    else:
        # Augment all rows without considering class
        augmented_rows = []
        
        for _, row in df.iterrows():
            original_text = row[text_column]
            
            for i in range(n_samples):
                new_row = row.copy()
                
                # Apply a random strategy
                strategy = np.random.choice(valid_strategies)
                new_text = strategy_funcs[strategy](original_text)
                new_row[text_column] = new_text
                
                augmented_rows.append(new_row)
        
        if augmented_rows:
            result = pd.concat([result, pd.DataFrame(augmented_rows)])
    
    logger.info(f"Increased samples from {len(df)} to {len(result)}")
    return result


def augment_image_data(image_paths: List[str], 
                      transformations: List[str] = ['rotate', 'flip', 'brightness'],
                      intensity: float = 0.5,
                      output_dir: Optional[str] = None,
                      n_samples: int = 1,
                      random_state: int = 42) -> List[str]:
    """
    Augment image data by applying transformations.
    
    Args:
        image_paths: List of paths to image files
        transformations: List of transformations to apply
        intensity: Intensity of transformations (0-1)
        output_dir: Directory to save augmented images
        n_samples: Number of augmented samples per original image
        random_state: Random seed
        
    Returns:
        List of paths to augmented images
    """
    # Set random seed
    np.random.seed(random_state)
    
    try:
        import cv2 #type:ignore
    except ImportError:
        logger.error("OpenCV required for image augmentation. Please install with 'pip install .[image]'")
        return image_paths
    
    if not output_dir and n_samples > 0:
        output_dir = "augmented_images"
        os.makedirs(output_dir, exist_ok=True)
    
    all_image_paths = image_paths.copy()
    
    # Define transformation functions
    def rotate_image(img, angle=None):
        if angle is None:
            angle = np.random.uniform(-30, 30) * intensity
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, rotation_matrix, (width, height))
    
    def flip_image(img, direction=None):
        if direction is None:
            direction = np.random.choice([-1, 0, 1])
        return cv2.flip(img, direction)
    
    def adjust_brightness(img, factor=None):
        if factor is None:
            factor = np.random.uniform(0.7, 1.3) * intensity
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def add_noise(img, noise_type='gaussian'):
        if noise_type == 'gaussian':
            mean = 0
            sigma = np.random.uniform(5, 15) * intensity
            noise = np.random.normal(mean, sigma, img.shape).astype(np.uint8)
            noisy_img = cv2.add(img, noise)
            return noisy_img
        return img
    
    def blur_image(img, kernel_size=None):
        if kernel_size is None:
            kernel_size = int(np.random.choice([3, 5, 7]) * intensity)
            # Ensure kernel_size is odd
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Map transformation names to functions
    transform_funcs = {
        'rotate': rotate_image,
        'flip': flip_image,
        'brightness': adjust_brightness,
        'noise': add_noise,
        'blur': blur_image
    }
    
    # Validate transformations
    valid_transformations = [t for t in transformations if t in transform_funcs]
    if not valid_transformations:
        logger.warning("No valid image transformations specified")
        return image_paths
    
    logger.info(f"Augmenting {len(image_paths)} images with transformations: {valid_transformations}")
    
    for img_path in image_paths:
        try:
            # Read original image
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image {img_path}")
                continue
                
            # Generate augmented images
            for i in range(n_samples):
                # Apply 1-3 random transformations
                n_transforms = np.random.randint(1, min(4, len(valid_transformations) + 1))
                transforms_to_apply = np.random.choice(valid_transformations, n_transforms, replace=False)
                
                # Apply selected transformations
                augmented_img = img.copy()
                for transform in transforms_to_apply:
                    augmented_img = transform_funcs[transform](augmented_img)
                
                # Save augmented image
                if output_dir:
                    base_name = os.path.basename(img_path)
                    name_parts = os.path.splitext(base_name)
                    aug_path = os.path.join(output_dir, f"{name_parts[0]}_aug{i}{name_parts[1]}")
                    cv2.imwrite(aug_path, augmented_img)
                    all_image_paths.append(aug_path)
                    
        except Exception as e:
            logger.error(f"Error augmenting image {img_path}: {str(e)}")
    
    logger.info(f"Generated {len(all_image_paths) - len(image_paths)} augmented images")
    return all_image_paths


def balance_classes(df: pd.DataFrame, 
                   target_column: str,
                   method: str = 'auto',
                   sampling_ratio: float = 1.0,
                   random_state: int = 42) -> pd.DataFrame:
    """
    Balance classes in the dataset using various techniques.
    
    Args:
        df: DataFrame to balance
        target_column: Column containing class labels
        method: Balancing method ('oversample', 'undersample', 'smote', 'adasyn', 'auto')
        sampling_ratio: Target ratio between classes (1.0 for perfect balance)
        random_state: Random seed
        
    Returns:
        Balanced DataFrame
    """
    # Get class distribution
    class_counts = df[target_column].value_counts()
    n_classes = len(class_counts)
    
    logger.info(f"Original class distribution: {class_counts.to_dict()}")
    
    # If only one class or already balanced, return original
    if n_classes <= 1 or (class_counts.max() / class_counts.min() < 1.2):
        logger.info("Classes are already balanced, no action needed")
        return df
    
    # Determine best method if 'auto'
    if method == 'auto':
        minority_count = class_counts.min()
        if minority_count < 10:
            # Very few samples in minority class, use oversampling
            method = 'oversample'
        elif len(df) > 10000 and minority_count < 1000:
            # Large dataset with small minority, use SMOTE
            method = 'smote'
        elif class_counts.max() / class_counts.min() > 10:
            # Severe imbalance, use ADASYN
            method = 'adasyn'
        else:
            # Moderate imbalance, use undersampling
            method = 'undersample'
        
        logger.info(f"Auto-selected balancing method: {method}")
    
    # Apply selected method
    if method == 'oversample':
        # Simple random oversampling
        result = df.copy()
        for cls_value, count in class_counts.items():
            if count < class_counts.max():
                cls_samples = df[df[target_column] == cls_value]
                n_to_sample = int(class_counts.max() * sampling_ratio) - count
                
                if n_to_sample > 0:
                    # Sample with replacement
                    oversamples = cls_samples.sample(n_to_sample, replace=True, random_state=random_state)
                    result = pd.concat([result, oversamples])
        
        logger.info(f"Oversampling: Increased samples from {len(df)} to {len(result)}")
        return result
        
    elif method == 'undersample':
        # Random undersampling
        result = pd.DataFrame()
        target_count = int(class_counts.min() * sampling_ratio)
        
        for cls_value in class_counts.index:
            cls_samples = df[df[target_column] == cls_value]
            
            # Sample without replacement
            if len(cls_samples) > target_count:
                sampled = cls_samples.sample(target_count, random_state=random_state)
            else:
                sampled = cls_samples
                
            result = pd.concat([result, sampled])
        
        logger.info(f"Undersampling: Reduced samples from {len(df)} to {len(result)}")
        return result
        
    elif method in ['smote', 'adasyn']:
        # Use SMOTE or ADASYN via imbalanced-learn
        try:
            if method == 'smote':
                from imblearn.over_sampling import SMOTE
                sampler = SMOTE(sampling_strategy='auto', random_state=random_state)
            else:  # adasyn
                from imblearn.over_sampling import ADASYN
                sampler = ADASYN(sampling_strategy='auto', random_state=random_state)
                
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Handle non-numeric columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                logger.info(f"One-hot encoding {len(categorical_cols)} categorical columns for SMOTE/ADASYN")
                X = pd.get_dummies(X)
            
            X_resampled, y_resampled, *_ = sampler.fit_resample(X, y)
            
            logger.info(f"{method.upper()}: Modified samples from {len(df)} to {len(X_resampled)}")
            
            result = X_resampled.copy()
            result[target_column] = y_resampled
            return pd.DataFrame(result, columns=X.columns.tolist() + [target_column])
            
        except ImportError:
            logger.error(f"imbalanced-learn package required for {method}. Please install with 'pip install .[full]'")
            return df
        except Exception as e:
            logger.error(f"Error applying {method}: {str(e)}")
            logger.info("Falling back to random oversampling")
            # Fall back to oversampling
            return balance_classes(df, target_column, 'oversample', sampling_ratio, random_state)
    else:
        logger.warning(f"Unknown balancing method: {method}")
        return df
