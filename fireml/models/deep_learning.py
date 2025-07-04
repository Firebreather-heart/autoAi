import os
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Literal
from fireml.settings import Settings
from tensorflow.keras import layers, Sequential, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


class DeepModel:
    def __init__(self, 
                 data: pd.DataFrame, 
                 target: pd.Series, 
                 testdata: pd.DataFrame, 
                 testTarget: pd.Series, 
                 task: str = 'cls', 
                 **kwargs):
        """
        Initializes the DeepModel class for building and training deep learning models.

        Args:
            data: Training data (preprocessed).
            target: Training target values.
            testdata: Test data (preprocessed).
            testTarget: Test target values.
            task: Task type ('cls' for classification, 'reg' for regression).
            kwargs: Additional configuration parameters:
                - multiclass: Number of classes for classification (int)
                - complexity: Model complexity ('low', 'medium', 'high', 'auto')
                - dropout_rate: Dropout rate for regularization
                - regularization: L1/L2 regularization strength
                - batch_size: Training batch size
                - patience: Early stopping patience
                - learning_rate: Initial learning rate
                - model_name: Name for saving the model
        """
        self.data = data
        self.target = target
        self.testdata = testdata
        self.testTarget = testTarget
        self.task = task
        self.input_shape = [data.shape[1]]
        self.config = self._load_configuration(kwargs)
        self.model = None
        self.settings = Settings()
        
        self._initialize_task(kwargs)
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories for model outputs."""
        output_dir = self.settings.output_directory
        self.model_dir = os.path.join(output_dir, 'models')
        self.log_dir = os.path.join(self.settings.log_directory, 'tensorboard_logs')
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _load_configuration(self, kwargs: Dict) -> Dict:
        """Load model configuration from kwargs with sensible defaults."""
        return {
            'complexity': kwargs.get('complexity', 'auto'),
            'dropout_rate': kwargs.get('dropout_rate', 0.2),
            'l2_regularization': kwargs.get('regularization', 0.001),
            'batch_size': kwargs.get('batch_size', 32),
            'patience': kwargs.get('patience', 20),
            'min_delta': kwargs.get('min_delta', 0.001),
            'max_epochs': kwargs.get('max_epochs', 500),
            'validation_split': kwargs.get('validation_split', 0.2),
            'learning_rate': kwargs.get('learning_rate', 0.001),
            'model_name': kwargs.get('model_name', 'deep_model')
        }

    def _initialize_task(self, kwargs):
        """Initializes task-specific attributes."""
        if self.task == 'cls':
            self.activation = 'sigmoid'
            self.outLayer = kwargs.get('multiclass', 1)
            if self.outLayer > 1:
                self.activation = 'softmax'
                # Set appropriate loss for multi-class problems
                self.loss = 'categorical_crossentropy'
                self.metrics = ['accuracy']
            else:
                self.loss = 'binary_crossentropy'
                self.metrics = ['accuracy', 'AUC']
        elif self.task == 'reg':
            self.outLayer = 1
            self.activation = 'linear'  # Linear is better for regression outputs
            self.loss = 'mse'
            self.metrics = ['mae', 'mse']
        else:
            raise ValueError(f"Invalid argument passed for task: {self.task}")

    def determine_complexity(self) -> Literal['low', 'medium', 'high']:
        """
        Determine model complexity based on data characteristics.
        
        Considers:
        1. Data size (rows × columns)
        2. Feature count
        3. Target complexity (unique values for classification)
        4. Class imbalance (for classification)
        
        Returns:
            Complexity level: 'low', 'medium', or 'high'
        """
        if self.config['complexity'] != 'auto':
            return self.config['complexity']
        
        # Consider dataset size
        data_size = self.data.shape[0] * self.data.shape[1]
        feature_count = self.data.shape[1]
        
        # Check for class imbalance in classification tasks
        if self.task == 'cls':
            if self.outLayer > 1:  # Multi-class
                # Count occurrences of each class
                value_counts = self.target.value_counts()
                min_class_size = value_counts.min()
                max_class_size = value_counts.max()
                imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float('inf')
                class_complexity = self.outLayer / 2  # More classes = more complex
            else:  # Binary
                value_counts = self.target.value_counts()
                if len(value_counts) == 2:
                    minority = min(value_counts)
                    majority = max(value_counts)
                    imbalance_ratio = majority / minority if minority > 0 else float('inf')
                else:
                    imbalance_ratio = 1.0
                class_complexity = 1.0
        else:  # Regression
            # For regression, consider target variance as complexity indicator
            target_variance = np.var(self.target)
            target_mean = np.mean(self.target)
            relative_variance = target_variance / (target_mean**2) if target_mean != 0 else target_variance
            class_complexity = min(5.0, relative_variance * 10)
            imbalance_ratio = 1.0
        
        # Calculate complexity score
        complexity_score = 0
        
        # Data size factor (0-3 points)
        if data_size < 1000:
            complexity_score += 0
        elif data_size < 10000:
            complexity_score += 1
        elif data_size < 100000:
            complexity_score += 2
        else:
            complexity_score += 3
            
        # Feature count factor (0-3 points)
        if feature_count < 10:
            complexity_score += 0
        elif feature_count < 50:
            complexity_score += 1
        elif feature_count < 100:
            complexity_score += 2
        else:
            complexity_score += 3
            
        # Class complexity factor (0-3 points)
        complexity_score += min(3, class_complexity)
        
        # Imbalance factor (0-2 points)
        if imbalance_ratio < 1.5:
            complexity_score += 0
        elif imbalance_ratio < 5:
            complexity_score += 1
        else:
            complexity_score += 2
            
        # Determine complexity level
        if complexity_score <= 4:
            return 'low'
        elif complexity_score <= 8:
            return 'medium'
        else:
            return 'high'

    def design_architecture(self) -> List[int]:
        """
        Design neural network architecture based on problem complexity.
        
        Returns:
            List of integers representing layer sizes
        """
        complexity = self.determine_complexity()
        feature_count = self.data.shape[1]
        
        # Base architecture size based on feature count
        if feature_count <= 10:
            base_width = 16
        elif feature_count <= 50:
            base_width = 32
        elif feature_count <= 100:
            base_width = 64
        else:
            base_width = 128
            
        # Design architecture based on complexity
        if complexity == 'low':
            # Simple architecture for low complexity
            # One hidden layer is often sufficient
            return [base_width]
            
        elif complexity == 'medium':
            # Medium complexity with decreasing width
            return [base_width, base_width // 2]
            
        else:  # high complexity
            # Deep architecture for complex problems
            # Start wide and gradually decrease
            return [base_width * 2, base_width, base_width // 2, base_width // 4]

    def create_layers(self, model: Sequential, layer_sizes: List[int]) -> Sequential:
        """
        Add layers to model based on architecture design.
        
        Args:
            model: Sequential model to add layers to
            layer_sizes: List of integers representing neurons in each layer
            
        Returns:
            Updated Sequential model
        """
        dropout_rate = self.config['dropout_rate']
        l2_reg = self.config['l2_regularization']
        
        for units in layer_sizes:
            model.add(layers.Dense(
                units=units,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_reg)
            ))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
            
        return model

    def compile_model(self) -> Sequential:
        """Creates and compiles the deep learning model based on problem complexity."""
        # Design architecture based on problem complexity
        layer_sizes = self.design_architecture()
        
        # Create initial model with input layer
        model = Sequential()
        model.add(layers.Dense(
            layer_sizes[0], 
            activation='relu',
            kernel_regularizer=regularizers.l2(self.config['l2_regularization']),
            input_shape=self.input_shape
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(self.config['dropout_rate']))
        
        # Add hidden layers
        if len(layer_sizes) > 1:
            model = self.create_layers(model, layer_sizes[1:])
        
        # Add output layer
        model.add(layers.Dense(self.outLayer, activation=self.activation))
        
        # Configure optimizer with learning rate
        opt = optimizers.Adam(learning_rate=self.config['learning_rate'])
        
        # Compile model
        model.compile(
            optimizer=opt,
            loss=self.loss,
            metrics=self.metrics
        )
        
        # Display model summary
        model.summary()
        
        self.model = model
        return model

    def train_model(self) -> dict:
        """Trains the model with appropriate callbacks and saves results."""
        if self.model is None:
            self.compile_model()
        
        # Model checkpoint callback
        model_path = os.path.join(self.model_dir, f"{self.config['model_name']}_best.keras")
        
        # Create callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                min_delta=self.config['min_delta'],
                restore_best_weights=True,
                verbose=1
            ),
            # Save best model
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                mode='min'
            ),
            # Learning rate reduction on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['patience'] // 3,
                min_lr=0.00001,
                verbose=1
            ),
            # TensorBoard logging
            TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                write_graph=True
            )
        ]
        
        print(f"Training model with {self.config['complexity']} complexity...")
        
        # Train model
        history = self.model.fit(
            self.data, self.target,
            validation_split=self.config['validation_split'],
            batch_size=self.config['batch_size'],
            epochs=self.config['max_epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        print('Evaluating deep model...')
        evaluation = self.model.evaluate(self.testdata, self.testTarget, verbose=1)
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, f"{self.config['model_name']}_final.keras")
        self.model.save(final_model_path)
        print(f"Model saved to {final_model_path}")
        
        # Return training results
        return {
            'history': history.history,
            'evaluation': dict(zip(self.model.metrics_names, evaluation)),
            'model_path': final_model_path,
            'complexity': self.determine_complexity(),
            'architecture': [layer.units for layer in self.model.layers if hasattr(layer, 'units')]
        }

    def roll_over(self) -> dict:
        """
        Builds, trains, and returns the trained model with performance metrics.
        
        Returns:
            Dictionary containing model, results, and performance metrics
        """
        self.compile_model()
        results = self.train_model()
        return {
            'model': self.model,
            'results': results
        }