# FireAutoML

**FireAutoML** is an open-source, low-code Python library that automates the end-to-end process of building, evaluating, and deploying machine learning models. It is designed to assist data scientists by handling repetitive tasks such as data cleaning, feature engineering, model training, and hyperparameter tuning, allowing them to focus on delivering insights and business value.

The tool can be used as a Python library, a command-line interface (CLI), or a backend API for custom web applications.

## ✨ Features

- **Automated Data Preprocessing**: Handles missing values, scales numerical features, and encodes categorical variables.
- **Intelligent Feature Engineering**: Automatically selects the most relevant features for your model.
- **Advanced Data Augmentation**: Supports augmentation for tabular, text, and image data to improve model robustness.
- **Imbalanced Data Handling**: Integrates techniques like SMOTE and ADASYN to compensate for class imbalance.
- **Multi-Model Training**: Trains and evaluates a suite of models (e.g., RandomForest, XGBoost, SVM, Deep Learning) in parallel.
- **Comprehensive Evaluation**: Generates detailed reports with performance metrics, visualizations (confusion matrix, ROC curves), and actionable recommendations.
- **Flexible Usage**: Can be used as a library, CLI tool, or a REST API backend.
- **Extensible**: Designed to be easily extended with custom models, preprocessing steps, and evaluation metrics.

## 🚀 Installation

### Core Installation

For the core functionality (tabular data processing and standard ML models), you can install the library via pip:

```bash
pip install .
```

Or directly from the repository (if you have one):
```bash
pip install git+https://github.com/firebreather-heart/autoAI
```

### Optional Dependencies

FireAutoML uses `extras` to manage optional dependencies for specialized tasks. This keeps the core installation lightweight.

- **Deep Learning** (TensorFlow):
  ```bash
  pip install .[deep_learning]
  ```

- **Web API** (Flask):
  ```bash
  pip install .[web]
  ```

- **Text & Image Processing**:
  ```bash
  pip install .[text,image]
  ```

- **Full Installation** (all optional features):
  ```bash
  pip install .[full]
  ```

## 💻 Command-Line Interface (CLI) Usage

The `fireml` CLI is the quickest way to analyze a dataset.

### `analyze`

Run the full AutoML pipeline on a dataset.

```bash
# Auto-detect target and task
fireml analyze --input data.csv --output ./my_analysis

# Specify target and task
fireml analyze --input data.csv --target "price" --task regression
```

To run the tool without installation run
```bash
python -m fireml.cli <add-the-desired-arguments>
```

### `report`

Generate a data validation and exploration report.

```bash
fireml report --input data.csv --output report.html
```

### `web`

Launch the REST API server.

```bash
# Install web dependencies first
pip install .[web]

# Run the server
fireml web --host 0.0.0.0 --port 8000
```

## 🐍 Library Usage

You can integrate FireAutoML into your Python scripts for more control.

### High-Level Pipeline

For a quick, end-to-end analysis, use the `run_full_pipeline` function.

```python
from fireml.data_loader import load_data
from fireml.main_pipeline import run_full_pipeline

# Load data
df, metadata = load_data('path/to/your/data.csv')

# Run the pipeline
results = run_full_pipeline(
    df,
    target_column='your_target_column',
    task_type='classification' # or 'regression' or 'auto'
)

# The best model and evaluation report are saved in the 'output' directory.
print("Analysis complete. Best model:", results['summary']['best_model']['name'])
```

### Modular Usage

For more granular control, you can use individual components of the library. This is useful for integrating FireAutoML into existing data science workflows.

#### 1. Data Loading and Validation

```python
from fireml.data_loader import load_data
from fireml.utils import validate_dataframe
import json

df, metadata = load_data('data.csv')
validation_report = validate_dataframe(df)

print("Data Validation Issues:")
print(json.dumps(validation_report['issues'], indent=2))
```

#### 2. Preprocessing and Feature Engineering

```python
from fireml.preprocessing import manual_missing_NonObject_fix, encoding, dataNorm, feature_selector

# Handle missing values
# Note: This example assumes 'target' is the name of your target column.
df_clean = manual_missing_NonObject_fix(df, target_column='target')

# Separate features and target
features = df_clean.drop(columns=['target'])
target = df_clean['target']

# Normalize numerical features
# dataNorm returns a list of dataframes with different scaling methods
normalized_features_list = dataNorm(features)
scaled_features = normalized_features_list[0] # Using the first scaler (e.g., StandardScaler)

# Encode categorical features
encoded_features = encoding(scaled_features)

# Select the best features
selected_features, _ = feature_selector(encoded_features, target)
print(f"Selected {len(selected_features.columns)} features.")
```

#### 3. Data Augmentation (for Classification)

```python
from fireml.preprocessing import balance_classes
import pandas as pd

# Assuming a classification task with imbalanced classes
# Combine features and target for balancing
data_to_balance = pd.concat([selected_features, target], axis=1)

balanced_df = balance_classes(data_to_balance, target_column='target', method='smote')

# Separate again
X_final = balanced_df.drop(columns=['target'])
y_final = balanced_df['target']

print("Original class distribution:\n", target.value_counts())
print("Balanced class distribution:\n", y_final.value_counts())
```

#### 4. Model Training and Evaluation

```python
from sklearn.model_selection import train_test_split
from fireml.models import train_models
from fireml.evaluation import ModelEvaluator

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# Train a suite of models
trained_models, predictions, metrics = train_models(
    X_train, y_train, X_test, y_test, task_type='classification'
)

# Evaluate the models
evaluator = ModelEvaluator(task_type='classification')
for model_name, model_instance in trained_models:
    evaluator.evaluate_model(model_name, model_instance, X_test, y_test, X_train, y_train)

# Generate a final report
report = evaluator.generate_report(output_format='json')
print("Best Model:", report['summary']['best_model']['name'])
```

## 🌐 API for Web Integration

FireAutoML includes a Flask-based REST API, perfect for integrating with a Next.js or other modern frontend.

### Endpoints

- `POST /api/analyze`

  Triggers a full analysis pipeline.

  **Request Body**: `multipart/form-data`
  - `file`: The dataset file (e.g., CSV).
  - `target` (optional): The name of the target column.
  - `task` (optional): `classification` or `regression`.

  **Response**: `JSON`
  ```json
  {
    "message": "Analysis started. Check status using the task ID.",
    "task_id": "some-unique-task-id"
  }
  ```

- `GET /api/status/<task_id>`

  Checks the status of an analysis task.

  **Response**: `JSON`
  ```json
  {
    "status": "completed",
    "result_path": "/api/results/some-unique-task-id/evaluation_report.html"
  }
  ```

- `GET /api/results/<task_id>/<filename>`

  Retrieves a result file (e.g., the HTML report).

## 🛠️ Project Structure

The structure of the `fireml` package is organized by functionality.

```
fireml/
├── __init__.py             # Initializes the package
├── cli.py                  # Command-line interface logic
├── data_loader.py          # Data loading from various sources
├── evaluation.py           # Model evaluation and reporting
├── main_pipeline.py        # High-level function for the full pipeline
├── settings.py             # Application settings and configuration
├── models/                 # ML models
│   ├── __init__.py
│   ├── deep_learning.py
│   └── ensemble.py
├── preprocessing/          # Data preprocessing steps
│   ├── __init__.py
│   ├── autoImputer.py
│   ├── data_augmentation.py
│   └── feature_engineering.py
├── utils/                  # Helper utilities
│   ├── __init__.py
│   ├── helpers.py
│   ├── logging_config.py
│   └── validation.py
└── web/                    # Flask API for web integration
    ├── __init__.py
    └── app.py
```

## 📄 License

FireAutoML is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## 📞 Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers.

---

**FireAutoML** - Empowering Data Science with Automation
