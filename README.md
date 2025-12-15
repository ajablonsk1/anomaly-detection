# Network Traffic Classification

Universal machine learning framework for network traffic classification. Supports multiple datasets, configurable ML models, and automated comparison of different model configurations.

## Features

- **Multiple datasets support**: DDoS detection, Tor traffic detection, application classification
- **Three ML algorithms**: Random Forest, XGBoost, SVM
- **Flexible configuration**: Separate configs for datasets and model parameters
- **Batch processing**: Run multiple model configurations automatically
- **Automated visualizations**: Confusion matrices, feature importance, model comparison charts
- **Classification types**: Binary, multiclass, or both depending on dataset

## Project Structure

```
.
├── main.py                 # Main entry point
├── classifiers.py          # ML model wrappers
├── data_utils.py           # Data loading and preprocessing
├── plots.py                # Visualization functions
├── requirements.txt        # Python dependencies
├── datasets/
│   └── datasets.json       # Dataset configurations
├── models/                 # Model configurations
│   ├── model_base.json       # Default parameters
│   ├── model_fast_test.json      # Quick testing
│   ├── model_high_estimators.json
│   ├── model_regularized.json
│   └── model_linear_svm.json
├── data/                   # CSV data files (not included)
└── outputs/                # Generated results and plots
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# List available datasets
python main.py --list-datasets

# List available model configurations
python main.py --list-models

# Run analysis with all model configs
python main.py -d ddos

# Run with smaller dataset for testing
python main.py -d darknet_tor -s
```

## Usage

### Basic Commands

```bash
# Single model configuration
python main.py -d ddos -c base

# Multiple configurations
python main.py -d ddos -c base fast_test regularized

# Specific ML models only
python main.py -d ddos -m rf xgb

# Custom output directory
python main.py -d ddos -o results/experiment1

# Quiet mode (less output)
python main.py -d ddos -q
```

### Command Line Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--dataset` | `-d` | Dataset to analyze |
| `--smaller` | `-s` | Use smaller dataset for quick testing |
| `--config` | `-c` | Specific model config(s) to use |
| `--models` | `-m` | ML models to use (rf, xgb, svm) |
| `--models-dir` | | Directory with model configs (default: models/) |
| `--datasets-config` | | Path to datasets config |
| `--output-dir` | `-o` | Output directory |
| `--list-datasets` | `-ld` | List available datasets |
| `--list-models` | `-lm` | List model configurations |
| `--quiet` | `-q` | Reduce output verbosity |

## Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| `ddos` | binary + multiclass | DDoS attack detection (BENIGN vs ATTACK + attack types) |
| `darknet_tor` | multiclass only | Darknet traffic detection (Tor, Non-Tor, VPN, Non-VPN) |
| `darknet_app` | multiclass only | Application classification (AUDIO, VIDEO, CHAT, etc.) |

### Adding New Datasets

Edit `datasets/datasets.json`:

```json
{
  "my_dataset": {
    "name": "My Custom Dataset",
    "files": ["data/mydata.csv"],
    "files_smaller": ["data/mydata_smaller.csv"],
    "label_column": "Label",
    "binary_positive_class": "NORMAL",
    "binary_labels": ["NORMAL", "ANOMALY"],
    "drop_columns": ["ID", "Timestamp"],
    "column_mapping": {},
    "binary_only": false,
    "multiclass_only": false
  }
}
```

## Model Configurations

Each config in `models/` directory defines:

```json
{
  "name": "Configuration Name",
  "description": "What this config does",
  "preprocessing": {
    "variance_threshold": 0.0,
    "correlation_threshold": 0.95,
    "test_size": 0.3,
    "random_state": 42
  },
  "models": {
    "rf": { "n_estimators": 100, "max_depth": 20, ... },
    "xgb": { "n_estimators": 100, "max_depth": 10, ... },
    "svm": { "C": 1.0, "kernel": "rbf", ... }
  }
}
```

### Available Configurations

| Config            | Description |
|-------------------|-------------|
| `model_base.json` | Default balanced parameters |
| `model_fast_test.json`  | Minimal params for quick testing |
| `model_high_estimators.json` | More trees for potentially better accuracy |
| `model_regularized.json` | Stronger regularization to prevent overfitting |
| `model_linear_svm.json` | Linear SVM kernel for high-dimensional data |

## Output Files

After running analysis, the following files are generated in `outputs/<dataset>/`:

| File | Description |
|------|-------------|
| `all_results.csv` | Combined results from all configurations |
| `results_<config>.csv` | Results for specific configuration |
| `confusion_matrices_<config>.png` | Confusion matrix visualizations |
| `feature_importance_<config>.png` | Top 15 important features |
| `class_distribution.png` | Dataset class distribution |
| `model_comparison.png` | Comparison chart (when multiple configs) |

## Data Format

Expected CSV format with network flow features:

- Standard flow features (duration, packet counts, byte counts, etc.)
- Label column for classification target
- Supports CIC-IDS / CIC-DDoS dataset format

## Example Workflow

```bash
# 1. Quick test with fast config
python main.py -d ddos -s -c fast_test

# 2. Full analysis with all configs
python main.py -d ddos

# 3. Compare specific configurations
python main.py -d ddos -c base regularized high_estimators

# 4. Run only tree-based models (faster than SVM)
python main.py -d ddos -m rf xgb
```

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
