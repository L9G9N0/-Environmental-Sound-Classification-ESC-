# 🛠️ local Development & Setup Guide

This document provides guidelines for setting up your local environment, coding standards, and running test suites.

---

## 1. Development Environment Setup

We recommend developing on a UNIX-like operating system (macOS or Linux).

### Local Setup
1. **Clone the repository and enter the directory**:
   ```bash
   cd ESC_Project
   ```
2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   ```
3. **Activate the environment**:
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # .venv\Scripts\activate   # On Windows
   ```
4. **Upgrade pip and install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 2. Coding Standards & Guidelines

To maintain production-grade code quality, follow these coding guidelines:

* **PEP 8 Compliance**: Use standard PEP 8 formatting (4-space indentation, descriptive variable names, snake_case for functions, PascalCase for classes).
* **Type Hints**: All functions and methods must include type hints:
  ```python
  def build_ast_model(config: PipelineConfig) -> ASTForAudioClassification:
  ```
* **Docstrings**: Document all classes, modules, and public functions using Sphinx or Google style docstrings:
  ```python
  """
  Initializes the dataset.
  
  Args:
      data_list: List of tuples (file_path, class_id).
      preprocessor: Configured instance of AudioPreprocessor.
  """
  ```
* **Clean Logging**: Use the logger instead of raw print statements:
  ```python
  import logging
  logger = logging.getLogger("ESC_Pipeline")
  logger.info("Initializing AST...")
  ```
* **Error Handling**: Catch specific exceptions and include descriptive error logs. Do not catch generic `Exception` blocks without logging a stack trace.

---

## 3. Running & Adding Tests

### Run the Test Suite
Run the test suites using Python's unittest module:
```bash
# Run all tests (Data Pipeline + Model Trainer)
python -m unittest discover -s tests

# Run only model training tests
python -m unittest tests/test_model.py

# Run only DSP preprocessing tests
python -m unittest tests/test_pipeline.py
```

### Adding New Test Cases
When adding a new test class or case, follow these conventions:
1. Save the file in the `tests/` directory with the `test_` prefix (e.g., `tests/test_inference.py`).
2. Inherit from `unittest.TestCase`.
3. Put shared setups (like configuration overrides or directory setups) in the `setUpClass` class method.
4. Clean up any file output artifacts inside the `tearDownClass` class method.
5. Create helper methods inside the class to isolate dependencies, like the `SyntheticASTDataset` helper used in `test_model.py`.

---

## 4. Debugging & Logging

* **Local Logs**: The data pipeline logs execution progress to the file `logs/data_pipeline.log` (created by Member 1) and trainer logs are written to the path configured in `config.yaml` (default: `outputs/logs/data_pipeline.log`).
* **TensorBoard Telemetry**: Run TensorBoard locally to debug training metrics, learning curves, and loss anomalies:
   ```bash
   tensorboard --logdir outputs/tensorboard
   ```
* **PyTorch Device Debugging**: To check which device is active during execution, search the terminal output for:
  - `Found CUDA device` (NVIDIA GPUs)
  - `Found macOS Apple Silicon GPU acceleration (MPS)` (Apple M-series chips)
  - `No GPU accelerator found. Standardizing to CPU execution` (CPU fallback)
