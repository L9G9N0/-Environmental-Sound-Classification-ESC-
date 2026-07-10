# Environmental Sound Classification (ESC) using Audio Spectrogram Transformer (AST)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A5%97-Transformers-yellow.svg)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains a production-grade implementation of the **Environmental Sound Classification (ESC)** project using the **ESC-50 Dataset** and the **Audio Spectrogram Transformer (AST)**. 

The codebase is split into modular components:
1. **Member 1 (Data Engineering)**: Programmatic downloader, metadata validation, digital signal processing (resampling, mono mixing, padding/truncating), Log-Mel Spectrogram extraction, and lazy-loading PyTorch datasets.
2. **Member 2 (Modeling & MLOps)**: Pre-trained AST model loading, classifier head replacement (50 classes), linear probing (backbone freezing), early stopping, model checkpointing (resume/restart capability), CSV/TensorBoard logging, and hyperparameter configuration.

---

## 🛠️ Folder Structure

```text
ESC_Project/
├── dataset/             # Ingested ESC-50 dataset (auto-downloaded/unzipped)
├── configs/             # Configuration files
│   └── config.yaml      # Preprocessing, model architecture & training hyperparameters
├── src/                 # Source code package
│   ├── __init__.py      # Package marker
│   ├── config.py        # YAML configuration parser mapping to dataclasses
│   ├── downloader.py    # Downloads and extracts the ESC-50 dataset
│   ├── metadata.py      # Metadata loading, cleaning, validation, and EDA
│   ├── preprocessing.py # Core DSP (resampling, mono mixing, log-Mel transform)
│   ├── dataset.py       # Custom PyTorch Dataset class (lazy-loading)
│   ├── dataloader.py    # Cross-validation splits and PyTorch DataLoaders
│   ├── model.py         # Loads and configures the AST model and linear probing
│   ├── trainer.py       # ASTTrainer class (training loops, early stopping, checkpoints)
│   ├── train.py         # Orchestrates download, preprocessing, modeling, and training
│   └── utils.py         # Logging setup and execution timers
├── tests/               # Automated test suites
│   ├── __init__.py
│   ├── test_pipeline.py # Preprocessing & dataloader tests (uses synthetic data)
│   └── test_model.py    # Model loading, optimizer steps, checkpoint, and overfitting tests
├── outputs/             # Visually exported plots, logs, checkpoints, and TensorBoard files
│   ├── checkpoints/     # Saved model weights (best_model.pt, latest_model.pt)
│   ├── logs/            # CSV training logs
│   └── tensorboard/     # TensorBoard runs
├── logs/                # System data_pipeline.log file
├── requirements.txt     # Dependency list
├── LICENSE              # MIT License
└── README.md            # Project documentation (this file)
```

---

## 🚀 Installation & Environment Setup

Ensure you have Python 3.10+ installed. Run the following commands to set up the development environment:

```bash
# 1. Clone the repository and navigate to the project directory
cd ESC_Project

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# 4. Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Preprocessing & Audio Specifications

The data pipeline preprocesses audio to match the inductive biases of the pre-trained AST model (`MIT/ast-finetuned-audioset-10-10-0.4593`):

1. **Resampling**: Resamples signals to exactly **16 kHz**.
2. **Mono Mixing**: Averages stereo signals to single-channel (mono).
3. **Duration Standardization**: Pads or truncates waveforms to exactly **5.0 seconds** (80,000 samples).
4. **Log-Mel Transform**: Mel-filterbank warping with $128$ frequency bins, $400$ window size (25ms), and $160$ stride (10ms).
5. **Normalization**: Standardizes features using AudioSet population statistics (Mean: `-4.2677393`, Std: `4.5689974`).

---

## 🧑‍💻 How to Train the Model

You can launch and manage training runs directly from the CLI.

### 1. Start a New Training Run (Default settings)
To run linear probing (encoder frozen, classification head trainable) for 10 epochs:
```bash
python src/train.py
```
This script will download and verify the dataset if missing, preprocess the splits, load the model, and run the training process.

### 2. Full Fine-Tuning (Backbone Unfrozen)
To train all parameters end-to-end with a smaller learning rate:
```bash
python src/train.py --freeze false --epochs 15 --lr 5e-5
```

### 3. Resuming Training
To resume training after a hardware failure or manual stop, point the script to the latest checkpoint file:
```bash
python src/train.py --resume outputs/checkpoints/latest_model.pt
```

### 4. Real-time TensorBoard Monitoring
To view live learning curves, launch TensorBoard and open `http://localhost:6006` in your browser:
```bash
tensorboard --logdir outputs/tensorboard
```

---

## 🧪 Testing the Codebase

The codebase includes two test suites:
- **`tests/test_pipeline.py`**: Verifies data loading, resampling, Mel-extraction, and dataset batching.
- **`tests/test_model.py`**: Verifies model loading, classification head replacements, optimizer steps, checkpoint serialization, and overfitting convergence.

To run all tests:
```bash
python -m unittest discover tests
```

---

## 📖 Deep-Dive Documentation
For a complete theoretical deep-dive on how the AST works, self-attention mechanisms, loss calculations, backpropagation, and handover details for Member 3, read the [Member 2 Documentation Guide](file:///Users/legend27648/agy_project/AI%20Audio/ESC_Project/docs/MEMBER_2.md).
