# Environmental Sound Classification (ESC) — Preprocessing & Data Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains **Member 1's** completed module for the Environmental Sound Classification (ESC) project using the **ESC-50** dataset and the **Audio Spectrogram Transformer (AST)**. 

Member 1 is responsible for the complete data engineering, validation, ingestion, and preprocessing pipeline. This pipeline delivers clean, validated, normalized Log-Mel Spectrogram features, completely formatted for direct model consumption by **Member 2** for AST model training.

---

## 🛠️ Architecture & Folder Structure

The project directory structure is designed to enforce clean separation of concerns:

```text
ESC_Project/
├── dataset/             # Ingested ESC-50 dataset (auto-downloaded/unzipped)
├── configs/             # Configuration files
│   └── config.yaml      # Unified hyperparameters (SR, FFT, Normalization)
├── src/                 # Preprocessing source code package
│   ├── __init__.py      # Package marker
│   ├── config.py        # Yaml configuration parser mapping to dataclasses
│   ├── downloader.py    # Robust programmatic downloader and unzipper
│   ├── metadata.py      # Metadata loading, cleaning, validation, and EDA
│   ├── preprocessing.py # Core DSP (resampling, mono mixing, log-Mel transform)
│   ├── dataset.py       # Custom PyTorch Dataset class
│   ├── dataloader.py    # Cross-validation splits and PyTorch Dataloaders
│   └── utils.py         # Logging setup and execution timers
├── tests/               # Automated test suites
│   ├── __init__.py
│   └── test_pipeline.py # Unit & integration tests (uses synthetic & real data)
├── outputs/             # Visually exported plots (waveforms, spectrograms)
├── logs/                # Run logs for system debugging
├── requirements.txt     # Dependency lockfile
├── .gitignore           # Git ignore file
├── LICENSE              # MIT License
└── README.md            # Project documentation (this file)
```

---

## 🚀 Installation & Environment Setup

Ensure you have Python 3.10+ installed. Follow these commands to set up the development environment:

```bash
# 1. Clone the repository and navigate to the project directory
cd ESC_Project

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 4. Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Preprocessing Specification for AST

The preprocessor strictly conforms to the inductive biases expected by the pre-trained AST model (`MIT/ast-finetuned-audioset-10-10-0.4593`):

1.  **Resampling**: Downsamples original 44.1 kHz signals to exactly **16 kHz**.
2.  **Mono Mixing**: Reduces multi-channel inputs to single-channel (mono) by averaging.
3.  **Duration Standardization**: Clips or zero-pads waveforms to exactly **5.0 seconds** ($16,000 \text{ Hz} \times 5 \text{ s} = 80,000 \text{ samples}$).
4.  **Spectrogram Transform**: Computes a Mel Spectrogram with:
    *   Window size (`n_fft`): 400 samples (25ms)
    *   Stride (`hop_length`): 160 samples (10ms)
    *   Mel bands (`n_mels`): 128
5.  **Log Amplitude Scaling**: Converts power spectrograms to decibels using $10 \times \log_{10}(\text{amplitude})$.
6.  **Normalization**: Standardizes log-Mel features using AudioSet statistics:
    *   **Mean**: `-4.2677393`
    *   **Standard Deviation**: `4.5689974`

---

## 🧑‍💻 How Member 2 Will Use This Work (Handover Guide)

Member 2 can easily import the pre-built dataloader builder and feed the output batches directly to their training loops:

```python
import sys
import os

# Append the project path
sys.path.append(os.path.abspath("src"))

from src.config import PipelineConfig
from src.metadata import ESC50Metadata
from src.preprocessing import AudioPreprocessor
from src.dataloader import get_train_val_test_datasets, build_dataloaders

# 1. Load Configurations
config = PipelineConfig.from_yaml("configs/config.yaml")

# 2. Initialize Preprocessor
preprocessor = AudioPreprocessor(config)

# 3. Load & Audit Metadata
metadata = ESC50Metadata(config)
metadata.load_and_validate()
metadata.get_summary_statistics() # Optional: prints metadata balance audit

# 4. Generate Datasets (Toggle use_hf=True if loading AST via Hugging Face Transformers)
train_ds, val_ds, test_ds = get_train_val_test_datasets(
    config, metadata, preprocessor, use_hf=True
)

# 5. Build PyTorch DataLoaders
train_loader, val_loader, test_loader = build_dataloaders(
    config, train_ds, val_ds, test_ds
)

# 6. Train loop consumption
for batch_idx, (spectrograms, labels) in enumerate(train_loader):
    # spectrograms shape: [batch_size, time_frames, n_mels] (i.e. [16, 1024, 128])
    # labels shape: [batch_size]
    print(f"Batch {batch_idx}: Tensors ready for training! Shapes: {spectrograms.shape}, Labels: {labels.shape}")
    break
```

---

## 🧪 Testing the Pipeline

We provide a complete test suite to verify configuration parameters, DSP resamplers, mono-mixing, and data collation:

```bash
# From the ESC_Project root directory, run:
python -m unittest tests/test_pipeline.py
```

Expected output:
*   Generates a local 5-second, 44.1 kHz stereo audio file.
*   Asserts resampling to 16 kHz mono.
*   Verifies 2D spectrogram tensor boundaries.
*   Performs a mock training batch run.
*   Attempts download, extraction, and validation of the official ESC-50 dataset.

---

## 🛡️ License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
