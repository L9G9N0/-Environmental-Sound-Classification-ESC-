# 📜 Changelog

All notable changes to this project will be documented in this file. This project follows Semantic Versioning (SemVer).

---

## [1.1.0] - 2026-07-10
### Added (Member 2 Modeling Integration)
* **AST Model Adaptation**: Integrated Hugging Face's `ASTForAudioClassification` with custom head resizing (50 classes) using `ignore_mismatched_sizes=True` flags.
* **Linear Probing Mechanism**: Built parameters freezing switches for the transformer encoder backbone (`freeze_encoder`).
* **Trainer Class**: Developed the `ASTTrainer` module implementing forward/backward epoch loops, validation evaluations, early stopping, and learning rate schedulers.
* **Checkpoint & Resume Features**: Implemented state dictionary serialization for model weights, optimizer variables, and scheduler states. Resolves the validation loss sync order bug.
* **CLI script**: Developed the `train.py` wrapper to start or resume runs, customize epochs, and override learning rates via terminal arguments.
* **Unit Test Suite**: Created `tests/test_model.py` which dry-runs training, checks freezing, verifies checkpoint loads, and validates overfitting convergence.
* **TensorBoard Integration**: Added SummaryWriter logging to output training and validation metrics.

### Changed
* **Configuration Mapping**: Updated `configs/config.yaml` and `src/config.py` to support ModelConfig and TrainingConfig.
* **Requirements Lock**: Updated `requirements.txt` to include `tensorboard>=2.12.0`.
* **Documentation**: Overwrote `README.md` with complete architecture and setup guides, and added `docs/MEMBER_2.md`.

---

## [1.0.0] - 2026-07-10
### Added (Member 1 Ingestion Integration)
* **Programmatic Downloader**: Developed `downloader.py` to fetch, extract, and verify the ESC-50 ZIP archive from GitHub.
* **Metadata Auditor**: Created `metadata.py` to load and audit `esc50.csv` columns, clean null values, and create class mappings.
* **DSP Preprocessor**: Created `preprocessing.py` handling resampling (16kHz), mono-mixing, duration standardizing (5.0s), and extracting log-Mel spectrograms.
* **Dataset & DataLoader**: Created `dataset.py` (lazy loading) and `dataloader.py` (splitting folds and creating DataLoader batches).
* **Test pipeline**: Created `tests/test_pipeline.py` verifying DSP operations.
* **Student Manual**: Created `docs/EXPLANATION.md` explaining DSP terms and dataset details.
