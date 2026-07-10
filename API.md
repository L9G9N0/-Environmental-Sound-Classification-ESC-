# 🔌 API Reference Guide

This document provides developer references for CLI commands, configuration parameters, and Python classes in the ESC-50 AST pipeline.

---

## 1. CLI API Reference

The script [src/train.py](file:///Users/legend27648/agy_project/AI%20Audio/ESC_Project/src/train.py) is the command-line interface for starting and resuming training runs:

```bash
python src/train.py [arguments]
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | `str` | `configs/config.yaml` | Path to the YAML configuration file. |
| `--resume` | `str` | `None` | Path to a model checkpoint (`latest_model.pt`) to resume training from. |
| `--epochs` | `int` | `None` | Overrides the number of training epochs defined in the YAML file. |
| `--lr` | `float` | `None` | Overrides the learning rate defined in the YAML file. |
| `--freeze` | `str` | `None` | Overrides encoder freeze settings: `true` (Linear Probing), `false` (Fine-Tuning). |

### Examples
```bash
# Start default training
python src/train.py

# Run full fine-tuning with a custom learning rate for 20 epochs
python src/train.py --freeze false --lr 5e-5 --epochs 20

# Resume interrupted training from checkpoint
python src/train.py --resume outputs/checkpoints/latest_model.pt
```

---

## 2. Configuration Class Reference

Defined in [src/config.py](file:///Users/legend27648/agy_project/AI%20Audio/ESC_Project/src/config.py):

### `PipelineConfig`
The main configuration class. Load it using:
```python
from src.config import PipelineConfig
config = PipelineConfig.from_yaml("configs/config.yaml")
```

- **`data`**: `DataConfig`
  - `download_url` (`str`): Target dataset ZIP URL.
  - `zip_name` (`str`): Downloaded file name.
  - `raw_data_dir` (`str`): Extraction directory.
  - `metadata_csv` (`str`): Path to `esc50.csv`.
  - `audio_dir` (`str`): Path to the WAV files folder.
- **`preprocessing`**: `PreprocessingConfig`
  - `target_sr` (`int`): Target sampling rate (e.g., 16000).
  - `duration_sec` (`float`): Clip duration (e.g., 5.0).
  - `mono` (`bool`): Toggle stereo-to-mono average.
  - `n_mels` (`int`): Number of Mel bins (e.g., 128).
  - `n_fft` (`int`): FFT window size (e.g., 400).
  - `hop_length` (`int`): Frame step size (e.g., 160).
  - `power` (`float`): Power spectrogram exponent (e.g., 2.0).
  - `normalize` (`bool`): Toggle subtraction of mean and division by standard deviation.
  - `mean` (`float`): Normalization mean.
  - `std` (`float`): Normalization standard deviation.
- **`dataset`**: `DatasetConfig`
  - `train_folds` (`List[int]`): List of fold numbers for training (e.g., `[1, 2, 3]`).
  - `val_folds` (`List[int]`): List of fold numbers for validation (e.g., `[4]`).
  - `test_folds` (`List[int]`): List of fold numbers for testing (e.g., `[5]`).
- **`dataloader`**: `DataloaderConfig`
  - `batch_size` (`int`): Training batch size.
  - `num_workers` (`int`): Multi-processing workers.
  - `pin_memory` (`bool`): Speed up CPU-to-GPU data transfers.
- **`model`**: `ModelConfig`
  - `checkpoint_name` (`str`): Pre-trained Hugging Face model checkpoint (e.g., `"MIT/ast-finetuned-audioset-10-10-0.4593"`).
  - `num_classes` (`int`): Target classification categories (e.g., 50).
  - `freeze_encoder` (`bool`): Freeze transformer backbone weights.
  - `use_hf` (`bool`): Generate `1024x128` input shape.
- **`training`**: `TrainingConfig`
  - `epochs` (`int`): Training epochs.
  - `learning_rate` (`float`): Initial learning rate.
  - `weight_decay` (`float`): Regularization penalty.
  - `scheduler_type` (`str`): Scheduler choice: `"cosine"`, `"step"`, `"plateau"`.
  - `scheduler_patience` (`int`): Epoch patience for Plateau decay.
  - `step_size` (`int`): Interval for step decay.
  - `gamma` (`float`): Learning rate decay scale factor.
  - `early_stopping_patience` (`int`): Early stopping patience limit.
  - `checkpoint_dir` (`str`): Output folder for checkpoints.
  - `log_dir` (`str`): Output folder for CSV logs.
  - `tb_dir` (`str`): Output folder for TensorBoard event files.

---

## 3. Core Class Reference

### `AudioPreprocessor`
Handles raw audio loading, standardized DSP transforms, and spectrogram generation.

* **File**: `src/preprocessing.py`
* **Initialization**:
  ```python
  from src.preprocessing import AudioPreprocessor
  preprocessor = AudioPreprocessor(config)
  ```
* **Methods**:
  - `load_audio(file_path: str) -> Tuple[torch.Tensor, int]`: Loads WAV files using torchaudio or soundfile fallback.
  - `standardize_waveform(waveform: torch.Tensor, src_sr: int) -> torch.Tensor`: Downsamples, averages channels, and crops/pads duration.
  - `compute_log_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor`: Extracts standard log-Mel features using torchaudio.
  - `extract_features_hf(waveform: torch.Tensor) -> torch.Tensor`: Generates `1024x128` features using Hugging Face's `ASTFeatureExtractor`.
  - `process_file(file_path: str, use_hf: bool = False) -> torch.Tensor`: Loads, standardizes, and extracts features in a single call.

### `ESC50Dataset`
Lazy-loading custom dataset subclass.

* **File**: `src/dataset.py`
* **Initialization**:
  ```python
  from src.dataset import ESC50Dataset
  dataset = ESC50Dataset(data_list, preprocessor, use_hf=True)
  ```
* **Methods**:
  - `__len__() -> int`: Returns the total number of audio samples.
  - `__getitem__(idx: int) -> Tuple[torch.Tensor, int]`: Returns a tuple containing the normalized feature tensor and its integer label.

### `build_ast_model`
Model loader utility.

* **File**: `src/model.py`
* **Call**:
  ```python
  from src.model import build_ast_model
  model = build_ast_model(config)
  ```
* **Functionality**:
  - Loads the pre-trained Hugging Face model (`ASTForAudioClassification`).
  - Swaps the classification head to support 50 target classes.
  - Freezes the encoder backbone if `freeze_encoder` is enabled.
  - Audits and logs parameter counts.

### `ASTTrainer`
Manages the training execution loops, validation, early stopping, CSV/TB logging, and checkpoints.

* **File**: `src/trainer.py`
* **Initialization**:
  ```python
  from src.trainer import ASTTrainer
  trainer = ASTTrainer(config, model, train_loader, val_loader, device)
  ```
* **Methods**:
  - `train_epoch(epoch: int) -> Tuple[float, float]`: Runs one training epoch and updates metrics.
  - `validate() -> Tuple[float, float]`: Evaluates the validation dataset split.
  - `fit(resume_path: Optional[str] = None) -> Dict[str, List[float]]`: Runs the complete training loops.
  - `save_checkpoint(epoch: int, val_loss: float, val_acc: float, filepath: str)`: Saves the execution state checkpoint.
  - `load_checkpoint(filepath: str)`: Reloads checkpoint data to resume training.
