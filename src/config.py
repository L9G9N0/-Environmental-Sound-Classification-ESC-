import os
from dataclasses import dataclass
from typing import List
import yaml

@dataclass
class DataConfig:
    download_url: str
    zip_name: str
    raw_data_dir: str
    metadata_csv: str
    audio_dir: str

@dataclass
class PreprocessingConfig:
    target_sr: int
    duration_sec: float
    mono: bool
    n_mels: int
    n_fft: int
    hop_length: int
    power: float
    normalize: bool
    mean: float
    std: float

@dataclass
class DatasetConfig:
    train_folds: List[int]
    val_folds: List[int]
    test_folds: List[int]

@dataclass
class DataloaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool

@dataclass
class ModelConfig:
    checkpoint_name: str
    num_classes: int
    freeze_encoder: bool
    use_hf: bool

@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    weight_decay: float
    scheduler_type: str
    scheduler_patience: int
    step_size: int
    gamma: float
    early_stopping_patience: int
    checkpoint_dir: str
    log_dir: str
    tb_dir: str

@dataclass
class PipelineConfig:
    data: DataConfig
    preprocessing: PreprocessingConfig
    dataset: DatasetConfig
    dataloader: DataloaderConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Loads and parses a YAML configuration file into structured dataclasses."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")
            
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        try:
            data = DataConfig(**config_dict["data"])
            preprocessing = PreprocessingConfig(**config_dict["preprocessing"])
            dataset = DatasetConfig(**config_dict["dataset"])
            dataloader = DataloaderConfig(**config_dict["dataloader"])
            model = ModelConfig(**config_dict["model"])
            training = TrainingConfig(**config_dict["training"])
            
            return cls(
                data=data,
                preprocessing=preprocessing,
                dataset=dataset,
                dataloader=dataloader,
                model=model,
                training=training
            )
        except KeyError as e:
            raise KeyError(f"Missing required configuration key in YAML: {e}")
        except TypeError as e:
            raise TypeError(f"Invalid type or parameter mismatch in configuration loading: {e}")
