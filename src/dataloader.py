import logging
from typing import Tuple
from torch.utils.data import DataLoader
from src.config import PipelineConfig
from src.metadata import ESC50Metadata
from src.preprocessing import AudioPreprocessor
from src.dataset import ESC50Dataset

logger = logging.getLogger("ESC_Pipeline")

def get_train_val_test_datasets(
    config: PipelineConfig,
    metadata: ESC50Metadata,
    preprocessor: AudioPreprocessor,
    use_hf: bool = False
) -> Tuple[ESC50Dataset, ESC50Dataset, ESC50Dataset]:
    """
    Splits the metadata into Train, Validation, and Test datasets based on configured folds.
    
    Args:
        config: Loaded PipelineConfig.
        metadata: Initialized and validated ESC50Metadata object.
        preprocessor: Configured AudioPreprocessor object.
        use_hf: If True, uses HuggingFace ASTFeatureExtractor.
        
    Returns:
        A tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_folds = config.dataset.train_folds
    val_folds = config.dataset.val_folds
    test_folds = config.dataset.test_folds
    
    # Check for fold overlaps to prevent data leakage
    overlapping_folds = set(train_folds) & set(val_folds) | set(train_folds) & set(test_folds) | set(val_folds) & set(test_folds)
    if overlapping_folds:
        raise ValueError(f"Folds overlap across splits! Data leakage will occur. Overlapping folds: {overlapping_folds}")
        
    logger.info(
        "Creating datasets - Train folds: %s, Val folds: %s, Test folds: %s",
        train_folds, val_folds, test_folds
    )
    
    # Query filepaths and labels from metadata manager
    train_files = metadata.get_audio_paths_and_labels(train_folds)
    val_files = metadata.get_audio_paths_and_labels(val_folds)
    test_files = metadata.get_audio_paths_and_labels(test_folds)
    
    logger.info(
        "Sample sizes - Train: %d, Val: %d, Test: %d",
        len(train_files), len(val_files), len(test_files)
    )
    
    # Initialize the custom PyTorch Dataset classes
    train_dataset = ESC50Dataset(train_files, preprocessor, use_hf=use_hf)
    val_dataset = ESC50Dataset(val_files, preprocessor, use_hf=use_hf)
    test_dataset = ESC50Dataset(test_files, preprocessor, use_hf=use_hf)
    
    return train_dataset, val_dataset, test_dataset

def build_dataloaders(
    config: PipelineConfig,
    train_dataset: ESC50Dataset,
    val_dataset: ESC50Dataset,
    test_dataset: ESC50Dataset
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates optimized PyTorch DataLoaders for each dataset split.
    
    Args:
        config: Loaded PipelineConfig.
        train_dataset: Custom Train ESC50Dataset.
        val_dataset: Custom Val ESC50Dataset.
        test_dataset: Custom Test ESC50Dataset.
        
    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    dl_config = config.dataloader
    
    logger.info(
        "Building DataLoaders - Batch Size: %d, Num Workers: %d, Pin Memory: %s",
        dl_config.batch_size, dl_config.num_workers, dl_config.pin_memory
    )
    
    # Train Loader: Shuffle, drop partial last batch to stabilize training
    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_config.batch_size,
        shuffle=True,
        num_workers=dl_config.num_workers,
        pin_memory=dl_config.pin_memory,
        drop_last=True
    )
    
    # Validation Loader: No shuffle, no drop_last
    val_loader = DataLoader(
        val_dataset,
        batch_size=dl_config.batch_size,
        shuffle=False,
        num_workers=dl_config.num_workers,
        pin_memory=dl_config.pin_memory,
        drop_last=False
    )
    
    # Test Loader: No shuffle, no drop_last
    test_loader = DataLoader(
        test_dataset,
        batch_size=dl_config.batch_size,
        shuffle=False,
        num_workers=dl_config.num_workers,
        pin_memory=dl_config.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
