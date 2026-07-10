import argparse
import logging
import os
import sys
import torch

# Adjust path if running directly from src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PipelineConfig
from src.utils import setup_logging, time_execution
from src.downloader import ESC50Downloader
from src.metadata import ESC50Metadata
from src.preprocessing import AudioPreprocessor
from src.dataloader import get_train_val_test_datasets, build_dataloaders
from src.model import build_ast_model
from src.trainer import ASTTrainer

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for model training."""
    parser = argparse.ArgumentParser(description="AST Environmental Sound Classification Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a model checkpoint to resume training from"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the learning rate"
    )
    parser.add_argument(
        "--freeze",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Override encoder freeze settings (true = linear probing, false = full fine-tuning)"
    )
    return parser.parse_args()

def get_device() -> torch.device:
    """Selects the best available accelerator (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger = logging.getLogger("ESC_Pipeline")
        logger.info("Found CUDA device: %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger = logging.getLogger("ESC_Pipeline")
        logger.info("Found macOS Apple Silicon GPU acceleration (MPS).")
    else:
        device = torch.device("cpu")
        logger = logging.getLogger("ESC_Pipeline")
        logger.info("No GPU accelerator found. Standardizing to CPU execution.")
    return device

def main() -> None:
    """Main training orchestration function."""
    args = parse_args()

    # 1. Load configuration file
    try:
        config = PipelineConfig.from_yaml(args.config)
    except Exception as e:
        print(f"CRITICAL: Failed to load configuration from {args.config}. Error: {e}")
        sys.exit(1)

    # 2. Overrides from command line
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.freeze is not None:
        config.model.freeze_encoder = (args.freeze == "true")

    # 3. Setup Logging
    logger = setup_logging(log_dir=config.training.log_dir, log_level=logging.INFO)
    logger.info("==================================================")
    logger.info("Starting Audio Spectrogram Transformer Training Process")
    logger.info("==================================================")

    # 4. Ingest and Verify Dataset
    with time_execution("Dataset download and extraction"):
        try:
            downloader = ESC50Downloader(config)
            downloader.run_pipeline()
        except Exception as e:
            logger.error("Dataset ingestion failed. Aborting training. Error: %s", str(e))
            sys.exit(1)

    # 5. Load and Audit Metadata
    metadata = ESC50Metadata(config)
    try:
        metadata.load_and_validate()
        metadata.get_summary_statistics()
    except Exception as e:
        logger.error("Metadata auditing failed. Aborting training. Error: %s", str(e))
        sys.exit(1)

    # 6. Initialize Audio Preprocessor
    preprocessor = AudioPreprocessor(config)

    # 7. Create Datasets (Use use_hf=True to format inputs for AST)
    use_hf = config.model.use_hf
    logger.info("Formatting data with HuggingFace compatible features: %s", str(use_hf))
    train_ds, val_ds, test_ds = get_train_val_test_datasets(
        config,
        metadata,
        preprocessor,
        use_hf=use_hf
    )

    # 8. Build DataLoaders
    train_loader, val_loader, _ = build_dataloaders(config, train_ds, val_ds, test_ds)

    # 9. Determine Acceleration Device
    device = get_device()

    # 10. Construct Model
    model = build_ast_model(config)

    # 11. Instantiate Trainer and Fit
    trainer = ASTTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    with time_execution("Model Training Loop"):
        try:
            trainer.fit(resume_path=args.resume)
        except Exception as e:
            logger.error("Training loop crashed. Exception: %s", str(e))
            sys.exit(1)

    logger.info("AST Training Pipeline successfully completed.")

if __name__ == "__main__":
    main()
