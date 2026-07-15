"""
predict.py

Runs inference on a single real-world .wav file (e.g. a phone recording,
not from the ESC-50 dataset) and prints the predicted sound class.

Usage:
    python src/predict.py --audio path/to/recording.wav
    python src/predict.py --audio path/to/recording.wav --checkpoint outputs/checkpoints/best_model.pt --topk 3
"""

import argparse
import logging
import os
import sys
import warnings

# Suppress noisy library output for a clean demo/CLI experience.
# Model was already downloaded during training, so offline mode avoids
# redundant Hugging Face Hub network calls and their log lines entirely.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# Adjust path if running directly from src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

from src.config import PipelineConfig
from src.model import build_ast_model
from src.metadata import ESC50Metadata
from src.preprocessing import AudioPreprocessor

logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ESC_Pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AST ESC-50 Inference on a single real-world audio file")
    parser.add_argument("--audio", type=str, required=True, help="Path to a .wav file to classify")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pt",
        help="Path to the trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Number of top predictions to display (default: 1)",
    )
    return parser.parse_args()


def get_class_names(config: PipelineConfig) -> dict:
    """Returns a dict mapping label index -> human-readable class name (e.g. 3 -> 'dog')."""
    try:
        metadata = ESC50Metadata(config)
        metadata.load_and_validate()
        return metadata.id_to_class
    except Exception as e:
        logger.warning("Could not load class names from metadata (%s). Falling back to numeric labels.", str(e))
        num_classes = config.model.num_classes
        return {i: str(i) for i in range(num_classes)}


def main():
    args = parse_args()

    logger.info("=" * 50)
    logger.info("AST ESC-50 Inference - Real World Audio Prediction")
    logger.info("=" * 50)

    if not os.path.exists(args.audio):
        logger.error("Audio file not found at: %s", args.audio)
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        logger.error("Checkpoint not found at: %s", args.checkpoint)
        sys.exit(1)

    config = PipelineConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build preprocessor - identical steps used during training (resample, mono, pad/truncate, feature extraction)
    preprocessor = AudioPreprocessor(config)
    use_hf = config.model.use_hf

    logger.info("Processing audio file: %s", args.audio)
    features = preprocessor.process_file(args.audio, use_hf=use_hf)

    # Add batch dimension: (time_frames, n_mels) -> (1, time_frames, n_mels)
    features = features.unsqueeze(0).to(device)

    # Build model architecture and load trained weights
    model = build_ast_model(config)
    model.to(device)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Run inference
    with torch.no_grad():
        outputs = model(input_values=features)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze(0)  # shape: (num_classes,)

    id_to_class = get_class_names(config)

    topk = min(args.topk, probs.shape[0])
    top_probs, top_indices = torch.topk(probs, k=topk)

    logger.info("-" * 50)
    logger.info("PREDICTION RESULTS for: %s", args.audio)
    logger.info("-" * 50)

    for rank in range(topk):
        idx = top_indices[rank].item()
        confidence = top_probs[rank].item()
        class_name = id_to_class.get(idx, str(idx))
        logger.info("#%d: %s (confidence: %.2f%%)", rank + 1, class_name, confidence * 100)

    logger.info("-" * 50)


if __name__ == "__main__":
    main()
