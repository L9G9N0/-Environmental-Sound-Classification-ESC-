"""
evaluate.py

Loads the best trained checkpoint and evaluates it on the held-out TEST split
(Fold 5) which was never seen during training or validation.

Outputs:
    - Accuracy, Precision, Recall, F1-Score (weighted, across 50 classes)
    - A 50x50 confusion matrix plot saved to outputs/plots/confusion_matrix.png

Usage:
    python src/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pt
"""

import argparse
import logging
import os
import sys

# Adjust path if running directly from src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from src.config import PipelineConfig
from src.model import build_ast_model
from src.metadata import ESC50Metadata
from src.preprocessing import AudioPreprocessor
from src.dataloader import get_train_val_test_datasets, build_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ESC_Pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AST Environmental Sound Classification Evaluator")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pt",
        help="Path to the trained model checkpoint (.pt) to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/plots",
        help="Directory to save the confusion matrix plot",
    )
    return parser.parse_args()


def get_class_names(config: PipelineConfig) -> list:
    """
    Loads human-readable class names (e.g. 'dog', 'rain') mapped by label
    index, for use in the confusion matrix plot. Falls back to numeric
    labels (0-49) if metadata loading fails for any reason.
    """
    try:
        metadata = ESC50Metadata(config)
        metadata.load_and_validate()
        id_to_class = metadata.id_to_class  # dict: {target_int: category_str}
        num_classes = config.model.num_classes
        return [id_to_class.get(i, str(i)) for i in range(num_classes)]
    except Exception as e:
        logger.warning("Could not load class names from metadata (%s). Falling back to numeric labels.", str(e))
        return [str(i) for i in range(config.model.num_classes)]


def evaluate(model, test_loader, device) -> tuple:
    """Runs inference over the test set and returns (all_preds, all_labels)."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(input_values=features)
            logits = outputs.logits
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(all_preds, all_labels, class_names, output_path):
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(22, 20))
    sns.heatmap(
        cm,
        annot=False,
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - ESC-50 Test Set (Fold 5)")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    logger.info("Confusion matrix plot saved to: %s", output_path)
    plt.close()


def main():
    args = parse_args()

    logger.info("=" * 50)
    logger.info("Starting Evaluation on Held-Out Test Set (Fold 5)")
    logger.info("=" * 50)

    config = PipelineConfig.from_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Build metadata and preprocessor exactly as train.py does
    metadata = ESC50Metadata(config)
    metadata.load_and_validate()

    preprocessor = AudioPreprocessor(config)

    use_hf = config.model.use_hf

    # Build datasets/dataloaders exactly as done in training
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        config, metadata, preprocessor, use_hf=use_hf
    )
    _, _, test_loader = build_dataloaders(config, train_dataset, val_dataset, test_dataset)
    logger.info("Test set size: %d samples", len(test_dataset))

    # Build model architecture, then load trained weights
    model = build_ast_model(config)
    model.to(device)

    if not os.path.exists(args.checkpoint):
        logger.error("Checkpoint not found at: %s", args.checkpoint)
        sys.exit(1)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model weights loaded successfully.")

    # Run inference
    all_preds, all_labels = evaluate(model, test_loader, device)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    logger.info("-" * 50)
    logger.info("FINAL TEST SET RESULTS (Fold 5, unseen during training)")
    logger.info("-" * 50)
    logger.info("Accuracy:  %.4f (%.2f%%)", accuracy, accuracy * 100)
    logger.info("Precision: %.4f", precision)
    logger.info("Recall:    %.4f", recall)
    logger.info("F1-Score:  %.4f", f1)
    logger.info("-" * 50)

    # Confusion matrix
    class_names = get_class_names(config)
    output_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(all_preds, all_labels, class_names, output_path)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
