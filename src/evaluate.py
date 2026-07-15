import argparse
import logging
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Adjust path if running directly from src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PipelineConfig
from src.utils import setup_logging, time_execution
from src.metadata import ESC50Metadata
from src.preprocessing import AudioPreprocessor
from src.dataloader import get_train_val_test_datasets, build_dataloaders
from src.model import build_ast_model

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(description="AST Environmental Sound Classification Evaluator")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pt",
        help="Path to the trained model checkpoint to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save evaluation reports and plots (default: outputs)"
    )
    return parser.parse_args()

def get_device() -> torch.device:
    """Selects the best available accelerator (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str) -> None:
    """Generates and saves a high-quality 50x50 confusion matrix plot using Matplotlib."""
    fig, ax = plt.subplots(figsize=(24, 24))
    
    # Draw confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Configure ticks
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    
    # Set titles and labels
    ax.set_title("ESC-50 Confusion Matrix (Unseen Fold 5 Test Split)", fontsize=20, pad=20, fontweight='bold')
    ax.set_xlabel("Predicted Label", fontsize=14, labelpad=15)
    ax.set_ylabel("True Label", fontsize=14, labelpad=15)
    
    # Draw gridlines for separation
    ax.set_xticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(class_names) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Annotate significant confusion values inside cells if we want,
    # but with 50x50, annotating every cell makes it too cluttered.
    # Instead, we will annotate cells that have a value >= 1.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                ax.text(
                    j, i, format(val, 'd'),
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=8,
                    fontweight='bold' if i == j else 'normal'
                )
                
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main() -> None:
    """Main evaluation orchestrator."""
    args = parse_args()
    
    # 1. Load configuration
    try:
        config = PipelineConfig.from_yaml(args.config)
    except Exception as e:
        print(f"CRITICAL: Failed to load configuration from {args.config}. Error: {e}")
        sys.exit(1)
        
    # 2. Setup Logging
    logger = setup_logging(log_dir=config.training.log_dir, log_level=logging.INFO)
    logger.info("==================================================")
    logger.info("Starting Audio Spectrogram Transformer Evaluation")
    logger.info("==================================================")
    logger.info("Using configuration: %s", args.config)
    logger.info("Evaluating checkpoint: %s", args.checkpoint)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Load Metadata
    metadata = ESC50Metadata(config)
    metadata.load_and_validate()
    class_names = [metadata.id_to_class[i] for i in range(config.model.num_classes)]
    
    # 4. Initialize Preprocessor
    preprocessor = AudioPreprocessor(config)
    
    # 5. Create Test Dataset and DataLoader
    use_hf = config.model.use_hf
    train_ds, val_ds, test_ds = get_train_val_test_datasets(
        config,
        metadata,
        preprocessor,
        use_hf=use_hf
    )
    
    _, _, test_loader = build_dataloaders(config, train_ds, val_ds, test_ds)
    logger.info("Test split size: %d samples", len(test_ds))
    
    # 6. Determine device and build model
    device = get_device()
    logger.info("Running evaluation on device: %s", device)
    
    model = build_ast_model(config)
    
    # 7. Load model weights
    if not os.path.exists(args.checkpoint):
        logger.error("Checkpoint file not found: %s", args.checkpoint)
        sys.exit(1)
        
    logger.info("Loading checkpoint weights from: %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # 8. Evaluation loop
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    logger.info("Running inference on test loader...")
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Testing"):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(input_values=features)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    test_loss = total_loss / len(test_ds)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 9. Compute Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Precision, Recall, F1 for Macro and Weighted
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    logger.info("--- Evaluation Performance Summary ---")
    logger.info("Test Loss:          %.4f", test_loss)
    logger.info("Test Accuracy:      %.2f%%", accuracy * 100)
    logger.info("Macro Precision:    %.4f", macro_precision)
    logger.info("Macro Recall:       %.4f", macro_recall)
    logger.info("Macro F1-Score:     %.4f", macro_f1)
    logger.info("Weighted Precision: %.4f", weighted_precision)
    logger.info("Weighted Recall:    %.4f", weighted_recall)
    logger.info("Weighted F1-Score:  %.4f", weighted_f1)
    logger.info("--------------------------------------")
    
    # Save metrics to JSON file
    metrics = {
        "test_loss": float(test_loss),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1)
    }
    
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Saved evaluation metrics JSON report to: %s", metrics_path)
    
    # 10. Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(config.model.num_classes))
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    logger.info("Plotting and saving 50x50 confusion matrix to: %s", cm_path)
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # 11. Print top confusions
    # Find indices where true != pred
    confused_mask = all_labels != all_preds
    confused_trues = all_labels[confused_mask]
    confused_preds = all_preds[confused_mask]
    
    confusion_pairs = {}
    for t, p in zip(confused_trues, confused_preds):
        pair = (class_names[t], class_names[p])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
    sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top 5 Confusions (True Class -> Predicted Class):")
    for pair, count in sorted_confusions[:5]:
        logger.info("  %s -> %s: %d times", pair[0], pair[1], count)

    logger.info("Evaluation pipeline completed successfully.")

if __name__ == "__main__":
    main()
