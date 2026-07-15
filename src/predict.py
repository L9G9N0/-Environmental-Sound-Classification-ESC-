import argparse
import os
import sys
import torch
import torch.nn.functional as F

# Adjust path if running directly from src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PipelineConfig
from src.metadata import ESC50Metadata
from src.preprocessing import AudioPreprocessor
from src.model import build_ast_model

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for single-file audio inference."""
    parser = argparse.ArgumentParser(description="Inference script for ESC-50 sound classification")
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the WAV audio file to classify"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the config YAML file (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/checkpoints/best_model.pt",
        help="Path to the trained model checkpoint (default: outputs/checkpoints/best_model.pt)"
    )
    return parser.parse_args()

def main() -> None:
    """Main inference routine."""
    args = parse_args()
    
    # 1. Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
    config = PipelineConfig.from_yaml(args.config)
    
    # 2. Check audio file
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found at {args.audio_path}")
        sys.exit(1)
        
    # 3. Check checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        sys.exit(1)
        
    # 4. Load metadata to get class mapping
    try:
        metadata = ESC50Metadata(config)
        metadata.load_and_validate()
        id_to_class = metadata.id_to_class
    except Exception as e:
        print(f"Error: Could not load metadata CSV: {e}")
        sys.exit(1)
        
    # 5. Initialize preprocessor and process the audio file
    print(f"Preprocessing audio: {args.audio_path}")
    preprocessor = AudioPreprocessor(config)
    try:
        # Get preprocessed features: shape (time_frames, n_mels)
        features = preprocessor.process_file(args.audio_path, use_hf=config.model.use_hf)
    except Exception as e:
        print(f"Error preprocessing audio file: {e}")
        sys.exit(1)
        
    # Add batch dimension: shape (1, time_frames, n_mels)
    features = features.unsqueeze(0)
    
    # 6. Load model
    print(f"Loading model checkpoint: {args.checkpoint}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Build AST Model
    model = build_ast_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # 7. Run Inference
    features = features.to(device)
    with torch.no_grad():
        outputs = model(input_values=features)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).squeeze(0)
        
    # 8. Print predictions
    topk_probs, topk_indices = torch.topk(probs, k=5)
    
    print("\n" + "="*50)
    print("      ESC SOUND CLASSIFICATION RESULTS      ")
    print("="*50)
    print(f"File: {args.audio_path}")
    print("-"*50)
    
    top_class_id = topk_indices[0].item()
    top_class_prob = topk_probs[0].item()
    top_class_name = id_to_class[top_class_id]
    print(f"PREDICTED CLASS: {top_class_name.upper()} (Confidence: {top_class_prob*100:.2f}%)")
    print("-"*50)
    print("Top 5 Predictions:")
    for i in range(5):
        idx = topk_indices[i].item()
        prob = topk_probs[i].item()
        class_name = id_to_class[idx]
        print(f"  {i+1}. {class_name:<25} : {prob*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
