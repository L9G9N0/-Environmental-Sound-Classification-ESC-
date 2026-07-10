import logging
from transformers import ASTForAudioClassification
from src.config import PipelineConfig

logger = logging.getLogger("ESC_Pipeline")

def build_ast_model(config: PipelineConfig) -> ASTForAudioClassification:
    """
    Initializes the ASTForAudioClassification model with a custom classification head
    matching the configured number of classes, and optionally freezes the encoder.
    
    Args:
        config: Loaded PipelineConfig containing model and architecture settings.
        
    Returns:
        ASTForAudioClassification model configured for downstream training.
    """
    model_cfg = config.model
    logger.info("Initializing Audio Spectrogram Transformer from checkpoint: %s", model_cfg.checkpoint_name)
    
    try:
        # Load the model with target number of classes
        # ignore_mismatched_sizes=True resets the classification head since it doesn't match AudioSet's 527 classes
        model = ASTForAudioClassification.from_pretrained(
            model_cfg.checkpoint_name,
            num_labels=model_cfg.num_classes,
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        logger.error("Failed to load AST model from checkpoint: %s. Error: %s", model_cfg.checkpoint_name, str(e))
        raise e

    # Freeze the encoder (backbone) if requested (Linear Probing mode)
    if model_cfg.freeze_encoder:
        logger.info("Freezing AST encoder backbone (Linear Probing mode).")
        # In ASTForAudioClassification, the backbone is model.audio_spectrogram_transformer
        if hasattr(model, "audio_spectrogram_transformer"):
            for param in model.audio_spectrogram_transformer.parameters():
                param.requires_grad = False
        else:
            logger.warning("Backbone 'audio_spectrogram_transformer' not found. Freezing entire model except classifier.")
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
    else:
        logger.info("AST encoder backbone is unfrozen (Full Fine-Tuning mode).")

    # Ensure the custom classification head is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Audit and log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info("--- Model Parameter Audit ---")
    logger.info("Total Parameters:     {:,}".format(total_params))
    logger.info("Trainable Parameters: {:,}".format(trainable_params))
    logger.info("Frozen Parameters:    {:,}".format(frozen_params))
    logger.info("-----------------------------")

    return model
