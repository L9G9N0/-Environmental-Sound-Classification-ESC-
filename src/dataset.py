import logging
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
from src.preprocessing import AudioPreprocessor

logger = logging.getLogger("ESC_Pipeline")

class ESC50Dataset(Dataset):
    """Custom PyTorch Dataset for ESC-50 Environmental Sound Classification."""
    
    def __init__(
        self, 
        data_list: List[Tuple[str, int]], 
        preprocessor: AudioPreprocessor,
        use_hf: bool = False
    ) -> None:
        """
        Initializes the dataset.
        
        Args:
            data_list: List of tuples (file_path, class_id).
            preprocessor: Configured instance of AudioPreprocessor.
            use_hf: If True, uses HuggingFace ASTFeatureExtractor, producing (time_frames, n_mels).
                    If False, uses torchaudio transforms, producing (n_mels, time_frames).
        """
        self.data_list = data_list
        self.preprocessor = preprocessor
        self.use_hf = use_hf

    def __len__(self) -> int:
        """Returns the total number of samples in this dataset split."""
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Fetches, preprocesses, and returns a single log-Mel spectrogram and its label.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            features: Tensor of log-Mel spectrogram features.
            label: Integer target class ID.
        """
        file_path, label = self.data_list[idx]
        
        try:
            # Process raw WAV to 2D log-Mel spectrogram
            features = self.preprocessor.process_file(file_path, use_hf=self.use_hf)
            return features, label
            
        except Exception as e:
            logger.error(
                "Error processing dataset index %d: file %s, label %d. Error: %s",
                idx, file_path, label, str(e)
            )
            # Raise an descriptive exception to prevent silent data corruption in the training loop
            raise RuntimeError(f"Data loading failed at index {idx} for file {file_path}: {e}")
