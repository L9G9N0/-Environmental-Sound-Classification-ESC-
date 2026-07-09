import logging
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from transformers import ASTFeatureExtractor
from src.config import PipelineConfig

logger = logging.getLogger("ESC_Pipeline")

class AudioPreprocessor:
    """Handles raw audio loading, DSP operations, and feature conversion to Log-Mel Spectrograms."""
    
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config.preprocessing
        self.target_sr = self.config.target_sr
        self.target_samples = int(self.target_sr * self.config.duration_sec)
        
        # Define Torchaudio MelSpectrogram pipeline
        # n_fft is window size (400 samples is 25ms at 16kHz)
        # hop_length is step size (160 samples is 10ms at 16kHz)
        self.mel_transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=self.target_sr,
                n_fft=self.config.n_fft,
                win_length=self.config.n_fft,
                hop_length=self.config.hop_length,
                n_mels=self.config.n_mels,
                power=self.config.power
            ),
            T.AmplitudeToDB(top_db=80.0)
        )
        
        # Load HuggingFace AST Feature Extractor as an alternative or reference
        try:
            self.hf_feature_extractor = ASTFeatureExtractor(
                sampling_rate=self.target_sr,
                num_mel_bins=self.config.n_mels,
                max_length=1024, # default AST sequence length (10 seconds, pads shorter clips)
                do_normalize=self.config.normalize,
                mean=self.config.mean,
                std=self.config.std
            )
            logger.info("HuggingFace ASTFeatureExtractor initialized successfully.")
        except Exception as e:
            logger.warning("Could not initialize HuggingFace ASTFeatureExtractor. Fallback to torchaudio only. Error: %s", str(e))
            self.hf_feature_extractor = None

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Loads a WAV file from disk. Uses torchaudio with a soundfile fallback for robustness."""
        try:
            waveform, sr = torchaudio.load(file_path)
            return waveform, sr
        except Exception as e:
            logger.debug("torchaudio.load failed or backend missing. Falling back to soundfile loader. Detail: %s", str(e))
            try:
                import soundfile as sf
                data, sr = sf.read(file_path)
                waveform = torch.tensor(data, dtype=torch.float32)
                # Convert shape (samples, channels) to (channels, samples)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.T
                return waveform, sr
            except Exception as sf_e:
                logger.error("Both torchaudio and soundfile failed to load: %s. Error: %s", file_path, str(sf_e))
                raise sf_e


    def standardize_waveform(self, waveform: torch.Tensor, src_sr: int) -> torch.Tensor:
        """Applies resampling, mono-mixing, and length standardization (padding/truncation)."""
        # 1. Resample to target rate (e.g. 16 kHz)
        if src_sr != self.target_sr:
            resampler = T.Resample(orig_freq=src_sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # 2. Convert to Mono (average channels if stereo or multi-channel)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 3. Standardize duration (truncate if too long, zero-pad if too short)
        num_channels, num_samples = waveform.shape
        if num_samples > self.target_samples:
            waveform = waveform[:, :self.target_samples]
        elif num_samples < self.target_samples:
            padding_needed = self.target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))

        return waveform

    def compute_log_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Generates log-Mel spectrogram using torchaudio pipeline."""
        # mel_transform expects shape (..., time)
        mel_spec = self.mel_transform(waveform)  # Shape: (1, n_mels, time_frames)
        
        # Remove channel dimension: (n_mels, time_frames)
        mel_spec = mel_spec.squeeze(0)
        
        # Apply normalization using pre-computed model statistics
        if self.config.normalize:
            mel_spec = (mel_spec - self.config.mean) / self.config.std
            
        return mel_spec

    def extract_features_hf(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts features using HuggingFace's ASTFeatureExtractor (100% official compatibility)."""
        if self.hf_feature_extractor is None:
            raise RuntimeError("HuggingFace ASTFeatureExtractor was not initialized.")
            
        # Waveform must be flattened 1D array for the HF extractor
        waveform_numpy = waveform.squeeze(0).numpy()
        
        # Extract features (automatically handles log-mel and padding/normalization)
        inputs = self.hf_feature_extractor(waveform_numpy, sampling_rate=self.target_sr, return_tensors="pt")
        
        # Returns shape: (batch_size=1, time_frames=1024, n_mels=128)
        # We squeeze the batch dimension to return (time_frames, n_mels)
        return inputs["input_values"].squeeze(0)

    def process_file(self, file_path: str, use_hf: bool = False) -> torch.Tensor:
        """Combines load, standardize, and feature extraction in a single call."""
        waveform, sr = self.load_audio(file_path)
        standard_waveform = self.standardize_waveform(waveform, sr)
        
        if use_hf:
            return self.extract_features_hf(standard_waveform)
        else:
            return self.compute_log_mel_spectrogram(standard_waveform)
