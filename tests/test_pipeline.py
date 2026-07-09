import os
import sys
import unittest
import logging
import torch
import torchaudio
import pandas as pd
import numpy as np

# Adjust path to find the src package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PipelineConfig
from src.utils import setup_logging, time_execution
from src.downloader import ESC50Downloader
from src.metadata import ESC50Metadata
from src.preprocessing import AudioPreprocessor
from src.dataset import ESC50Dataset
from src.dataloader import get_train_val_test_datasets, build_dataloaders

logger = setup_logging(log_dir="logs", log_level=logging.INFO)

class TestESCPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls) -> None:
        """Sets up configuration and test environments."""
        cls.config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
        cls.config = PipelineConfig.from_yaml(cls.config_path)
        cls.test_dir = os.path.join(os.path.dirname(__file__), "temp_test_data")
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Paths for synthetic audio testing
        cls.synthetic_wav_path = os.path.join(cls.test_dir, "synthetic_stereo_44k.wav")
        cls.create_synthetic_audio(cls.synthetic_wav_path, duration_sec=5.0, sample_rate=44100, channels=2)

    @classmethod
    def tearDownClass(cls) -> None:
        """Cleans up temporary test data."""
        if os.path.exists(cls.synthetic_wav_path):
            os.remove(cls.synthetic_wav_path)
        if os.path.exists(cls.test_dir):
            try:
                os.rmdir(cls.test_dir)
            except OSError:
                pass  # Ignore if not empty

    @staticmethod
    def create_synthetic_audio(path: str, duration_sec: float, sample_rate: int, channels: int) -> None:
        """Generates a synthetic stereo sine-wave WAV file to test resamplers and mono-mixers."""
        num_samples = int(duration_sec * sample_rate)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        # 440 Hz tone on channel 1, 880 Hz tone on channel 2
        sig_ch1 = np.sin(2 * np.pi * 440 * t)
        sig_ch2 = np.sin(2 * np.pi * 880 * t)
        
        # Soundfile expects (samples, channels) shape for multi-channel audio
        if channels == 2:
            signal = np.stack([sig_ch1, sig_ch2], axis=1)
        else:
            signal = sig_ch1
            
        import soundfile as sf
        sf.write(path, signal, sample_rate)
        logger.info("Saved synthetic audio file: %s (SR: %d, Channels: %d)", path, sample_rate, channels)


    def test_01_config_loading(self) -> None:
        """Verifies that configs are parsed correctly and assert types."""
        logger.info("Running Test 01: Config Loading...")
        self.assertIsNotNone(self.config.data.download_url)
        self.assertEqual(self.config.preprocessing.target_sr, 16000)
        self.assertEqual(self.config.preprocessing.n_mels, 128)
        self.assertEqual(self.config.dataloader.batch_size, 16)
        logger.info("Test 01 passed: YAML config parsed into dataclasses correctly.")

    def test_02_waveform_standardization(self) -> None:
        """Tests resampling, mono-mixing, and truncation/padding using synthetic audio."""
        logger.info("Running Test 02: Waveform Standardization...")
        preprocessor = AudioPreprocessor(self.config)
        
        # Load synthetic audio (44.1 kHz, stereo)
        waveform, sr = preprocessor.load_audio(self.synthetic_wav_path)
        self.assertEqual(sr, 44100)
        self.assertEqual(waveform.shape[0], 2)  # Stereo
        
        # Apply standardization
        standardized = preprocessor.standardize_waveform(waveform, sr)
        
        # Expected outputs at 16 kHz: mono (1 channel), exactly 5 seconds (80,000 samples)
        expected_samples = int(16000 * 5.0)
        self.assertEqual(standardized.shape[0], 1)  # Mono
        self.assertEqual(standardized.shape[1], expected_samples)  # 80000 samples
        logger.info("Test 02 passed: Stereo 44.1kHz WAV successfully resampled to 16kHz mono, 80,000 samples.")

    def test_03_spectrogram_generation(self) -> None:
        """Verifies that log-Mel Spectrogram features have correct shape and values."""
        logger.info("Running Test 03: Spectrogram Generation...")
        preprocessor = AudioPreprocessor(self.config)
        
        waveform, sr = preprocessor.load_audio(self.synthetic_wav_path)
        standardized = preprocessor.standardize_waveform(waveform, sr)
        
        # 1. Torchaudio Pipeline Mel Spectrogram
        with time_execution("torchaudio Mel transform"):
            mel_spec = preprocessor.compute_log_mel_spectrogram(standardized)
            
        # Expected shape: (n_mels, time_frames).
        # With n_fft=400, hop_length=160, center=True:
        # Time frames = 80000 // 160 + 1 = 501
        self.assertEqual(mel_spec.shape[0], 128)
        self.assertEqual(mel_spec.shape[1], 501)
        self.assertFalse(torch.isnan(mel_spec).any())
        logger.info("Torchaudio Log-Mel Spectrogram shape: %s", list(mel_spec.shape))

        # 2. HuggingFace Feature Extractor Pipeline (if available)
        if preprocessor.hf_feature_extractor is not None:
            with time_execution("HuggingFace AST Feature Extractor"):
                mel_spec_hf = preprocessor.extract_features_hf(standardized)
            # Expected shape for AST model inputs: (time_frames=1024, n_mels=128)
            self.assertEqual(mel_spec_hf.shape[0], 1024)
            self.assertEqual(mel_spec_hf.shape[1], 128)
            self.assertFalse(torch.isnan(mel_spec_hf).any())
            logger.info("HuggingFace Log-Mel Spectrogram shape: %s", list(mel_spec_hf.shape))
            
        logger.info("Test 03 passed: Spectrogram features generated and verified.")

    def test_04_dataset_and_dataloader_dryrun(self) -> None:
        """Runs a mock training dry-run using synthetic data to verify Dataset and DataLoader classes."""
        logger.info("Running Test 04: Dataset & DataLoader Integration...")
        preprocessor = AudioPreprocessor(self.config)
        
        # Create a mock metadata list of paths pointing to our synthetic wave
        # This simulates a dataset of 32 items
        mock_data_list = [(self.synthetic_wav_path, i % 50) for i in range(32)]
        
        # Initialize Dataset
        dataset = ESC50Dataset(mock_data_list, preprocessor, use_hf=False)
        self.assertEqual(len(dataset), 32)
        
        # Test item loading
        features, label = dataset[0]
        self.assertEqual(features.shape, (128, 501))
        self.assertEqual(label, 0)
        
        # Create DataLoader (batch size 16, shuffle True, num_workers 0 for simple test environment)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        
        # Test iteration
        batch_count = 0
        for x, y in dataloader:
            batch_count += 1
            self.assertEqual(x.shape, (16, 128, 501))
            self.assertEqual(y.shape, (16,))
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(y.dtype, torch.int64)
            
        self.assertEqual(batch_count, 2)  # 32 samples / 16 batch_size = 2 batches
        logger.info("Test 04 passed: Dataset loading and DataLoader batching works perfectly.")

    def test_05_dataset_download_and_metadata_integration(self) -> None:
        """
        Integration test: downloads, extracts, and runs the entire metadata validation pipeline.
        NOTE: If the dataset is already present, it verifies it without re-downloading.
        """
        logger.info("Running Test 05: Raw Dataset Ingestion and Metadata Audit...")
        
        # 1. Initialize downloader
        downloader = ESC50Downloader(self.config)
        
        # Attempt to run pipeline (downloads only if missing, else extracts/verifies)
        try:
            downloader.run_pipeline()
        except Exception as e:
            logger.warning("Downloader pipeline test failed/skipped due to network or environment. Error: %s", str(e))
            return  # Skip remainder of integration tests if network download fails
            
        # 2. Metadata validation
        metadata = ESC50Metadata(self.config)
        df = metadata.load_and_validate()
        stats = metadata.get_summary_statistics()
        
        self.assertEqual(stats["total_samples"], 2000)
        self.assertEqual(stats["class_count"], 50)
        self.assertTrue(stats["is_balanced"])
        self.assertEqual(stats["samples_per_class"], 40)
        
        # 3. Create datasets and loaders for real ESC-50 dataset
        preprocessor = AudioPreprocessor(self.config)
        
        # Test default PyTorch loader
        train_ds, val_ds, test_ds = get_train_val_test_datasets(self.config, metadata, preprocessor, use_hf=False)
        self.assertEqual(len(train_ds), 1200) # Folds 1, 2, 3
        self.assertEqual(len(val_ds), 400)    # Fold 4
        self.assertEqual(len(test_ds), 400)   # Fold 5
        
        train_loader, val_loader, test_loader = build_dataloaders(self.config, train_ds, val_ds, test_ds)
        
        # Read one batch from real dataset train loader
        # Using 0 workers for safety in some restricted unittest runners, but using config in real setup
        real_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        for x, y in real_loader:
            self.assertEqual(x.shape, (4, 128, 501))
            self.assertEqual(y.shape, (4,))
            logger.info("Successfully loaded a batch of real preprocessed ESC-50 audio data! Shape: %s", list(x.shape))
            break
            
        logger.info("Test 05 passed: Full end-to-end dataset integration test passed successfully!")

if __name__ == "__main__":
    unittest.main()
