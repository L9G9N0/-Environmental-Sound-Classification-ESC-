import logging
import os
import urllib.request
import zipfile
import ssl
from tqdm import tqdm
from src.config import PipelineConfig

# macOS certificate verification workaround:
# Standard Python installations on macOS often do not configure root certificates,
# causing urllib to fail with SSL verification errors. We set an unverified context as a fallback.
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

logger = logging.getLogger("ESC_Pipeline")


class TqdmUpTo(tqdm):
    """Provides progress updates to tqdm during urllib downloads."""
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class ESC50Downloader:
    """Manages downloading, extraction, and integrity checking of the ESC-50 dataset."""
    
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.raw_dir = config.data.raw_data_dir
        self.zip_path = os.path.join(self.raw_dir, config.data.zip_name)
        
        # Absolute paths for extraction and verification
        self.extracted_marker_dir = os.path.join(self.raw_dir, "ESC-50-master")
        self.metadata_csv = os.path.join(self.raw_dir, "ESC-50-master", "meta", "esc50.csv")
        self.audio_dir = os.path.join(self.raw_dir, "ESC-50-master", "audio")

    def download(self) -> None:
        """Downloads the ESC-50 zip file from GitHub if it does not already exist."""
        os.makedirs(self.raw_dir, exist_ok=True)
        
        if os.path.exists(self.zip_path):
            logger.info("Dataset zip already exists at: %s. Skipping download.", self.zip_path)
            return

        logger.info("Starting download of ESC-50 dataset from: %s", self.config.data.download_url)
        try:
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc="Downloading ESC-50") as t:
                urllib.request.urlretrieve(
                    self.config.data.download_url,
                    filename=self.zip_path,
                    reporthook=t.update_to
                )
            logger.info("Download completed successfully and saved to: %s", self.zip_path)
        except Exception as e:
            logger.error("Failed to download dataset. Error: %s", str(e))
            if os.path.exists(self.zip_path):
                os.remove(self.zip_path)  # Cleanup partial download
            raise e

    def extract(self) -> None:
        """Extracts the downloaded zip file and verifies directory structure."""
        if os.path.exists(self.metadata_csv) and os.path.exists(self.audio_dir):
            logger.info("Dataset already extracted and verified at: %s", self.extracted_marker_dir)
            return

        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"Cannot extract. Zip file not found at: {self.zip_path}")

        logger.info("Extracting %s to %s...", self.zip_path, self.raw_dir)
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # Get list of files to show extraction progress
                file_list = zip_ref.namelist()
                for file in tqdm(file_list, desc="Extracting files"):
                    zip_ref.extract(file, self.raw_dir)
            logger.info("Extraction completed successfully.")
        except Exception as e:
            logger.error("Extraction failed. Error: %s", str(e))
            raise e

    def verify_dataset(self) -> bool:
        """Verifies that all expected dataset assets are present and look correct."""
        if not os.path.exists(self.metadata_csv):
            logger.error("Verification failed: Metadata CSV not found at %s", self.metadata_csv)
            return False
            
        if not os.path.exists(self.audio_dir):
            logger.error("Verification failed: Audio directory not found at %s", self.audio_dir)
            return False

        # Count files in the audio folder
        audio_files = [f for f in os.listdir(self.audio_dir) if f.endswith(".wav")]
        logger.info("Verification check: found %d audio files in %s", len(audio_files), self.audio_dir)
        
        if len(audio_files) != 2000:
            logger.warning("Expected 2000 audio files in ESC-50, but found %d files.", len(audio_files))
            return False

        logger.info("ESC-50 Dataset verification passed. 2000 audio files and metadata CSV verified.")
        return True

    def run_pipeline(self) -> None:
        """Downloads, extracts, and verifies the dataset in a single transaction."""
        self.download()
        self.extract()
        if not self.verify_dataset():
            raise RuntimeError("Dataset verification failed after download and extraction.")
