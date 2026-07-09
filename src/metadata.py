import logging
import os
from typing import Dict, List, Tuple
import pandas as pd
from src.config import PipelineConfig

logger = logging.getLogger("ESC_Pipeline")

class ESC50Metadata:
    """Loads, validates, and analyzes metadata for the ESC-50 dataset."""
    
    REQUIRED_COLUMNS = {"filename", "fold", "target", "category", "esc10"}

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.csv_path = config.data.metadata_csv
        self.df: pd.DataFrame = pd.DataFrame()
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

    def load_and_validate(self) -> pd.DataFrame:
        """Loads metadata CSV and performs rigorous validation checks on columns and data types."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Metadata CSV not found at expected path: {self.csv_path}")
            
        logger.info("Loading ESC-50 metadata from: %s", self.csv_path)
        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            logger.error("Failed to read metadata CSV. Error: %s", str(e))
            raise e

        # 1. Validate Columns
        missing_cols = self.REQUIRED_COLUMNS - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Metadata CSV is missing required columns: {missing_cols}")

        # 2. Check for null values
        null_counts = self.df[list(self.REQUIRED_COLUMNS)].isnull().sum()
        if null_counts.any():
            logger.warning("Null values found in metadata columns:\n%s", null_counts)
            self.df.dropna(subset=list(self.REQUIRED_COLUMNS), inplace=True)
            logger.info("Dropped rows with null values in required columns.")

        # 3. Check for duplicates
        duplicate_filenames = self.df.duplicated(subset=["filename"]).sum()
        if duplicate_filenames > 0:
            logger.warning("Duplicate filenames detected in metadata: %d. Keeping first occurrences.", duplicate_filenames)
            self.df.drop_duplicates(subset=["filename"], keep="first", inplace=True)

        # 4. Validate folds and target ranges
        invalid_folds = self.df[(self.df["fold"] < 1) | (self.df["fold"] > 5)]
        if not invalid_folds.empty:
            raise ValueError(f"Folds must be between 1 and 5. Found invalid folds:\n{invalid_folds}")

        invalid_targets = self.df[(self.df["target"] < 0) | (self.df["target"] > 49)]
        if not invalid_targets.empty:
            raise ValueError(f"Target IDs must be between 0 and 49. Found invalid targets:\n{invalid_targets}")

        # 5. Build class mappings
        unique_classes = self.df[["target", "category"]].drop_duplicates().sort_values("target")
        self.class_to_id = dict(zip(unique_classes["category"], unique_classes["target"]))
        self.id_to_class = dict(zip(unique_classes["target"], unique_classes["category"]))

        logger.info("Metadata loaded successfully. Row count: %d. Classes mapped: %d.", len(self.df), len(self.class_to_id))
        return self.df

    def get_summary_statistics(self) -> Dict[str, any]:
        """Performs Exploratory Data Analysis (EDA) on the metadata distributions."""
        if self.df.empty:
            raise RuntimeError("Metadata not loaded. Call load_and_validate() first.")

        total_samples = len(self.df)
        fold_counts = self.df["fold"].value_counts().to_dict()
        class_counts = self.df["category"].value_counts().to_dict()
        esc10_counts = self.df["esc10"].value_counts().to_dict()
        
        # Verify perfect balance
        is_balanced = len(set(class_counts.values())) == 1
        samples_per_class = list(class_counts.values())[0] if is_balanced else -1

        summary = {
            "total_samples": total_samples,
            "fold_distribution": fold_counts,
            "class_count": len(class_counts),
            "is_balanced": is_balanced,
            "samples_per_class": samples_per_class,
            "esc10_samples": esc10_counts.get(True, 0)
        }
        
        logger.info("--- Metadata Summary Statistics ---")
        logger.info("Total audio clips: %d", total_samples)
        logger.info("Number of classes: %d (Is balanced: %s)", len(class_counts), str(is_balanced))
        if is_balanced:
            logger.info("Audio clips per class: %d", samples_per_class)
        logger.info("Fold distribution: %s", fold_counts)
        logger.info("ESC-10 subset clips: %d", summary["esc10_samples"])
        logger.info("----------------------------------")
        
        return summary

    def get_audio_paths_and_labels(self, folds: List[int]) -> List[Tuple[str, int]]:
        """Returns a list of tuples (absolute_audio_path, class_id) for the specified folds."""
        if self.df.empty:
            raise RuntimeError("Metadata not loaded. Call load_and_validate() first.")
            
        filtered_df = self.df[self.df["fold"].isin(folds)]
        
        data_list = []
        for _, row in filtered_df.iterrows():
            # Construct path relative to config.data.audio_dir
            audio_path = os.path.join(self.config.data.audio_dir, row["filename"])
            data_list.append((audio_path, int(row["target"])))
            
        return data_list
