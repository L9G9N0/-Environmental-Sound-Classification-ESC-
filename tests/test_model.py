import os
import sys
import unittest
import shutil
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Adjust path to find the src package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import PipelineConfig
from src.utils import setup_logging
from src.model import build_ast_model
from src.trainer import ASTTrainer

logger = setup_logging(log_dir="logs", log_level=logging.INFO)

class SyntheticASTDataset(Dataset):
    """Generates synthetic audio features matching the AST input shape [1024, 128] for rapid tests."""
    def __init__(self, size: int = 8, num_classes: int = 50) -> None:
        self.size = size
        self.num_classes = num_classes
        # Generate random inputs and deterministic labels
        self.features = torch.randn(size, 1024, 128)
        self.labels = torch.randint(0, num_classes, (size,))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.features[idx], int(self.labels[idx])

class TestASTModeling(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Sets up test configuration and temporary outputs directory."""
        cls.config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
        cls.config = PipelineConfig.from_yaml(cls.config_path)
        
        # Override paths for test isolation
        cls.test_out_dir = os.path.join(os.path.dirname(__file__), "temp_test_outputs")
        cls.config.training.checkpoint_dir = os.path.join(cls.test_out_dir, "checkpoints")
        cls.config.training.log_dir = os.path.join(cls.test_out_dir, "logs")
        cls.config.training.tb_dir = os.path.join(cls.test_out_dir, "tb")
        
        # Select device
        if torch.cuda.is_available():
            cls.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            cls.device = torch.device("mps")
        else:
            cls.device = torch.device("cpu")
            
        logger.info("Test device selected: %s", cls.device)

    @classmethod
    def tearDownClass(cls) -> None:
        """Cleans up temporary outputs created during test execution."""
        if os.path.exists(cls.test_out_dir):
            shutil.rmtree(cls.test_out_dir)
            logger.info("Removed temporary test outputs directory: %s", cls.test_out_dir)

    def test_01_model_loading_and_freezing(self) -> None:
        """Verifies model loader correctly initializes layers, custom classes, and freezing states."""
        logger.info("Running Test 01: Model Loading & Freezing State...")
        
        # 1. Test Frozen Encoder (Linear Probing)
        self.config.model.freeze_encoder = True
        model_frozen = build_ast_model(self.config)
        self.assertEqual(model_frozen.classifier.dense.out_features, 50)
        
        # Check that encoder weights are frozen and classifier weights are trainable
        for name, param in model_frozen.named_parameters():
            if "classifier" in name:
                self.assertTrue(param.requires_grad, f"Classifier parameter {name} should be trainable.")
            elif "audio_spectrogram_transformer" in name:
                self.assertFalse(param.requires_grad, f"Encoder parameter {name} should be frozen.")

        # 2. Test Unfrozen Encoder (Full Fine-Tuning)
        self.config.model.freeze_encoder = False
        model_unfrozen = build_ast_model(self.config)
        
        # Check that all encoder parameters are now trainable
        for name, param in model_unfrozen.named_parameters():
            self.assertTrue(param.requires_grad, f"Parameter {name} should be trainable in fine-tuning mode.")
            
        logger.info("Test 01 passed: Model loaded, classification head swapped, and encoder frozen/unfrozen successfully.")

    def test_02_trainer_initialization(self) -> None:
        """Tests that the ASTTrainer class instantiates and sets up logs, schedulers, and directories."""
        logger.info("Running Test 02: Trainer Initialization...")
        self.config.model.freeze_encoder = True
        model = build_ast_model(self.config)
        
        train_ds = SyntheticASTDataset(size=8)
        val_ds = SyntheticASTDataset(size=4)
        
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
        
        trainer = ASTTrainer(
            config=self.config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device
        )
        
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.criterion)
        self.assertTrue(os.path.exists(self.config.training.log_dir))
        self.assertTrue(os.path.exists(trainer.csv_log_path))
        logger.info("Test 02 passed: Trainer initialized with correct configurations and directories.")

    def test_03_training_dryrun(self) -> None:
        """Runs a mock training dry-run of 1 epoch to verify forward pass, backward pass, and eval logic."""
        logger.info("Running Test 03: Training Dry-run (1 Epoch)...")
        self.config.model.freeze_encoder = True
        self.config.training.epochs = 1
        
        model = build_ast_model(self.config)
        train_ds = SyntheticASTDataset(size=8)
        val_ds = SyntheticASTDataset(size=4)
        
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
        
        trainer = ASTTrainer(
            config=self.config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device
        )
        
        # Run fit
        history = trainer.fit()
        
        self.assertEqual(len(history["epoch"]), 1)
        self.assertIn("train_loss", history)
        self.assertIn("val_loss", history)
        self.assertEqual(len(history["train_loss"]), 1)
        
        # Check that latest and best model checkpoints were written
        self.assertTrue(os.path.exists(os.path.join(self.config.training.checkpoint_dir, "latest_model.pt")))
        self.assertTrue(os.path.exists(os.path.join(self.config.training.checkpoint_dir, "best_model.pt")))
        logger.info("Test 03 passed: One training epoch dry-run and validation ran successfully without crashes.")

    def test_04_checkpoint_saving_and_loading(self) -> None:
        """Verifies that checkpoints can be saved, reloaded, and training can be resumed correctly."""
        logger.info("Running Test 04: Checkpoint Saving & Resuming...")
        self.config.model.freeze_encoder = True
        self.config.training.epochs = 1
        
        model = build_ast_model(self.config)
        train_ds = SyntheticASTDataset(size=8)
        val_ds = SyntheticASTDataset(size=4)
        
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
        
        trainer1 = ASTTrainer(
            config=self.config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device
        )
        
        # Run 1 epoch to save states
        trainer1.fit()
        checkpoint_path = os.path.join(self.config.training.checkpoint_dir, "latest_model.pt")
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # Load states in a new trainer instance
        model2 = build_ast_model(self.config)
        trainer2 = ASTTrainer(
            config=self.config,
            model=model2,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device
        )
        
        trainer2.load_checkpoint(checkpoint_path)
        
        # Verify resumed parameters
        self.assertEqual(trainer2.start_epoch, 1)
        self.assertEqual(trainer2.best_val_loss, trainer1.best_val_loss)
        self.assertEqual(len(trainer2.history["epoch"]), 1)
        logger.info("Test 04 passed: Checkpoint successfully saved, reloaded, and resumed state verified.")

    def test_05_overfit_sanity_check(self) -> None:
        """Sanity check: overfits a single batch of size 2 over 15 steps. The loss must decrease."""
        logger.info("Running Test 05: Overfitting Sanity Check...")
        self.config.model.freeze_encoder = True
        
        model = build_ast_model(self.config).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
        
        # Create a single batch of 2 elements
        x = torch.randn(2, 1024, 128).to(self.device)
        y = torch.tensor([12, 34]).to(self.device)
        
        # Overfit loop
        initial_loss = None
        final_loss = None
        
        model.train()
        for step in range(15):
            optimizer.zero_grad()
            outputs = model(input_values=x)
            loss = criterion(outputs.logits, y)
            loss.backward()
            optimizer.step()
            
            if step == 0:
                initial_loss = loss.item()
            if step == 14:
                final_loss = loss.item()
                
            logger.info("  Step %d/15 | Loss: %.6f", step + 1, loss.item())
            
        logger.info("Initial Loss: %.6f | Final Loss: %.6f", initial_loss, final_loss)
        self.assertLess(final_loss, initial_loss, "Loss did not decrease during overfitting test.")
        logger.info("Test 05 passed: Overfitting test successfully verified that gradients flow and loss decreases.")

if __name__ == "__main__":
    unittest.main()
