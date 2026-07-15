import os
import csv
import logging
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import ASTForAudioClassification
from src.config import PipelineConfig

logger = logging.getLogger("ESC_Pipeline")

class ASTTrainer:
    """Manages the training, validation, checkpointing, and evaluation pipeline for AST."""

    def __init__(
        self,
        config: PipelineConfig,
        model: ASTForAudioClassification,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ) -> None:
        """
        Initializes the trainer.

        Args:
            config: Master configuration dataclass.
            model: The AST model to train.
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
            device: Device to train on (cuda, mps, or cpu).
        """
        self.config = config
        self.train_cfg = config.training
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Define optimizer: only optimize parameters that require gradients (handle frozen encoder)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay
        )

        # Initialize Learning Rate Scheduler
        self.scheduler = self._build_scheduler()

        # Directories
        os.makedirs(self.train_cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train_cfg.log_dir, exist_ok=True)
        os.makedirs(self.train_cfg.tb_dir, exist_ok=True)

        # TensorBoard writer
        self.tb_writer = SummaryWriter(log_dir=self.train_cfg.tb_dir)

        # CSV Logging file
        self.csv_log_path = os.path.join(self.train_cfg.log_dir, "training_history.csv")
        self._init_csv_log()

        # Training states
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Builds learning rate scheduler based on config."""
        s_type = self.train_cfg.scheduler_type.lower()
        if s_type == "cosine":
            logger.info("Using CosineAnnealingLR scheduler.")
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_cfg.epochs
            )
        elif s_type == "step":
            logger.info("Using StepLR scheduler. Step size: %d, Gamma: %.2f", self.train_cfg.step_size, self.train_cfg.gamma)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.train_cfg.step_size,
                gamma=self.train_cfg.gamma
            )
        elif s_type == "plateau":
            logger.info("Using ReduceLROnPlateau scheduler. Patience: %d, Gamma: %.2f", self.train_cfg.scheduler_patience, self.train_cfg.gamma)
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.train_cfg.scheduler_patience,
                factor=self.train_cfg.gamma
            )
        else:
            logger.info("No learning rate scheduler configured.")
            return None

    def _init_csv_log(self) -> None:
        """Initializes the CSV log file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_log_path):
            with open(self.csv_log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    def _write_csv_log(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float
    ) -> None:
        """Appends a row of metrics to the CSV log file."""
        with open(self.csv_log_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr])

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Runs a single training epoch over the dataloader."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.perf_counter()
        
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # Move to target device
            features = features.to(self.device)  # Shape: (B, 1024, 128)
            labels = labels.to(self.device)      # Shape: (B,)

            self.optimizer.zero_grad()

            # Forward pass
            # AST expects input_values argument
            outputs = self.model(input_values=features)
            logits = outputs.logits
            
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimization step
            self.optimizer.step()

            # Track statistics
            total_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                step_acc = (preds == labels).float().mean().item()
                logger.info(
                    "Epoch [%d/%d] | Batch [%d/%d] | Loss: %.4f | Acc: %.2f%%",
                    epoch, self.train_cfg.epochs, batch_idx + 1, len(self.train_loader), loss.item(), step_acc * 100
                )

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        duration = time.perf_counter() - start_time
        
        logger.info(
            "Epoch %d Train Summary -> Loss: %.4f | Acc: %.2f%% | Time: %.2fs",
            epoch, epoch_loss, epoch_acc * 100, duration
        )
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Runs validation on the validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in self.val_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(input_values=features)
            logits = outputs.logits
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * features.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        val_loss = total_loss / total
        val_acc = correct / total
        
        logger.info("Validation Summary -> Loss: %.4f | Acc: %.2f%%", val_loss, val_acc * 100)
        return val_loss, val_acc

    def fit(self, resume_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Orchestrates the entire training flow.

        Args:
            resume_path: Path to checkpoint to resume training from.
        """
        if resume_path:
            self.load_checkpoint(resume_path)

        early_stopping_counter = 0
        logger.info("Starting training loop from Epoch %d to %d.", self.start_epoch + 1, self.train_cfg.epochs)

        for epoch in range(self.start_epoch + 1, self.train_cfg.epochs + 1):
            # 1. Train
            train_loss, train_acc = self.train_epoch(epoch)

            # 2. Validate
            val_loss, val_acc = self.validate()

            # 3. Log Learning Rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 4. Update Scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 5. Track History
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            # Write to CSV log
            self._write_csv_log(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)

            # Write to TensorBoard
            self.tb_writer.add_scalar("Loss/Train", train_loss, epoch)
            self.tb_writer.add_scalar("Loss/Val", val_loss, epoch)
            self.tb_writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.tb_writer.add_scalar("Accuracy/Val", val_acc, epoch)
            self.tb_writer.add_scalar("LR", current_lr, epoch)

            # 6. Check for Best Model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                early_stopping_counter = 0
                
                best_path = os.path.join(self.train_cfg.checkpoint_dir, "best_model.pt")
                self.save_checkpoint(epoch, val_loss, val_acc, best_path)
                logger.info("New best model saved to %s (Val Loss: %.4f, Val Acc: %.2f%%)", best_path, val_loss, val_acc * 100)
            else:
                early_stopping_counter += 1
                logger.info("No improvement in validation loss. Early stopping counter: %d/%d", early_stopping_counter, self.train_cfg.early_stopping_patience)

            # 7. Save Latest Checkpoint
            latest_path = os.path.join(self.train_cfg.checkpoint_dir, "latest_model.pt")
            self.save_checkpoint(epoch, val_loss, val_acc, latest_path)


            # 8. Early Stopping Check
            if early_stopping_counter >= self.train_cfg.early_stopping_patience:
                logger.warning("Early stopping triggered. Training stopped at Epoch %d.", epoch)
                break

        # Close TensorBoard writer
        self.tb_writer.close()
        logger.info("Training complete. Best Validation Loss: %.4f | Best Validation Accuracy: %.2f%%", self.best_val_loss, self.best_val_acc * 100)
        return self.history

    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, filepath: str) -> None:
        """Saves a complete training checkpoint for reproducing or resuming training."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_loss": self.best_val_loss,
            "history": self.history
        }
        try:
            torch.save(checkpoint, filepath)
            logger.debug("Successfully saved checkpoint to: %s", filepath)
        except Exception as e:
            logger.error("Failed to save checkpoint. Error: %s", str(e))

    def load_checkpoint(self, filepath: str) -> None:
        """Loads a training checkpoint to resume training."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

        logger.info("Resuming training from checkpoint: %s", filepath)
        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and checkpoint["scheduler_state_dict"]:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.start_epoch = checkpoint["epoch"]
            self.best_val_loss = checkpoint["best_val_loss"]
            self.history = checkpoint["history"]
            logger.info("Checkpoint loaded successfully. Resuming from Epoch %d with best Val Loss: %.4f", self.start_epoch, self.best_val_loss)
        except Exception as e:
            logger.error("Failed to load checkpoint from: %s. Error: %s", filepath, str(e))
            raise e
