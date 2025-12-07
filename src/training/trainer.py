"""
Trainer for fine-tuning MoE models with imbalance loss.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional
from tqdm import tqdm
import json
from pathlib import Path


class MoETrainer:
    """Trainer for MoE models with expert imbalance regularization."""
    
    def __init__(self,
                 model,
                 train_dataloader: DataLoader,
                 eval_dataloader: Optional[DataLoader] = None,
                 learning_rate: float = 5e-5,
                 num_epochs: int = 3,
                 warmup_steps: int = 100,
                 device: str = "cuda",
                 output_dir: str = "results/checkpoints",
                 log_steps: int = 10,
                 eval_steps: int = 100,
                 save_steps: int = 500):
        """
        Initialize trainer.
        
        Args:
            model: MoE model (wrapped with imbalance loss)
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            device: Device to train on
            output_dir: Directory to save checkpoints
            log_steps: Steps between logging
            eval_steps: Steps between evaluations
            save_steps: Steps between saving checkpoints
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_steps = log_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.num_epochs = num_epochs
        
        # Optimizer and scheduler
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.global_step = 0
        self.training_history = []
        
    def train_step(self, batch: Dict) -> Dict:
        """
        Perform one training step.
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        total_loss, loss_dict = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        
        return loss_dict
    
    def evaluate(self) -> Dict:
        """
        Evaluate the model.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_imbalance_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                total_loss_val, loss_dict = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += loss_dict['total_loss']
                total_lm_loss += loss_dict['lm_loss']
                total_imbalance_loss += loss_dict['imbalance_loss']
                num_batches += 1
        
        return {
            'eval_total_loss': total_loss / num_batches,
            'eval_lm_loss': total_lm_loss / num_batches,
            'eval_imbalance_loss': total_imbalance_loss / num_batches
        }
    
    def train(self):
        """Run the training loop."""
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Total steps: {len(self.train_dataloader) * self.num_epochs}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            epoch_loss = 0.0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")
            
            for batch in progress_bar:
                loss_dict = self.train_step(batch)
                epoch_loss += loss_dict['total_loss']
                
                # Logging
                if self.global_step % self.log_steps == 0:
                    avg_loss = epoch_loss / (self.global_step % len(self.train_dataloader) + 1)
                    progress_bar.set_postfix({
                        'loss': f"{loss_dict['total_loss']:.4f}",
                        'lm_loss': f"{loss_dict['lm_loss']:.4f}",
                        'imb_loss': f"{loss_dict['imbalance_loss']:.4f}"
                    })
                    
                    self.training_history.append({
                        'step': self.global_step,
                        'epoch': epoch + 1,
                        **loss_dict
                    })
                
                # Evaluation
                if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        print(f"\nStep {self.global_step} - Eval: {eval_metrics}")
                        self.training_history[-1].update(eval_metrics)
                
                # Checkpoint saving
                if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
            
            # End of epoch evaluation
            eval_metrics = self.evaluate()
            if eval_metrics:
                print(f"\nEnd of Epoch {epoch + 1} - Eval: {eval_metrics}")
        
        # Save final model
        self.save_checkpoint("final")
        self.save_training_history()
        
        print("\nTraining completed!")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), checkpoint_dir / "model.pt")
        
        # Save optimizer and scheduler
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step
        }, checkpoint_dir / "training_state.pt")
        
        print(f"Checkpoint saved to {checkpoint_dir}")
    
    def save_training_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        self.model.load_state_dict(torch.load(checkpoint_path / "model.pt"))
        
        # Load training state
        training_state = torch.load(checkpoint_path / "training_state.pt")
        self.optimizer.load_state_dict(training_state['optimizer'])
        self.scheduler.load_state_dict(training_state['scheduler'])
        self.global_step = training_state['global_step']
        
        print(f"Checkpoint loaded from {checkpoint_path}")


if __name__ == "__main__":
    print("MoE Trainer module - use from training scripts")
