#!/usr/bin/env python3
"""
Training script for GPT-OSS with Mixture of Experts.

This script demonstrates how to train the GPT-OSS model with proper
handling of MoE auxiliary losses and modern training techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import math

from model import GPTOSS
from config import get_config
from data import prepare_shakespeare_data, SimpleTokenizer
from utils import (
    count_parameters, get_device, set_seed, save_model_checkpoint,
    load_model_checkpoint, format_time, print_model_summary
)
from generate import TextGenerator

# ============================================================================
# TRAINING UTILITIES
# Comprehensive training script for GPT-OSS
# ============================================================================

class Trainer:
    """Training class for GPT-OSS models."""
    
    def __init__(self, model: GPTOSS, config: Any, device: torch.device,
                 tokenizer: Optional[SimpleTokenizer] = None):
        self.model = model
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        
        # Move model to device
        self.model.to(device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def train_step(self, batch: Dict[str, torch.Tensor], 
                  optimizer: torch.optim.Optimizer,
                  criterion: nn.Module,
                  gradient_clip_val: Optional[float] = None) -> Dict[str, float]:
        """Single training step.
        
        Args:
            batch: Batch of data
            optimizer: Optimizer
            criterion: Loss function
            gradient_clip_val: Gradient clipping value
            
        Returns:
            Dictionary with loss metrics
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids)
        
        # Calculate main loss
        logits = outputs['logits']
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Add auxiliary loss if using MoE
        aux_loss = 0.0
        if 'aux_loss' in outputs and outputs['aux_loss'] is not None:
            aux_loss = outputs['aux_loss']
            loss += 0.01 * aux_loss  # Small weight for auxiliary loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
        
        # Optimizer step
        optimizer.step()
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
        }
    
    def evaluate(self, dataloader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Evaluate model on validation data.
        
        Args:
            dataloader: Validation data loader
            criterion: Loss function
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        total_aux_loss = 0
        num_batches = 0
        num_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                logits = outputs['logits']
                
                # Calculate loss
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                aux_loss = 0.0
                if 'aux_loss' in outputs and outputs['aux_loss'] is not None:
                    aux_loss = outputs['aux_loss'].item()
                    loss += 0.01 * outputs['aux_loss']
                
                total_loss += loss.item()
                total_aux_loss += aux_loss
                num_batches += 1
                num_tokens += (labels != -100).sum().item()  # Count non-padding tokens
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_aux_loss = total_aux_loss / num_batches if num_batches > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'aux_loss': avg_aux_loss,
            'perplexity': perplexity,
            'num_tokens': num_tokens
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             num_epochs: int, learning_rate: float = 1e-4,
             weight_decay: float = 0.01, gradient_clip_val: float = 1.0,
             save_dir: str = "checkpoints", save_every: int = 1,
             eval_every: int = 1, log_every: int = 100) -> None:
        """Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            gradient_clip_val: Gradient clipping value
            save_dir: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            eval_every: Evaluate every N epochs
            log_every: Log every N steps
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Cosine annealing scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Total training steps: {total_steps:,}")
        print(f"Device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            self.model.train()
            epoch_loss = 0
            epoch_aux_loss = 0
            num_batches = 0
            
            for step, batch in enumerate(train_loader):
                # Training step
                metrics = self.train_step(batch, optimizer, criterion, gradient_clip_val)
                
                epoch_loss += metrics['loss']
                epoch_aux_loss += metrics['aux_loss']
                num_batches += 1
                
                # Update scheduler
                scheduler.step()
                
                # Logging
                if step % log_every == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1}/{num_epochs}, Step {step+1}/{len(train_loader)}, "
                          f"Loss: {metrics['loss']:.4f}, LR: {current_lr:.2e}")
            
            # Calculate epoch metrics
            avg_train_loss = epoch_loss / num_batches
            avg_train_aux_loss = epoch_aux_loss / num_batches
            
            self.train_losses.append(avg_train_loss)
            self.learning_rates.append(scheduler.get_last_lr()[0])
            
            # Evaluation phase
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.evaluate(val_loader, criterion)
                self.val_losses.append(val_metrics['loss'])
                
                epoch_time = time.time() - epoch_start_time
                
                print(f"Epoch {epoch+1}/{num_epochs} completed in {format_time(epoch_time)}")
                print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                print(f"Val Perplexity: {val_metrics['perplexity']:.2f}")
                
                if avg_train_aux_loss > 0:
                    print(f"Train Aux Loss: {avg_train_aux_loss:.4f}, Val Aux Loss: {val_metrics['aux_loss']:.4f}")
                
                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    best_model_path = os.path.join(save_dir, "best_model.pt")
                    save_model_checkpoint(
                        self.model, optimizer, epoch, val_metrics['loss'],
                        best_model_path, self.config
                    )
                    print(f"New best model saved with val loss: {val_metrics['loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                save_model_checkpoint(
                    self.model, optimizer, epoch, avg_train_loss,
                    checkpoint_path, self.config
                )
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def generate_sample(self, prompt: str = "The", max_new_tokens: int = 100) -> str:
        """Generate a sample text for evaluation.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if self.tokenizer is None:
            return "No tokenizer available for text generation"
        
        generator = TextGenerator(self.model, self.tokenizer, self.device)
        return generator.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.8)

class GPTOSSTrainer:
    """Trainer class for GPT-OSS model.
    
    Handles training loop, loss computation, and optimization
    with proper MoE auxiliary loss handling.
    """
    
    def __init__(self, model: GPTOSS, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._configure_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._configure_scheduler()
        
        # Training state
        self.step = 0
        self.epoch = 0
        
    def _configure_optimizer(self) -> optim.Optimizer:
        """Configure optimizer with weight decay and parameter grouping."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases, layer norms, and embeddings
                if any(nd in name for nd in ['bias', 'ln_', 'wte', 'wpe']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.get('weight_decay', 0.1)},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return optim.AdamW(
            optimizer_groups,
            lr=self.config.get('learning_rate', 3e-4),
            betas=self.config.get('betas', (0.9, 0.95)),
            eps=self.config.get('eps', 1e-8)
        )
    
    def _configure_scheduler(self):
        """Configure learning rate scheduler with warmup and cosine decay."""
        warmup_steps = self.config.get('warmup_steps', 1000)
        max_steps = self.config.get('max_steps', 100000)
        min_lr_ratio = self.config.get('min_lr_ratio', 0.1)
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (max_steps - warmup_steps)
                progress = min(progress, 1.0)
                return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        logits, loss, aux_loss = self.model(x, y)
        
        # Total loss includes auxiliary loss for MoE load balancing
        total_loss = loss + aux_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['grad_clip']
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'aux_loss': aux_loss.item(),
            'total_loss': total_loss.item(),
            'lr': self.scheduler.get_last_lr()[0],
            'step': self.step
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                logits, loss, aux_loss = self.model(x, y)
                
                total_loss += loss.item()
                total_aux_loss += aux_loss.item()
                num_batches += 1
        
        return {
            'eval_loss': total_loss / num_batches,
            'eval_aux_loss': total_aux_loss / num_batches,
            'eval_total_loss': (total_loss + total_aux_loss) / num_batches
        }
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader = None):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Model has {self.model.get_num_params():,} parameters")
        
        max_steps = self.config.get('max_steps', 100000)
        eval_interval = self.config.get('eval_interval', 1000)
        log_interval = self.config.get('log_interval', 100)
        
        start_time = time.time()
        
        while self.step < max_steps:
            for batch in train_dataloader:
                if self.step >= max_steps:
                    break
                
                # Training step
                metrics = self.train_step(batch)
                
                # Logging
                if self.step % log_interval == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"Step {self.step:6d} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Aux: {metrics['aux_loss']:.4f} | "
                        f"LR: {metrics['lr']:.2e} | "
                        f"Time: {elapsed:.1f}s"
                    )
                
                # Evaluation
                if val_dataloader and self.step % eval_interval == 0:
                    eval_metrics = self.evaluate(val_dataloader)
                    print(
                        f"Eval {self.step:6d} | "
                        f"Loss: {eval_metrics['eval_loss']:.4f} | "
                        f"Aux: {eval_metrics['eval_aux_loss']:.4f}"
                    )
                
                # Save checkpoint
                if self.step % self.config.get('save_interval', 5000) == 0:
                    self.save_checkpoint()
        
        print("Training completed!")
    
    def save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'config': self.model.config
        }
        
        torch.save(checkpoint, f'checkpoint_step_{self.step}.pt')
        print(f"Saved checkpoint at step {self.step}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        
        print(f"Loaded checkpoint from step {self.step}")

def main():
    """Main training function."""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load configuration
    config = get_config('gpt-oss-small')
    print(f"\nUsing configuration: {config.name}")
    
    # Prepare data
    print("\nPreparing data...")
    train_loader, val_loader, tokenizer = prepare_shakespeare_data(
        block_size=config.block_size,
        batch_size=8  # Small batch size for demo
    )
    
    # Update config with correct vocab size
    config.vocab_size = tokenizer.vocab_size
    
    # Create model
    model = GPTOSS(config)
    
    # Print model summary
    print_model_summary(model, config)
    
    # Create trainer
    trainer = Trainer(model, config, device, tokenizer)
    
    # Training parameters
    training_args = {
        'num_epochs': 3,  # Small number for demo
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'gradient_clip_val': 1.0,
        'save_dir': 'checkpoints',
        'save_every': 1,
        'eval_every': 1,
        'log_every': 50
    }
    
    print(f"\nTraining arguments: {training_args}")
    
    # Start training
    trainer.train(train_loader, val_loader, **training_args)
    
    # Generate sample text
    print("\nGenerating sample text...")
    sample_text = trainer.generate_sample("ROMEO:", max_new_tokens=100)
    print(f"Generated text: {sample_text}")
    
    print("\nTraining demo completed!")

if __name__ == "__main__":
    main()