import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
import json
import os
from pathlib import Path

# ============================================================================
# UTILITY FUNCTIONS
# Helper functions for model training, evaluation, and management
# ============================================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                         epoch: int, loss: float, filepath: str,
                         config: Optional[Any] = None) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        config: Model configuration
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_model_checkpoint(filepath: str, model: nn.Module, 
                         optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Loss: {checkpoint.get('loss', 'Unknown')}")
    
    return checkpoint


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon
    else:
        return torch.device('cpu')


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def calculate_flops_per_token(config) -> int:
    """Calculate approximate FLOPs per token for the model.
    
    Args:
        config: Model configuration
        
    Returns:
        Approximate FLOPs per token
    """
    # Simplified FLOP calculation for transformer
    # This is an approximation based on the forward pass
    
    n_params = (
        config.vocab_size * config.n_embd +  # Token embeddings
        config.n_layer * (
            4 * config.n_embd * config.n_embd +  # Attention projections
            2 * config.n_embd * 4 * config.n_embd  # FFN
        ) +
        config.n_embd * config.vocab_size  # Output projection
    )
    
    # Approximate FLOPs per token (forward pass only)
    flops_per_token = 2 * n_params  # 2 FLOPs per parameter (multiply-add)
    
    return flops_per_token


def estimate_memory_usage(config, batch_size: int = 1, seq_len: Optional[int] = None) -> Dict[str, float]:
    """Estimate memory usage for the model.
    
    Args:
        config: Model configuration
        batch_size: Batch size
        seq_len: Sequence length (defaults to config.block_size)
        
    Returns:
        Dictionary with memory estimates in MB
    """
    if seq_len is None:
        seq_len = config.block_size
    
    # Parameter memory (assuming float32)
    param_count = count_parameters_from_config(config)
    param_memory = param_count * 4 / (1024 * 1024)  # 4 bytes per float32, convert to MB
    
    # Activation memory (rough estimate)
    activation_memory = (
        batch_size * seq_len * config.n_embd * config.n_layer * 8  # Rough estimate
    ) / (1024 * 1024)
    
    # Gradient memory (same as parameters for training)
    gradient_memory = param_memory
    
    # Optimizer state (Adam uses 2x parameter memory)
    optimizer_memory = param_memory * 2
    
    return {
        'parameters': param_memory,
        'activations': activation_memory,
        'gradients': gradient_memory,
        'optimizer': optimizer_memory,
        'total_training': param_memory + activation_memory + gradient_memory + optimizer_memory,
        'total_inference': param_memory + activation_memory
    }


def count_parameters_from_config(config) -> int:
    """Estimate parameter count from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Estimated parameter count
    """
    # Token embeddings
    token_emb = config.vocab_size * config.n_embd
    
    # Transformer layers
    per_layer = (
        # Attention
        3 * config.n_embd * config.n_embd +  # QKV projections
        config.n_embd * config.n_embd +      # Output projection
        # FFN
        config.n_embd * (4 * config.n_embd) +  # Up projection
        (4 * config.n_embd) * config.n_embd +  # Down projection
        # Layer norms
        2 * config.n_embd
    )
    
    # MoE adjustment
    if hasattr(config, 'use_moe') and config.use_moe:
        # Multiply FFN parameters by number of experts
        ffn_params = config.n_embd * (4 * config.n_embd) + (4 * config.n_embd) * config.n_embd
        per_layer = per_layer - ffn_params + (ffn_params * config.num_experts)
        # Add router parameters
        per_layer += config.n_embd * config.num_experts
    
    transformer_params = config.n_layer * per_layer
    
    # Output layer
    output_layer = config.n_embd * config.vocab_size
    
    # Final layer norm
    final_ln = config.n_embd
    
    total = token_emb + transformer_params + output_layer + final_ln
    
    return total


def print_model_summary(model: nn.Module, config: Any) -> None:
    """Print a comprehensive model summary.
    
    Args:
        model: PyTorch model
        config: Model configuration
    """
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    
    # Parameter counts
    param_counts = count_parameters(model)
    print(f"Total Parameters: {param_counts['total']:,}")
    print(f"Trainable Parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable Parameters: {param_counts['non_trainable']:,}")
    
    # Model size
    size_mb = get_model_size_mb(model)
    print(f"Model Size: {size_mb:.2f} MB")
    
    # Architecture details
    print(f"\nArchitecture:")
    print(f"- Layers: {config.n_layer}")
    print(f"- Hidden Size: {config.n_embd}")
    print(f"- Attention Heads: {config.n_head}")
    print(f"- Vocabulary Size: {config.vocab_size:,}")
    print(f"- Context Length: {config.block_size}")
    
    if hasattr(config, 'use_moe') and config.use_moe:
        print(f"- MoE Experts: {config.num_experts}")
        print(f"- Active Experts per Token: {config.top_k_experts}")
        active_params = param_counts['total'] // config.num_experts * config.top_k_experts
        print(f"- Active Parameters per Token: ~{active_params:,}")
    
    # Memory estimates
    memory_est = estimate_memory_usage(config)
    print(f"\nMemory Estimates (batch_size=1):")
    print(f"- Parameters: {memory_est['parameters']:.1f} MB")
    print(f"- Activations: {memory_est['activations']:.1f} MB")
    print(f"- Total Inference: {memory_est['total_inference']:.1f} MB")
    print(f"- Total Training: {memory_est['total_training']:.1f} MB")
    
    # Performance estimates
    flops = calculate_flops_per_token(config)
    print(f"\nPerformance Estimates:")
    print(f"- FLOPs per Token: ~{flops:,}")
    
    print("=" * 80)