import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# NORMALIZATION COMPONENTS
# These components stabilize training and improve convergence
# ============================================================================

class LayerNorm(nn.Module):
    """LayerNorm - Standard Layer Normalization with Bias Control.
    
    WHAT IT DOES:
    - Normalizes inputs across the feature dimension
    - Reduces internal covariate shift during training
    - Stabilizes gradients and accelerates convergence
    
    HOW IT WORKS:
    - Computes mean and variance across the last dimension
    - Normalizes: (x - mean) / sqrt(variance + eps)
    - Applies learnable scale (weight) and shift (bias) parameters
    
    WHY IT'S IMPORTANT:
    - Essential for training deep networks (20+ layers)
    - Allows higher learning rates
    - Reduces sensitivity to initialization
    - Standard in transformer architectures
    
    MATHEMATICAL FORMULA:
    LayerNorm(x) = γ * (x - μ) / σ + β
    where μ = mean(x), σ = std(x), γ = weight, β = bias
    """
    
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """RMSNorm - Root Mean Square Normalization (More Efficient Alternative).
    
    WHAT IT DOES:
    - Normalizes inputs using only the root mean square (no mean subtraction)
    - Provides similar benefits to LayerNorm with less computation
    - Used in modern LLMs like LLaMA, PaLM, and others
    
    HOW IT DIFFERS FROM LAYERNORM:
    - LayerNorm: (x - mean) / std  →  requires mean and variance computation
    - RMSNorm: x / rms  →  only requires RMS computation
    - ~15% faster than LayerNorm
    - Often performs as well or better than LayerNorm
    
    WHY IT'S BETTER:
    - Faster computation (no mean subtraction)
    - Better numerical stability
    - Simpler implementation
    - Proven effective in large language models
    
    MATHEMATICAL FORMULA:
    RMSNorm(x) = γ * x / sqrt(mean(x²) + ε)
    where γ = weight (learnable scale parameter)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization with learnable scale."""
        output = self._norm(x.float()).type_as(x)
        return output * self.weight