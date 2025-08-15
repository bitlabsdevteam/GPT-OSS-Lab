import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# FEEDFORWARD COMPONENTS
# These components handle the feedforward processing within transformer blocks
# ============================================================================

class SwiGLU(nn.Module):
    """SwiGLU - Swish-Gated Linear Unit (Advanced Activation Function).
    
    WHAT IT DOES:
    - Combines Swish activation with gating mechanism
    - Provides better gradient flow than ReLU
    - Used in modern LLMs like PaLM, LLaMA, and GLM
    
    HOW IT WORKS:
    - Split input into two parts: gate and value
    - Apply Swish (SiLU) activation to gate: swish(x) = x * sigmoid(x)
    - Element-wise multiply: gate * value
    - More expressive than standard ReLU-based FFN
    
    WHY IT'S BETTER THAN RELU:
    - ReLU: max(0, x) → hard cutoff, gradient = 0 for x < 0
    - Swish: x * sigmoid(x) → smooth, non-zero gradients everywhere
    - Gating: allows selective information flow
    - Better performance on language modeling tasks
    
    MATHEMATICAL FORMULA:
    SwiGLU(x) = Swish(xW₁ + b₁) ⊙ (xW₂ + b₂)
    where Swish(x) = x * σ(x), σ = sigmoid, ⊙ = element-wise multiply
    """
    
    def __init__(self, dim: int):
        super().__init__()
        # Note: We use dim * 2 because we split into gate and value
        self.w1 = nn.Linear(dim, dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(dim, dim, bias=False)  # Value projection
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.
        
        Args:
            x: Input tensor of shape [..., dim]
            
        Returns:
            Output tensor of shape [..., dim]
        """
        # Split into gate and value components
        gate = self.w1(x)  # Gate path
        value = self.w2(x)  # Value path
        
        # Apply Swish activation to gate and multiply with value
        return F.silu(gate) * value  # SiLU is the same as Swish


class FeedForward(nn.Module):
    """FeedForward Network - Position-wise Processing in Transformers.
    
    WHAT IT DOES:
    - Processes each position independently (no cross-position interaction)
    - Provides non-linear transformation after attention
    - Increases model capacity and expressiveness
    
    HOW IT WORKS:
    1. Expand: Linear projection to higher dimension (usually 4x)
    2. Activate: Apply non-linear activation (SwiGLU)
    3. Contract: Linear projection back to original dimension
    4. This creates a "bottleneck" that forces efficient representation
    
    WHY THIS ARCHITECTURE:
    - Attention handles cross-position relationships
    - FFN handles position-wise transformations
    - Expansion allows complex non-linear mappings
    - Contraction maintains consistent dimensionality
    
    ARCHITECTURAL CHOICES:
    - Uses SwiGLU instead of ReLU for better performance
    - 4x expansion ratio (standard in most transformers)
    - Dropout for regularization
    - No bias terms (following modern practices)
    
    INFORMATION FLOW:
    Input [B, T, C] → Expand [B, T, 4C] → Activate → Contract [B, T, C]
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = 4 * config.n_embd  # Standard 4x expansion
        
        # Two-layer MLP with SwiGLU activation
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.activation = SwiGLU(hidden_dim // 2)  # SwiGLU needs half the dimension
        self.c_proj = nn.Linear(hidden_dim // 2, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feedforward network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, n_embd]
        """
        # Expand to higher dimension
        x = self.c_fc(x)
        
        # Split for SwiGLU (gate and value paths)
        gate, value = x.chunk(2, dim=-1)
        
        # Apply SwiGLU activation
        x = F.silu(gate) * value
        
        # Contract back to original dimension
        x = self.c_proj(x)
        x = self.dropout(x)
        
        return x


class SimpleFeedForward(nn.Module):
    """Simple FeedForward Network - Traditional ReLU-based FFN.
    
    This is a simpler alternative to the SwiGLU-based feedforward.
    Useful for comparison or when computational resources are limited.
    """
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x