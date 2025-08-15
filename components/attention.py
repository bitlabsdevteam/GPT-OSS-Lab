import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ============================================================================
# ATTENTION COMPONENTS
# These components handle the core attention mechanism and positional encoding
# ============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """RoPE - Rotary Positional Embedding (Advanced Positional Encoding).
    
    WHAT IT DOES:
    - Encodes positional information directly into query and key vectors
    - Enables the model to understand token positions and relationships
    - Provides better length extrapolation than absolute positional embeddings
    
    HOW IT WORKS:
    - Applies rotation matrices to query and key vectors
    - Each position gets a unique rotation angle
    - Rotation angles decrease with higher dimensions
    - Preserves relative positional relationships
    
    WHY IT'S REVOLUTIONARY:
    - Better than absolute positional embeddings
    - Enables length extrapolation (trained on 2K, works on 4K+)
    - Maintains translation invariance
    - Used in GPT-NeoX, LLaMA, PaLM, and other SOTA models
    
    MATHEMATICAL FOUNDATION:
    - Uses complex exponentials: e^(i*m*θ) where m=position, θ=angle
    - Rotation matrix: [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    - Applied to pairs of dimensions in query/key vectors
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute rotation angles for each dimension pair
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache cos and sin values for efficiency
        self._set_cos_sin_cache(max_position_embeddings)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Pre-compute cos and sin values for efficiency."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin values for the given sequence length."""
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention - The Core of Transformer Architecture.
    
    WHAT IT DOES:
    - Allows the model to attend to different parts of the input simultaneously
    - Captures various types of relationships (syntactic, semantic, positional)
    - Enables parallel processing of different attention patterns
    
    HOW IT WORKS:
    1. Linear projections create Query, Key, Value matrices
    2. Split into multiple heads for parallel attention computation
    3. Apply scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d)V
    4. Apply causal mask for autoregressive generation
    5. Concatenate heads and apply output projection
    
    WHY MULTIPLE HEADS:
    - Each head can focus on different types of relationships
    - Head 1: syntactic relationships (subject-verb)
    - Head 2: semantic relationships (word meanings)
    - Head 3: positional relationships (word order)
    - Provides richer representation than single attention
    
    ARCHITECTURAL INNOVATIONS:
    - Uses RoPE for better positional understanding
    - Supports Flash Attention for memory efficiency
    - Causal masking for autoregressive generation
    - Dropout for regularization
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        # Query, Key, Value projections for all heads (computed in parallel)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Rotary Positional Embedding
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, 
            max_position_embeddings=config.block_size
        )
        
        # Causal mask for autoregressive generation
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, n_embd]
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape for multi-head attention: (B, T, C) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply rotary positional embedding
        cos, sin = self.rotary_emb(v, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Causal self-attention with Flash Attention optimization
        if hasattr(F, 'scaled_dot_product_attention'):
            # Use Flash Attention if available (PyTorch 2.0+)
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.config.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            # Manual attention computation (fallback)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y