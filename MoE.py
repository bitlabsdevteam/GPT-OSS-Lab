import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# ============================================================================
# GPT-OSS: Complete LLM Implementation from Tokenized Text to Linear Output
# Building a Language Model from Scratch - Bottom-Up Architecture
# 
# This implementation follows the complete pipeline:
# 1. Tokenized Text Input → Token Embeddings
# 2. Positional Encoding → Position-aware Representations  
# 3. Multi-Head Attention → Context Understanding
# 4. Feed-Forward Networks / MoE → Feature Processing
# 5. Layer Normalization → Training Stability
# 6. Residual Connections → Gradient Flow
# 7. Final Linear Layer → Vocabulary Predictions
# ============================================================================

# ============================================================================
# COMPONENT 1: NORMALIZATION LAYERS
# These components stabilize training and improve convergence
# ============================================================================

class LayerNorm(nn.Module):
    """Standard Layer Normalization.
    
    WHAT IT DOES:
    - Normalizes activations across the feature dimension
    - Stabilizes training by reducing internal covariate shift
    - Applied before attention and feed-forward layers (pre-norm)
    
    HOW IT WORKS:
    - Computes mean and variance across the last dimension
    - Normalizes: (x - mean) / sqrt(variance + eps)
    - Applies learnable scale (weight) and shift (bias) parameters
    
    WHY WE NEED IT:
    - Prevents gradient explosion/vanishing in deep networks
    - Allows higher learning rates and faster convergence
    - Essential for training stability in large language models
    """
    
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - Modern Alternative to LayerNorm.
    
    WHAT IT DOES:
    - Normalizes using only the root mean square (no mean centering)
    - More computationally efficient than standard LayerNorm
    - Used in modern LLMs like LLaMA, PaLM for better performance
    
    HOW IT WORKS:
    - Computes RMS: sqrt(mean(x²))
    - Normalizes: x / (RMS + eps)
    - Applies learnable scale parameter (no bias)
    
    WHY IT'S BETTER:
    - Simpler computation (no mean subtraction)
    - Often provides better training dynamics
    - Reduces memory usage and computation time
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)

# ============================================================================
# COMPONENT 2: POSITIONAL ENCODING
# These components inject position information into token representations
# ============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - Advanced Positional Encoding.
    
    WHAT IT DOES:
    - Encodes position information by rotating query/key vectors
    - Provides relative position awareness without absolute position embeddings
    - Enables better length extrapolation beyond training sequence lengths
    
    HOW IT WORKS:
    - Creates rotation matrices based on position and frequency
    - Applies rotation to query and key vectors in attention
    - Each dimension pair gets rotated by position-dependent angles
    
    WHY IT'S SUPERIOR:
    - Better extrapolation to longer sequences than learned positions
    - Naturally encodes relative distances between tokens
    - No additional parameters needed (computed on-the-fly)
    - Used in GPT-NeoX, LLaMA, and other modern LLMs
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # Compute the angles
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        return cos, sin

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors."""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ============================================================================
# COMPONENT 3: ATTENTION MECHANISM
# The core component that allows tokens to attend to each other
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention - The Heart of Transformer Architecture.
    
    WHAT IT DOES:
    - Allows each token to attend to all other tokens in the sequence
    - Captures relationships and dependencies between words
    - Processes information in parallel across multiple attention heads
    
    HOW IT WORKS:
    1. Linear projections create Query (Q), Key (K), Value (V) matrices
    2. Each head computes attention: Attention(Q,K,V) = softmax(QK^T/√d)V
    3. Multiple heads capture different types of relationships
    4. Outputs are concatenated and projected back to model dimension
    
    WHY MULTIPLE HEADS:
    - Different heads can focus on different aspects (syntax, semantics, etc.)
    - Increases model capacity without increasing sequence computation
    - Allows parallel processing of different relationship types
    
    ADVANCED FEATURES:
    - RoPE: Better positional encoding for long sequences
    - Flash Attention: Memory-efficient attention computation
    - Causal masking: Prevents looking at future tokens (for autoregressive LM)
    """
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Rotary positional embedding
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, config.block_size)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Flash Attention not available, using standard attention")
            # Causal mask for standard attention
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch_size, seq_len, embedding_dim
        
        # Compute Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply rotary positional embedding
        cos, sin = self.rotary_emb(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention computation
        if self.flash:
            # Use Flash Attention for efficiency
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            # Standard attention implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        
        # Reshape and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        
        return y

# ============================================================================
# COMPONENT 4: FEED-FORWARD NETWORKS
# These components process and transform token representations
# ============================================================================

class SwiGLU(nn.Module):
    """SwiGLU Activation - Advanced Feed-Forward Component.
    
    WHAT IT DOES:
    - Applies gated linear transformations with Swish activation
    - Processes each token's representation independently
    - Increases model capacity and non-linearity
    
    HOW IT WORKS:
    1. Split input into two paths: gate and value
    2. Gate path: apply Swish activation (x * sigmoid(x))
    3. Value path: linear transformation
    4. Element-wise multiply gate and value
    5. Project back to original dimension
    
    WHY SwiGLU vs GELU:
    - Better performance in large language models
    - Gating mechanism provides more selective activation
    - Used in PaLM, LLaMA, and other state-of-the-art models
    
    FORMULA: SwiGLU(x) = Swish(xW₁) ⊙ (xW₃) W₂
    """
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(2 * config.n_embd * 4 / 3)  # Standard scaling for SwiGLU
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)   # Down projection
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)   # Up projection
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: swish(x @ w1) * (x @ w3) @ w2
        gate = F.silu(self.w1(x))  # Swish activation
        up = self.w3(x)
        return self.dropout(self.w2(gate * up))

class FeedForward(nn.Module):
    """Standard Feed-Forward Network - Classic MLP Component.
    
    WHAT IT DOES:
    - Applies position-wise transformations to token representations
    - Increases model capacity through non-linear transformations
    - Processes each position independently (no cross-token interaction)
    
    HOW IT WORKS:
    1. Expand dimension by 4x (standard transformer scaling)
    2. Apply GELU activation for non-linearity
    3. Project back to original dimension
    4. Apply dropout for regularization
    
    WHY WE NEED IT:
    - Attention captures relationships, FFN processes individual tokens
    - Provides computational capacity for complex transformations
    - The 4x expansion gives model room to learn complex patterns
    
    ARCHITECTURE: Linear(d→4d) → GELU → Linear(4d→d) → Dropout
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

# ============================================================================
# COMPONENT 5: MIXTURE OF EXPERTS (MoE) ROUTING
# Advanced scaling technique that increases model capacity efficiently
# ============================================================================

class Router(nn.Module):
    """Top-K Router - The Brain of Mixture of Experts.
    
    WHAT IT DOES:
    - Decides which expert networks should process each token
    - Enables sparse computation: only activate relevant experts
    - Scales model capacity without proportional compute increase
    
    HOW IT WORKS:
    1. Compute routing logits for each token → expert assignment
    2. Apply softmax to get expert probabilities
    3. Select top-k experts with highest probabilities
    4. Normalize probabilities for selected experts
    5. Route tokens to chosen experts with computed weights
    
    WHY MoE ROUTING:
    - Allows massive model scaling (100B+ parameters)
    - Only activates ~1-10% of parameters per token
    - Different experts can specialize in different domains/patterns
    - Maintains constant computational cost regardless of expert count
    
    LOAD BALANCING:
    - Auxiliary loss encourages uniform expert utilization
    - Prevents expert collapse (all tokens going to few experts)
    - Essential for training stability and efficiency
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Router configuration
        self.top_k = config.top_k
        self.n_experts = config.n_experts
        self.use_aux_loss = config.use_aux_loss
        self.aux_loss_weight = config.aux_loss_weight
        
        # Router network - maps input to expert logits
        self.gate = nn.Linear(config.n_embd, config.n_experts, bias=False)
        
        # Capacity factor for load balancing
        self.capacity_factor = config.capacity_factor
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # Flatten to (B*T, D)
        
        # Compute router logits
        logits = self.gate(x_flat)  # (B*T, n_experts)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute auxiliary loss for load balancing
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.use_aux_loss and self.training:
            # Encourage uniform distribution across experts
            mean_probs = probs.mean(dim=0)
            aux_loss = (mean_probs * torch.log(mean_probs + 1e-8)).sum()
            aux_loss *= self.aux_loss_weight
        
        return top_k_probs, top_k_indices, aux_loss

class Expert(nn.Module):
    """Individual Expert Network - Specialized Processing Unit.
    
    WHAT IT DOES:
    - Specialized feed-forward network that becomes expert in specific patterns
    - Processes tokens routed to it by the Router
    - Part of the larger MoE system for efficient scaling
    
    HOW IT WORKS:
    - Same architecture as standard FFN (SwiGLU or standard)
    - Learns to specialize during training through routing decisions
    - Only activated when router selects it for specific tokens
    
    WHY SPECIALIZATION EMERGES:
    - Different experts see different subsets of data during training
    - Gradient updates naturally lead to specialization
    - Examples: syntax expert, math expert, code expert, etc.
    """
    
    def __init__(self, config):
        super().__init__()
        if config.use_swiglu:
            self.ffn = SwiGLU(config)
        else:
            self.ffn = FeedForward(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

class MixtureOfExperts(nn.Module):
    """Mixture of Experts Layer - Efficient Sparse Processing.
    
    WHAT IT DOES:
    - Coordinates routing and processing across multiple expert networks
    - Implements sparse activation: only some experts process each token
    - Combines expert outputs with router-determined weights
    
    HOW THE COMPLETE MoE PROCESS WORKS:
    1. Router analyzes each token and computes expert probabilities
    2. Top-k experts are selected for each token
    3. Tokens are batched and sent to their assigned experts
    4. Each expert processes its assigned tokens
    5. Expert outputs are weighted by router probabilities
    6. Final output combines all expert contributions per token
    
    EFFICIENCY BENEFITS:
    - Constant compute cost regardless of number of experts
    - Massive parameter scaling without proportional compute increase
    - Better performance than dense models with same compute budget
    
    IMPLEMENTATION DETAILS:
    - Batched processing for efficiency
    - Load balancing through auxiliary losses
    - Gradient routing for end-to-end training
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.n_experts = config.n_experts
        self.top_k = config.top_k
        
        # Router for expert selection
        self.router = Router(config)
        
        # Expert networks
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)
        
        # Route tokens to experts
        top_k_probs, top_k_indices, aux_loss = self.router(x)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process tokens through selected experts
        for i in range(self.top_k):
            expert_probs = top_k_probs[:, i]  # (B*T,)
            expert_indices = top_k_indices[:, i]  # (B*T,)
            
            # Group tokens by expert
            for expert_id in range(self.n_experts):
                # Find tokens assigned to this expert
                mask = (expert_indices == expert_id)
                if mask.any():
                    # Get tokens for this expert
                    expert_input = x_flat[mask]
                    
                    # Process through expert
                    expert_output = self.experts[expert_id](expert_input)
                    
                    # Weight by router probability and add to output
                    expert_probs_masked = expert_probs[mask].unsqueeze(-1)
                    output[mask] += expert_probs_masked * expert_output
        
        return output.view(B, T, D), aux_loss

# ============================================================================
# COMPONENT 6: TRANSFORMER BLOCK
# The fundamental building block that combines all components
# ============================================================================

class TransformerBlock(nn.Module):
    """Transformer Block - The Complete Processing Unit.
    
    WHAT IT DOES:
    - Combines attention and feed-forward processing in one block
    - Implements the core transformer architecture pattern
    - Can use either standard FFN or MoE for scaling
    
    HOW INFORMATION FLOWS:
    1. Input token representations enter the block
    2. Pre-normalization prepares inputs for attention
    3. Multi-head attention captures token relationships
    4. Residual connection preserves original information
    5. Pre-normalization prepares for feed-forward processing
    6. FFN/MoE processes individual token representations
    7. Another residual connection maintains gradient flow
    8. Output representations are richer and more contextual
    
    WHY THIS ARCHITECTURE:
    - Pre-norm: Better gradient flow and training stability
    - Residual connections: Enable very deep networks (100+ layers)
    - Attention + FFN: Captures both relationships and individual processing
    - Modular design: Easy to stack many blocks for deeper models
    
    RESIDUAL CONNECTIONS IMPORTANCE:
    - Solve vanishing gradient problem in deep networks
    - Allow information to flow directly from input to output
    - Enable training of very deep models (GPT-3 has 96 layers)
    """
    
    def __init__(self, config, use_moe: bool = False):
        super().__init__()
        
        # Layer normalization (pre-norm architecture)
        if config.use_rms_norm:
            self.ln_1 = RMSNorm(config.n_embd)
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        
        # Multi-head attention
        self.attn = MultiHeadAttention(config)
        
        # Feed-forward or MoE layer
        if use_moe:
            self.mlp = MixtureOfExperts(config)
            self.use_moe = True
        else:
            if config.use_swiglu:
                self.mlp = SwiGLU(config)
            else:
                self.mlp = FeedForward(config)
            self.use_moe = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # Feed-forward with residual connection
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.use_moe:
            mlp_out, aux_loss = self.mlp(self.ln_2(x))
            x = x + mlp_out
        else:
            x = x + self.mlp(self.ln_2(x))
        
        return x, aux_loss

@dataclass
class GPTOSSConfig:
    """Configuration class for GPT-OSS model.
    
    Contains all hyperparameters needed to configure the model architecture,
    including MoE-specific parameters and training settings.
    """
    
    # Model architecture
    vocab_size: int = 200000  # Large vocabulary as shown in diagram
    block_size: int = 131072  # 131k context length as shown
    n_layer: int = 64         # Number of transformer layers
    n_head: int = 64          # Number of attention heads
    n_embd: int = 2880        # Embedding dimension
    
    # MoE configuration
    n_experts: int = 8        # Number of experts per MoE layer
    top_k: int = 2           # Number of experts to route to
    moe_layers: list = None  # Which layers use MoE (None = every other layer)
    capacity_factor: float = 1.25  # Expert capacity factor
    
    # Training configuration
    dropout: float = 0.0
    bias: bool = False       # No bias for better efficiency
    
    # Advanced features
    use_rms_norm: bool = True    # Use RMSNorm instead of LayerNorm
    use_swiglu: bool = True      # Use SwiGLU activation
    
    # Auxiliary loss configuration
    use_aux_loss: bool = True
    aux_loss_weight: float = 0.01
    
    def __post_init__(self):
        # Default MoE layer configuration (every other layer)
        if self.moe_layers is None:
            self.moe_layers = list(range(1, self.n_layer, 2))  # Odd layers use MoE

# ============================================================================
# COMPONENT 7: COMPLETE LLM ARCHITECTURE
# The main model that orchestrates all components from input to output
# ============================================================================

class GPTOSS(nn.Module):
    """GPT-OSS: Complete Language Model from Tokenized Text to Linear Output.
    
    THE COMPLETE PIPELINE - BOTTOM TO TOP:
    
    1. TOKENIZED TEXT INPUT:
       - Raw text → tokenizer → integer token IDs
       - Shape: [batch_size, sequence_length]
    
    2. TOKEN EMBEDDINGS:
       - Token IDs → learned vector representations
       - Each token gets a dense vector (e.g., 4096 dimensions)
       - Shape: [batch_size, sequence_length, embedding_dim]
    
    3. POSITIONAL ENCODING:
       - Add position information to token embeddings
       - RoPE: rotary embeddings applied in attention layers
       - Enables model to understand token order and relationships
    
    4. TRANSFORMER LAYERS (REPEATED N TIMES):
       a) Layer Normalization → stabilizes training
       b) Multi-Head Attention → captures token relationships
       c) Residual Connection → preserves information flow
       d) Layer Normalization → prepares for next component
       e) Feed-Forward/MoE → processes individual tokens
       f) Residual Connection → maintains gradient flow
    
    5. FINAL LAYER NORMALIZATION:
       - Normalizes final hidden states before output projection
    
    6. LINEAR OUTPUT LAYER:
       - Projects hidden states to vocabulary size
       - Shape: [batch_size, sequence_length, vocab_size]
       - Each position gets probability distribution over all tokens
    
    7. LOSS COMPUTATION (TRAINING):
       - Cross-entropy loss between predictions and targets
       - Next token prediction objective
    
    WHY THIS ARCHITECTURE WORKS:
    - Attention captures long-range dependencies
    - Feed-forward adds computational capacity
    - Residual connections enable deep networks
    - Layer normalization stabilizes training
    - MoE scales capacity efficiently
    - Autoregressive training teaches language patterns
    """
    
    def __init__(self, config: GPTOSSConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        # Token and position embeddings
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            drop=nn.Dropout(config.dropout),                     # Embedding dropout
            h=nn.ModuleList([                                    # Transformer blocks
                TransformerBlock(config, use_moe=(i in config.moe_layers))
                for i in range(config.n_layer)
            ]),
            ln_f=RMSNorm(config.n_embd) if config.use_rms_norm else LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (share embeddings with output layer)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module):
        """Initialize model weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Forward Pass: Complete Data Flow from Tokenized Text to Linear Output.
        
        STEP-BY-STEP EXECUTION:
        
        INPUT: idx = tokenized text [batch_size, sequence_length]
        Example: [[15496, 995, 318, 257, 6291]]  # "Hello world is a test"
        
        Args:
            idx: Input token indices of shape (batch_size, sequence_length)
            targets: Target token indices for loss computation (optional)
            
        Returns:
            logits: Output logits of shape (batch_size, sequence_length, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
            aux_loss: Auxiliary loss from MoE layers for load balancing
        """
        device = idx.device
        b, t = idx.size()  # batch_size, sequence_length
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # STEP 1: TOKEN EMBEDDINGS
        # Convert token IDs to dense vector representations
        # Shape: [batch_size, seq_len] → [batch_size, seq_len, embedding_dim]
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        x = self.transformer.drop(tok_emb)  # Apply dropout for regularization
        
        # STEP 2: POSITIONAL INFORMATION
        # RoPE is applied inside attention layers, not here
        # This allows the model to understand token positions and relationships
        
        # STEP 3: TRANSFORMER LAYERS PROCESSING
        # Each layer refines the token representations
        total_aux_loss = torch.tensor(0.0, device=device)
        for block in self.transformer.h:
            x, aux_loss = block(x)
            total_aux_loss += aux_loss
            # After each layer, x has shape [batch_size, seq_len, embedding_dim]
            # but contains increasingly sophisticated representations
        
        # STEP 4: FINAL NORMALIZATION
        # Stabilize the final hidden states before output projection
        x = self.transformer.ln_f(x)  # Shape: [batch_size, seq_len, embedding_dim]
        
        # STEP 5: LINEAR OUTPUT PROJECTION
        # Convert hidden states to vocabulary probabilities
        if targets is not None:
            # TRAINING MODE: Compute loss for all positions
            logits = self.lm_head(x)  # Shape: [batch_size, seq_len, vocab_size]
            
            # STEP 6: LOSS COMPUTATION
            # Compare predictions with ground truth for next token prediction
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # INFERENCE MODE: Only compute logits for the last token (efficiency)
            logits = self.lm_head(x[:, [-1], :])  # Shape: [batch_size, 1, vocab_size]
            # This gives probability distribution over next possible tokens
            loss = None
        
        return logits, loss, total_aux_loss
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, exclude embedding parameters from count
            
        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """Generate new tokens using the model.
        
        Args:
            idx: Input token indices of shape (batch_size, sequence_length)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top-k most likely tokens
            
        Returns:
            Generated token indices including input
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (b, vocab_size)
            
            # Optionally crop to top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# ============================================================================
# Model Configurations for Different Sizes
# ============================================================================

def get_gpt_oss_20b_config() -> GPTOSSConfig:
    """Configuration for GPT-OSS 20B model (as shown in diagram)."""
    return GPTOSSConfig(
        vocab_size=200000,
        block_size=131072,
        n_layer=32,
        n_head=32,
        n_embd=2048,
        n_experts=8,
        top_k=2,
        dropout=0.0,
        bias=False,
        use_rms_norm=True,
        use_swiglu=True,
        use_aux_loss=True,
        aux_loss_weight=0.01,
    )

def get_gpt_oss_120b_config() -> GPTOSSConfig:
    """Configuration for GPT-OSS 120B model (as shown in diagram)."""
    return GPTOSSConfig(
        vocab_size=200000,
        block_size=131072,
        n_layer=64,
        n_head=64,
        n_embd=4096,
        n_experts=16,
        top_k=2,
        dropout=0.0,
        bias=False,
        use_rms_norm=True,
        use_swiglu=True,
        use_aux_loss=True,
        aux_loss_weight=0.01,
    )

# ============================================================================
# Example Usage
# ============================================================================

# ============================================================================
# COMPLETE ARCHITECTURE SUMMARY: FROM TOKENIZED TEXT TO LINEAR OUTPUT
# ============================================================================

"""
COMPLETE LLM PIPELINE - BOTTOM-UP ARCHITECTURE:

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. TOKENIZED TEXT INPUT                                                     │
│    Raw text: "Hello world" → Tokenizer → [15496, 995]                     │
│    Shape: [batch_size, sequence_length]                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. TOKEN EMBEDDINGS (wte)                                                  │
│    Token IDs → Dense vectors: [15496] → [0.1, -0.3, 0.8, ...]            │
│    Shape: [batch_size, seq_len, embedding_dim]                             │
│    Purpose: Convert discrete tokens to continuous representations           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. TRANSFORMER LAYERS (Repeated N times)                                   │
│    Each layer contains:                                                     │
│    ┌─────────────────────────────────────────────────────────────────────┐ │
│    │ a) RMSNorm → Stabilizes inputs                                      │ │
│    │ b) MultiHeadAttention + RoPE → Captures relationships              │ │
│    │ c) Residual Connection → Preserves information                      │ │
│    │ d) RMSNorm → Prepares for next component                           │ │
│    │ e) FeedForward/MoE → Processes individual tokens                   │ │
│    │ f) Residual Connection → Maintains gradient flow                   │ │
│    └─────────────────────────────────────────────────────────────────────┘ │
│    Progressive refinement: Each layer adds more sophisticated understanding │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. FINAL LAYER NORMALIZATION                                               │
│    Stabilizes final hidden states before output projection                 │
│    Shape: [batch_size, seq_len, embedding_dim]                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. LINEAR OUTPUT LAYER (lm_head)                                           │
│    Projects hidden states to vocabulary probabilities                      │
│    Shape: [batch_size, seq_len, vocab_size]                                │
│    Each position gets probability distribution over all possible tokens    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. LOSS COMPUTATION (Training) / SAMPLING (Inference)                      │
│    Training: Cross-entropy loss with next token targets                    │
│    Inference: Sample from probability distribution                          │
└─────────────────────────────────────────────────────────────────────────────┘

KEY ARCHITECTURAL INNOVATIONS:

1. MIXTURE OF EXPERTS (MoE):
   - Scales model capacity without proportional compute increase
   - Router selects top-k experts per token
   - Enables 120B+ parameter models with 20B compute cost

2. ROTARY POSITIONAL EMBEDDINGS (RoPE):
   - Better position encoding than absolute/learned embeddings
   - Enables length extrapolation beyond training sequences
   - Applied directly in attention computation

3. RMS NORMALIZATION:
   - More stable than LayerNorm
   - Faster computation
   - Better gradient flow in deep networks

4. SwiGLU ACTIVATION:
   - Gated activation function
   - Better performance than ReLU/GELU
   - Used in feed-forward networks

5. PRE-NORMALIZATION:
   - Apply normalization before attention/FFN
   - Better gradient flow than post-norm
   - Enables training of very deep models

MODEL SCALING:
- GPT-OSS 20B: 20B parameters, 32 layers, 2048 hidden size
- GPT-OSS 120B: 120B parameters, 64 layers, 4096 hidden size
- MoE enables massive scaling with constant compute per token

TRAINING OBJECTIVE:
- Next token prediction (autoregressive language modeling)
- Model learns to predict the next token given previous context
- This simple objective leads to emergent capabilities like reasoning, coding, etc.
"""

if __name__ == "__main__":
    print("=" * 80)
    print("GPT-OSS: Complete LLM Implementation from Tokenized Text to Linear Output")
    print("=" * 80)
    
    # Create a small model for testing
    config = GPTOSSConfig(
        vocab_size=50304,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        n_experts=4,
        top_k=2,
    )
    
    model = GPTOSS(config)
    
    # Print model info
    total_params = model.get_num_params()
    print(f"\nModel Configuration:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Layers: {config.n_layer}")
    print(f"- Hidden size: {config.n_embd}")
    print(f"- Attention heads: {config.n_head}")
    print(f"- Vocabulary size: {config.vocab_size:,}")
    print(f"- Context length: {config.block_size:,}")
    print(f"- MoE experts: {config.n_experts}")
    print(f"- Active experts per token: {config.top_k}")
    
    # Test forward pass
    print(f"\nForward Pass Demonstration:")
    x = torch.randint(0, config.vocab_size, (2, 64))  # (batch_size=2, seq_len=64)
    print(f"Input shape: {x.shape} (tokenized text)")
    
    logits, loss, aux_loss = model(x, x)  # Use x as both input and target for testing
    
    print(f"Output logits shape: {logits.shape} (vocabulary probabilities)")
    print(f"Loss: {loss.item():.4f}")
    print(f"Auxiliary loss: {aux_loss.item():.4f} (MoE load balancing)")
    
    # Test generation
    print(f"\nText Generation Demonstration:")
    model.eval()
    with torch.no_grad():
        prompt = x[:1, :10]
        print(f"Prompt shape: {prompt.shape}")
        
        generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
        print(f"Generated shape: {generated.shape}")
        print(f"Generated {generated.shape[1] - prompt.shape[1]} new tokens")
    
    print(f"\n" + "=" * 80)
    print("Complete LLM pipeline successfully demonstrated!")
    print("From tokenized text input to linear output layer with MoE scaling.")
    print("=" * 80)