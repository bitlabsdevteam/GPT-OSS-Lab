import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Import components from modular structure
from components.normalization import LayerNorm, RMSNorm
from components.attention import MultiHeadAttention
from components.feedforward import FeedForward
from components.moe import MixtureOfExperts
from config import GPTOSSConfig, get_config

# ============================================================================
# TRANSFORMER BLOCK
# The core building block that combines attention and feedforward processing
# ============================================================================


class TransformerBlock(nn.Module):
    """TransformerBlock - Complete Processing Unit of the Transformer.
    
    WHAT IT DOES:
    - Combines attention and feedforward processing
    - Implements the core transformer architecture
    - Processes tokens through self-attention and position-wise FFN
    
    HOW INFORMATION FLOWS:
    1. INPUT: Token representations [batch_size, seq_len, n_embd]
    2. ATTENTION: Tokens attend to each other (self-attention)
    3. ADD & NORM: Residual connection + normalization
    4. FEEDFORWARD: Position-wise processing (FFN or MoE)
    5. ADD & NORM: Another residual connection + normalization
    6. OUTPUT: Enhanced token representations [batch_size, seq_len, n_embd]
    
    WHY THIS ARCHITECTURE WORKS:
    - Attention handles relationships between tokens
    - FFN handles complex transformations within each position
    - Residual connections enable deep networks (gradient flow)
    - Layer normalization stabilizes training
    - Pre-normalization (norm before attention/FFN) improves training
    
    ARCHITECTURAL INNOVATIONS:
    - Pre-normalization instead of post-normalization
    - RMSNorm instead of LayerNorm (more efficient)
    - MoE support for sparse scaling
    - Modern activation functions (SwiGLU)
    """
    
    def __init__(self, config: GPTOSSConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Pre-attention normalization
        if config.use_rmsnorm:
            self.ln_1 = RMSNorm(config.n_embd)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        
        # Multi-head attention
        self.attn = MultiHeadAttention(config)
        
        # Pre-feedforward normalization
        if config.use_rmsnorm:
            self.ln_2 = RMSNorm(config.n_embd)
        else:
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        
        # Feedforward network (regular or MoE)
        if config.use_moe and (config.moe_layers is None or layer_idx in config.moe_layers):
            self.mlp = MixtureOfExperts(config)
            self.use_moe = True
        else:
            self.mlp = FeedForward(config)
            self.use_moe = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            
        Returns:
            output: Processed tensor [batch_size, seq_len, n_embd]
            aux_loss: MoE auxiliary loss (None for regular FFN)
        """
        aux_loss = None
        
        # Self-attention with residual connection (pre-norm)
        x = x + self.attn(self.ln_1(x))
        
        # Feedforward with residual connection (pre-norm)
        if self.use_moe:
            mlp_output, aux_loss = self.mlp(self.ln_2(x))
            x = x + mlp_output
        else:
            x = x + self.mlp(self.ln_2(x))
        
        return x, aux_loss


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
                TransformerBlock(config, layer_idx=i)
                for i in range(config.n_layer)
            ]),
            ln_f=RMSNorm(config.n_embd) if config.use_rmsnorm else LayerNorm(config.n_embd, bias=config.bias),
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
    
    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model.
        
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Forward Pass: Complete Data Flow from Tokenized Text to Linear Output.
        
        STEP-BY-STEP EXECUTION:
        
        INPUT: idx = tokenized text [batch_size, sequence_length]
        Example: [[15496, 995, 318, 257, 6291]]  # "Hello world is a test"
        
        Args:
            idx: Input token indices of shape (batch_size, sequence_length)
            targets: Target token indices for training (optional)
            
        Returns:
            Tuple of (logits, loss, auxiliary_loss)
            - logits: Output predictions of shape (batch_size, sequence_length, vocab_size)
            - loss: Cross-entropy loss if targets provided, else None
            - auxiliary_loss: MoE load balancing loss
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
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text using the model.
        
        Args:
            idx: Input token indices of shape (batch_size, sequence_length)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
            
        Returns:
            Generated token indices including the input
        """
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _, _ = self(idx_cond)
            
            # Get logits for the last position and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


if __name__ == "__main__":
    # Example usage with detailed explanations
    print("=" * 80)
    print("GPT-OSS: Complete LLM Implementation from Tokenized Text to Linear Output")
    print("=" * 80)
    
    # Create model configuration using the new config system
    config = get_config('gpt-oss-moe-small')  # Use predefined small MoE config
    
    print(f"\nModel Configuration: {config.__class__.__name__}")
    
    # Create model and count parameters
    model = GPTOSS(config)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"- Total Parameters: {total_params:,}")
    print(f"- Layers: {config.n_layer}")
    print(f"- Hidden Size: {config.n_embd}")
    print(f"- Attention Heads: {config.n_head}")
    print(f"- Vocabulary Size: {config.vocab_size:,}")
    print(f"- Context Length: {config.block_size}")
    if config.use_moe:
        print(f"- MoE Experts: {config.num_experts}")
        print(f"- Active Experts per Token: {config.top_k_experts}")
        active_params = total_params // config.num_experts * config.top_k_experts
        print(f"- Active Parameters per Token: ~{active_params:,}")
    
    model.eval()
    
    print(f"\n" + "=" * 50)
    print("FORWARD PASS DEMONSTRATION")
    print("=" * 50)
    
    # Create sample input (tokenized text)
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput Shape: {list(input_ids.shape)} (batch_size, seq_len)")
    print(f"Input represents: Tokenized text ready for processing")
    
    # Forward pass
    with torch.no_grad():
        logits, loss, aux_loss = model(input_ids, targets)
    
    print(f"\nOutput Logits Shape: {list(logits.shape)} (batch_size, seq_len, vocab_size)")
    print(f"Loss: {loss.item():.4f} (cross-entropy loss for training)")
    if aux_loss is not None:
        print(f"Auxiliary Loss: {aux_loss.item():.4f} (MoE load balancing)")
    
    print(f"\n" + "=" * 50)
    print("TEXT GENERATION DEMONSTRATION")
    print("=" * 50)
    
    # Generate text
    start_ids = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(start_ids, max_new_tokens=20, temperature=0.8)
    
    print(f"\nGeneration Input Shape: {list(start_ids.shape)}")
    print(f"Generated Output Shape: {list(generated.shape)}")
    print(f"Generated {generated.shape[1] - start_ids.shape[1]} new tokens")
    
    print(f"\n" + "=" * 80)
    print("COMPLETE LLM PIPELINE SUMMARY")
    print("=" * 80)
    print("""
This implementation demonstrates a complete LLM from tokenized text to linear output:

1. TOKEN EMBEDDINGS: Convert token IDs to dense vectors
2. POSITIONAL ENCODING: Add position information (RoPE)
3. TRANSFORMER LAYERS: Process through attention + feedforward blocks
   - Multi-head attention with RoPE positional encoding
   - MoE feedforward networks for efficient scaling
   - RMS normalization for training stability
   - Residual connections for gradient flow
4. FINAL NORMALIZATION: Stabilize final representations
5. LINEAR OUTPUT LAYER: Project to vocabulary probabilities
6. LOSS COMPUTATION: Cross-entropy + MoE auxiliary loss

KEY ARCHITECTURAL INNOVATIONS:
- Efficient Scaling: MoE provides 6x parameters with minimal compute increase
- Advanced Attention: RoPE enables better length extrapolation
- Modern Components: RMSNorm, SwiGLU activation for better performance
- Training Stability: Pre-normalization, auxiliary losses, gradient clipping

AVAILABLE MODEL CONFIGURATIONS:
- gpt-oss-small: ~124M parameters (dense, for development)
- gpt-oss-moe-small: ~400M total, ~150M active (MoE, for testing)
- gpt-oss-20b: ~20B parameters (dense, production)
- gpt-oss-120b: ~120B total, ~24B active (MoE, large scale)
""")
    
    print("\nImplementation complete! Ready for training and inference.")
    print("=" * 80)