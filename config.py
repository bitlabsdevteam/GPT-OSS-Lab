from dataclasses import dataclass
from typing import Optional

# ============================================================================
# MODEL CONFIGURATIONS
# These configurations define different model variants and their parameters
# ============================================================================

@dataclass
class GPTOSSConfig:
    """Configuration class for GPT-OSS models.
    
    This class defines all the hyperparameters and architectural choices
    for the GPT-OSS model variants. It supports both regular and MoE configurations.
    
    DESIGN PHILOSOPHY:
    - Modular: Easy to create new variants
    - Scalable: Supports models from millions to hundreds of billions of parameters
    - Flexible: Can enable/disable different features
    - Research-friendly: Easy to experiment with different architectures
    """
    
    # Model Architecture
    block_size: int = 2048          # Maximum sequence length
    vocab_size: int = 50304         # Vocabulary size (rounded to nearest multiple of 64)
    n_layer: int = 12               # Number of transformer layers
    n_head: int = 12                # Number of attention heads
    n_embd: int = 768               # Embedding dimension
    
    # Regularization
    dropout: float = 0.1            # Dropout probability
    bias: bool = False              # Whether to use bias in linear layers
    
    # Mixture of Experts (MoE) Configuration
    use_moe: bool = False           # Whether to use MoE layers
    num_experts: int = 8            # Number of experts in MoE layers
    top_k_experts: int = 2          # Number of experts to activate per token
    moe_layers: Optional[list] = None  # Which layers should use MoE (None = all)
    router_jitter_noise: float = 0.01  # Noise for router load balancing
    
    # Advanced Features
    use_flash_attention: bool = True    # Use Flash Attention if available
    use_rope: bool = True              # Use Rotary Positional Embedding
    use_rmsnorm: bool = True           # Use RMSNorm instead of LayerNorm
    use_swiglu: bool = True            # Use SwiGLU activation in FFN
    
    # Training Configuration
    weight_decay: float = 0.1          # Weight decay for optimizer
    learning_rate: float = 3e-4        # Learning rate
    beta1: float = 0.9                 # Adam beta1
    beta2: float = 0.95                # Adam beta2
    grad_clip: float = 1.0             # Gradient clipping value
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure vocab_size is divisible by 64 for efficiency
        if self.vocab_size % 64 != 0:
            self.vocab_size = ((self.vocab_size // 64) + 1) * 64
        
        # Validate MoE configuration
        if self.use_moe:
            assert self.num_experts > self.top_k_experts, "num_experts must be > top_k_experts"
            assert self.top_k_experts >= 1, "top_k_experts must be >= 1"
            
            # Default: use MoE in all layers if not specified
            if self.moe_layers is None:
                self.moe_layers = list(range(self.n_layer))
        
        # Validate attention configuration
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"


# ============================================================================
# PREDEFINED MODEL CONFIGURATIONS
# These are standard configurations for different model sizes
# ============================================================================

def get_gpt_oss_20b_config() -> GPTOSSConfig:
    """GPT-OSS 20B - Dense model configuration.
    
    SPECIFICATIONS:
    - Parameters: ~20 billion
    - Architecture: Dense transformer
    - Context: 2048 tokens
    - Vocabulary: 50,304 tokens
    - Layers: 44
    - Attention heads: 64
    - Hidden dimension: 4096
    
    USE CASES:
    - General language modeling
    - Fine-tuning for specific tasks
    - Research and experimentation
    - Baseline for MoE comparisons
    """
    return GPTOSSConfig(
        # Architecture
        block_size=2048,
        vocab_size=50304,
        n_layer=44,
        n_head=64,
        n_embd=4096,
        
        # Regularization
        dropout=0.1,
        bias=False,
        
        # No MoE for dense model
        use_moe=False,
        
        # Advanced features
        use_flash_attention=True,
        use_rope=True,
        use_rmsnorm=True,
        use_swiglu=True,
    )


def get_gpt_oss_120b_config() -> GPTOSSConfig:
    """GPT-OSS 120B - Mixture of Experts configuration.
    
    SPECIFICATIONS:
    - Total parameters: ~120 billion
    - Active parameters per token: ~24 billion (top-2 of 64 experts)
    - Architecture: Sparse MoE transformer
    - Context: 2048 tokens
    - Vocabulary: 50,304 tokens
    - Layers: 64
    - Attention heads: 64
    - Hidden dimension: 4096
    - Experts: 64 per MoE layer
    - Active experts: 2 per token
    
    EFFICIENCY:
    - 6x more parameters than 20B model
    - Only ~20% more compute per token
    - Massive scaling with minimal overhead
    
    USE CASES:
    - Large-scale language modeling
    - Multi-domain expertise
    - Research on sparse models
    - Efficient scaling experiments
    """
    return GPTOSSConfig(
        # Architecture
        block_size=2048,
        vocab_size=50304,
        n_layer=64,
        n_head=64,
        n_embd=4096,
        
        # Regularization
        dropout=0.1,
        bias=False,
        
        # MoE Configuration
        use_moe=True,
        num_experts=64,
        top_k_experts=2,
        moe_layers=None,  # Use MoE in all layers
        router_jitter_noise=0.01,
        
        # Advanced features
        use_flash_attention=True,
        use_rope=True,
        use_rmsnorm=True,
        use_swiglu=True,
    )


def get_gpt_oss_small_config() -> GPTOSSConfig:
    """GPT-OSS Small - Configuration for testing and development.
    
    SPECIFICATIONS:
    - Parameters: ~124 million
    - Architecture: Dense transformer
    - Context: 1024 tokens
    - Vocabulary: 50,304 tokens
    - Layers: 12
    - Attention heads: 12
    - Hidden dimension: 768
    
    USE CASES:
    - Development and testing
    - Quick experiments
    - Educational purposes
    - Proof of concept
    """
    return GPTOSSConfig(
        # Architecture
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        
        # Regularization
        dropout=0.1,
        bias=False,
        
        # No MoE for small model
        use_moe=False,
        
        # Advanced features
        use_flash_attention=True,
        use_rope=True,
        use_rmsnorm=True,
        use_swiglu=True,
    )


def get_gpt_oss_moe_small_config() -> GPTOSSConfig:
    """GPT-OSS MoE Small - Small MoE configuration for testing.
    
    SPECIFICATIONS:
    - Total parameters: ~400 million
    - Active parameters per token: ~150 million
    - Architecture: Sparse MoE transformer
    - Context: 1024 tokens
    - Vocabulary: 50,304 tokens
    - Layers: 12
    - Attention heads: 12
    - Hidden dimension: 768
    - Experts: 8 per MoE layer
    - Active experts: 2 per token
    
    USE CASES:
    - MoE development and testing
    - Understanding sparse models
    - Educational MoE examples
    - Quick MoE experiments
    """
    return GPTOSSConfig(
        # Architecture
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        
        # Regularization
        dropout=0.1,
        bias=False,
        
        # MoE Configuration
        use_moe=True,
        num_experts=8,
        top_k_experts=2,
        moe_layers=None,  # Use MoE in all layers
        router_jitter_noise=0.01,
        
        # Advanced features
        use_flash_attention=True,
        use_rope=True,
        use_rmsnorm=True,
        use_swiglu=True,
    )


# ============================================================================
# CONFIGURATION REGISTRY
# Easy access to all predefined configurations
# ============================================================================

CONFIG_REGISTRY = {
    'gpt-oss-small': get_gpt_oss_small_config,
    'gpt-oss-moe-small': get_gpt_oss_moe_small_config,
    'gpt-oss-20b': get_gpt_oss_20b_config,
    'gpt-oss-120b': get_gpt_oss_120b_config,
}


def get_config(model_name: str) -> GPTOSSConfig:
    """Get a predefined configuration by name.
    
    Args:
        model_name: Name of the model configuration
        
    Returns:
        GPTOSSConfig instance
        
    Raises:
        ValueError: If model_name is not found in registry
    """
    if model_name not in CONFIG_REGISTRY:
        available = list(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    return CONFIG_REGISTRY[model_name]()


def list_available_configs() -> list:
    """List all available predefined configurations.
    
    Returns:
        List of available configuration names
    """
    return list(CONFIG_REGISTRY.keys())