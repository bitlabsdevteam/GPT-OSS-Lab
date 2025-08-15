# GPT-OSS: Open Source GPT with Mixture of Experts

A complete implementation of GPT-OSS (Open Source GPT) with Mixture of Experts (MoE) architecture, built from scratch following the architectural diagram specifications for 20B and 120B parameter variants.

## 🏗️ Architecture Overview

GPT-OSS implements a modern transformer architecture with the following key components:

### Core Components
- **Multi-Head Attention** with Rotary Position Embeddings (RoPE)
- **Mixture of Experts (MoE)** layers for efficient scaling
- **SwiGLU** activation function for improved performance
- **RMSNorm** for better training stability
- **Flash Attention** support for memory efficiency

### Model Variants
- **GPT-OSS 20B**: 32 layers, 32 heads, 2048 embedding dimension
- **GPT-OSS 120B**: 64 layers, 64 heads, 4096 embedding dimension
- Both variants support 200K vocabulary and 131K context length

## 🚀 Key Features

### 1. Mixture of Experts (MoE)
- **Top-K Routing**: Each token is routed to the top-2 most relevant experts
- **Load Balancing**: Auxiliary loss ensures uniform expert utilization
- **Efficient Computation**: Only activated experts process tokens
- **Scalable Architecture**: Easy to add more experts without linear parameter growth

### 2. Advanced Attention Mechanism
- **Rotary Position Embeddings (RoPE)**: Better length extrapolation
- **Flash Attention**: Memory-efficient attention computation
- **Multi-Head Architecture**: Parallel attention computation

### 3. Modern Architectural Choices
- **SwiGLU Activation**: Gated linear units with Swish activation
- **RMSNorm**: Root Mean Square normalization for stability
- **Pre-Normalization**: Layer norm applied before attention/MLP
- **Weight Tying**: Shared embeddings between input and output layers

## 📁 File Structure

```
GPT-OSS-LAB/
├── MoE.py          # Main implementation file
└── README.md       # This documentation
```

## 🔧 Implementation Details

### Core Classes

#### `GPTOSS`
Main model class implementing the complete GPT-OSS architecture.

#### `TransformerBlock`
Individual transformer layer with optional MoE integration.

#### `MixtureOfExperts`
MoE layer with top-k routing and load balancing.

#### `MultiHeadAttention`
Attention mechanism with RoPE and Flash Attention support.

#### `Router`
Top-k router for expert selection with auxiliary loss.

#### `Expert`
Individual expert network (feed-forward layer).

### Configuration Classes

#### `GPTOSSConfig`
Comprehensive configuration for model architecture and training.

## 🎯 Usage Examples

### Basic Model Creation

```python
from MoE import GPTOSS, GPTOSSConfig

# Create a custom configuration
config = GPTOSSConfig(
    vocab_size=50304,
    block_size=1024,
    n_layer=12,
    n_head=12,
    n_embd=768,
    n_experts=8,
    top_k=2,
)

# Initialize model
model = GPTOSS(config)
print(f"Model has {model.get_num_params():,} parameters")
```

### Using Pre-configured Models

```python
from MoE import get_gpt_oss_20b_config, get_gpt_oss_120b_config

# 20B parameter model
config_20b = get_gpt_oss_20b_config()
model_20b = GPTOSS(config_20b)

# 120B parameter model
config_120b = get_gpt_oss_120b_config()
model_120b = GPTOSS(config_120b)
```

### Training Example

```python
import torch
import torch.nn.functional as F

# Prepare data
batch_size, seq_len = 4, 512
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Forward pass
logits, loss, aux_loss = model(input_ids, targets)

# Total loss includes auxiliary loss for load balancing
total_loss = loss + aux_loss

# Backward pass
total_loss.backward()
```

### Text Generation

```python
# Generate text
prompt = torch.randint(0, config.vocab_size, (1, 10))  # Starting tokens
generated = model.generate(
    prompt, 
    max_new_tokens=100, 
    temperature=0.8, 
    top_k=50
)
print(f"Generated {generated.shape[1]} tokens")
```

## ⚙️ Configuration Parameters

### Model Architecture
- `vocab_size`: Vocabulary size (default: 200,000)
- `block_size`: Maximum sequence length (default: 131,072)
- `n_layer`: Number of transformer layers
- `n_head`: Number of attention heads
- `n_embd`: Embedding dimension

### MoE Configuration
- `n_experts`: Number of experts per MoE layer
- `top_k`: Number of experts to route each token to
- `moe_layers`: List of layer indices that use MoE
- `capacity_factor`: Expert capacity multiplier

### Training Configuration
- `dropout`: Dropout probability
- `bias`: Whether to use bias in linear layers
- `use_aux_loss`: Enable auxiliary loss for load balancing
- `aux_loss_weight`: Weight for auxiliary loss

### Advanced Features
- `use_rms_norm`: Use RMSNorm instead of LayerNorm
- `use_swiglu`: Use SwiGLU activation instead of GELU

## 🔬 Technical Innovations

### 1. Efficient MoE Implementation
- **Batched Expert Processing**: Efficient computation across experts
- **Load Balancing**: Auxiliary loss prevents expert collapse
- **Capacity Management**: Prevents expert overload

### 2. Memory Optimizations
- **Flash Attention**: Reduces memory usage in attention computation
- **Weight Tying**: Shares parameters between embedding layers
- **Gradient Checkpointing Ready**: Can be easily integrated

### 3. Training Stability
- **RMSNorm**: More stable than LayerNorm
- **Proper Weight Initialization**: GPT-2 style initialization
- **Residual Scaling**: Scaled initialization for deep networks

## 📊 Model Statistics

### GPT-OSS 20B Configuration
- **Total Parameters**: ~20B
- **Active Parameters**: ~2.5B per forward pass
- **Experts**: 8 per MoE layer
- **Context Length**: 131K tokens
- **Vocabulary**: 200K tokens

### GPT-OSS 120B Configuration
- **Total Parameters**: ~120B
- **Active Parameters**: ~7.5B per forward pass
- **Experts**: 16 per MoE layer
- **Context Length**: 131K tokens
- **Vocabulary**: 200K tokens

## 🛠️ Requirements

```bash
pip install torch>=2.0.0
```

### Optional Dependencies
- Flash Attention 2 for memory efficiency
- Transformers library for tokenization
- Datasets library for training data

## 🚀 Getting Started

1. **Clone or download** the implementation files
2. **Install PyTorch** 2.0 or later
3. **Run the example**:
   ```bash
   python MoE.py
   ```
4. **Customize** the configuration for your use case
5. **Train** on your dataset

## 🔍 Architecture Highlights

### Mixture of Experts Benefits
- **Scalability**: Add experts without increasing computation per token
- **Specialization**: Different experts learn different patterns
- **Efficiency**: Only a subset of parameters are active per token
- **Performance**: Better performance than dense models of similar compute

### Modern Transformer Features
- **RoPE**: Better positional encoding for long sequences
- **SwiGLU**: Improved activation function
- **RMSNorm**: More stable normalization
- **Flash Attention**: Memory-efficient attention

## 📈 Performance Characteristics

- **Memory Efficient**: Flash Attention and optimized MoE routing
- **Compute Efficient**: Only top-k experts active per token
- **Scalable**: Easy to scale to larger models
- **Stable Training**: Modern normalization and initialization

## 🤝 Contributing

This implementation serves as a foundation for:
- Research into MoE architectures
- Educational purposes
- Custom model development
- Experimentation with different configurations

## 📚 References

- **Attention Is All You Need** (Transformer architecture)
- **Switch Transformer** (MoE concepts)
- **RoFormer** (Rotary Position Embeddings)
- **GLU Variants** (SwiGLU activation)
- **Root Mean Square Layer Normalization** (RMSNorm)

## ⚠️ Notes

- This is a research/educational implementation
- For production use, consider additional optimizations
- Training large models requires significant computational resources
- The implementation prioritizes clarity and completeness

---

**Built with ❤️ for the open source community**

Implementation follows the GPT-OSS architecture diagram specifications with modern best practices and efficient MoE routing.