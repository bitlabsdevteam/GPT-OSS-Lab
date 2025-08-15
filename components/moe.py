import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# ============================================================================
# MIXTURE OF EXPERTS (MoE) COMPONENTS
# These components implement the MoE system for efficient scaling
# ============================================================================

class Router(nn.Module):
    """Router - Traffic Controller for Mixture of Experts.
    
    WHAT IT DOES:
    - Decides which expert(s) should process each token
    - Learns to route tokens based on their content and context
    - Enables sparse computation (only active experts compute)
    
    HOW IT WORKS:
    1. Takes token embeddings as input
    2. Computes routing probabilities for each expert
    3. Selects top-k experts per token (usually k=2)
    4. Returns expert indices and routing weights
    
    WHY IT'S CRUCIAL:
    - Enables scaling to thousands of experts
    - Only 2-4 experts active per token (sparse computation)
    - Learns specialization automatically during training
    - Provides 10x+ parameter scaling with minimal compute increase
    
    ROUTING STRATEGIES:
    - Top-k routing: Select k best experts per token
    - Load balancing: Ensure experts get similar amounts of work
    - Auxiliary loss: Encourage balanced expert usage
    
    MATHEMATICAL FOUNDATION:
    Router(x) = softmax(xW_router)
    Top-k selection ensures sparsity and efficiency
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        
        # Router network: maps input to expert probabilities
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        
        # For load balancing
        self.jitter_noise = config.router_jitter_noise if hasattr(config, 'router_jitter_noise') else 0.01
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            
        Returns:
            expert_indices: Selected expert indices [batch_size, seq_len, top_k]
            expert_weights: Routing weights [batch_size, seq_len, top_k] 
            aux_loss: Auxiliary loss for load balancing
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Flatten for routing: [batch_size * seq_len, hidden_dim]
        x_flat = x.view(-1, hidden_dim)
        
        # Add jitter noise during training for better load balancing
        if self.training and self.jitter_noise > 0:
            x_flat = x_flat + torch.randn_like(x_flat) * self.jitter_noise
        
        # Compute routing logits: [batch_size * seq_len, num_experts]
        router_logits = self.router(x_flat)
        
        # Convert to probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts per token
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize weights (so they sum to 1 for each token)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Reshape back: [batch_size, seq_len, top_k]
        expert_indices = expert_indices.view(batch_size, seq_len, self.top_k)
        expert_weights = expert_weights.view(batch_size, seq_len, self.top_k)
        
        # Compute auxiliary loss for load balancing
        aux_loss = self._compute_aux_loss(router_probs)
        
        return expert_indices, expert_weights, aux_loss
    
    def _compute_aux_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss to encourage load balancing.
        
        This loss encourages equal usage of all experts, preventing
        the model from using only a few experts.
        """
        # Average probability of routing to each expert
        expert_usage = router_probs.mean(dim=0)  # [num_experts]
        
        # Ideal usage would be 1/num_experts for each expert
        ideal_usage = 1.0 / self.num_experts
        
        # L2 loss between actual and ideal usage
        aux_loss = torch.sum((expert_usage - ideal_usage) ** 2)
        
        return aux_loss


class Expert(nn.Module):
    """Expert - Specialized Processing Unit in MoE.
    
    WHAT IT DOES:
    - Each expert is a specialized feedforward network
    - Learns to handle specific types of tokens or patterns
    - Provides the actual computation in the MoE system
    
    HOW IT WORKS:
    - Standard feedforward network (expand → activate → contract)
    - Same architecture as regular FFN but specialized through routing
    - Multiple experts provide diverse processing capabilities
    
    WHY SPECIALIZATION EMERGES:
    - Router learns to send similar tokens to the same experts
    - Experts adapt to handle their assigned token types well
    - Natural specialization emerges (e.g., syntax vs semantics)
    - Each expert becomes an "expert" in its domain
    
    EXAMPLES OF LEARNED SPECIALIZATION:
    - Expert 1: Handles mathematical expressions
    - Expert 2: Processes named entities
    - Expert 3: Manages syntactic structures
    - Expert 4: Handles rare or technical vocabulary
    
    ARCHITECTURAL DETAILS:
    - Same capacity as regular FFN
    - Can use different activation functions per expert
    - Dropout for regularization
    - Efficient implementation for sparse computation
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_dim = 4 * config.n_embd  # Standard 4x expansion
        
        # Two-layer MLP (same as regular feedforward)
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process tokens assigned to this expert.
        
        Args:
            x: Input tensor [num_tokens, n_embd]
            
        Returns:
            Output tensor [num_tokens, n_embd]
        """
        # Expand and activate
        x = self.w1(x)
        x = F.gelu(x)  # Can be different activation per expert
        
        # Contract and regularize
        x = self.w2(x)
        x = self.dropout(x)
        
        return x


class MixtureOfExperts(nn.Module):
    """MixtureOfExperts - Coordinating Router and Experts.
    
    WHAT IT DOES:
    - Coordinates the entire MoE system
    - Routes tokens to appropriate experts
    - Combines expert outputs with learned weights
    - Manages sparse computation efficiently
    
    HOW THE COMPLETE MoE PROCESS WORKS:
    1. INPUT: Tokens arrive [batch_size, seq_len, n_embd]
    2. ROUTING: Router decides which experts handle each token
    3. DISPATCH: Tokens sent to selected experts (sparse)
    4. PROCESSING: Only active experts compute (efficiency!)
    5. COMBINE: Expert outputs weighted and summed
    6. OUTPUT: Final result [batch_size, seq_len, n_embd]
    
    EFFICIENCY BENEFITS:
    - Total parameters: num_experts × expert_size
    - Active parameters per token: top_k × expert_size
    - Example: 64 experts, top-2 → only 3% of experts active per token
    - Massive scaling with minimal compute increase
    
    IMPLEMENTATION DETAILS:
    - Efficient batching of expert computation
    - Load balancing to prevent expert collapse
    - Auxiliary loss for training stability
    - Support for different expert architectures
    
    SCALING EXAMPLE:
    - Regular model: 20B parameters, 20B active
    - MoE model: 120B parameters, 24B active (top-2 of 64 experts)
    - Result: 6x more parameters, only 20% more compute!
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        
        # Router for expert selection
        self.router = Router(config)
        
        # Create all experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through MoE system.
        
        Args:
            x: Input tensor [batch_size, seq_len, n_embd]
            
        Returns:
            output: Processed tensor [batch_size, seq_len, n_embd]
            aux_loss: Auxiliary loss for load balancing
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Route tokens to experts
        expert_indices, expert_weights, aux_loss = self.router(x)
        
        # Flatten input for expert processing
        x_flat = x.view(-1, hidden_dim)  # [batch_size * seq_len, hidden_dim]
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process tokens through selected experts
        for i in range(self.top_k):
            # Get expert indices and weights for position i
            expert_idx = expert_indices[:, :, i].flatten()  # [batch_size * seq_len]
            weights = expert_weights[:, :, i].flatten().unsqueeze(-1)  # [batch_size * seq_len, 1]
            
            # Process each expert
            for expert_id in range(self.num_experts):
                # Find tokens assigned to this expert
                expert_mask = (expert_idx == expert_id)
                if expert_mask.any():
                    # Get tokens for this expert
                    expert_input = x_flat[expert_mask]
                    
                    # Process through expert
                    expert_output = self.experts[expert_id](expert_input)
                    
                    # Add weighted output back
                    output[expert_mask] += weights[expert_mask] * expert_output
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, hidden_dim)
        
        return output, aux_loss