import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Callable
import time
import numpy as np
from model import GPTOSS
from config import get_config
from data import SimpleTokenizer
from utils import set_seed, get_device

# ============================================================================
# TEXT GENERATION UTILITIES
# Inference and text generation functions for GPT-OSS
# ============================================================================

def top_k_sampling(logits: torch.Tensor, k: int = 50, temperature: float = 1.0) -> torch.Tensor:
    """Apply top-k sampling to logits.
    
    Args:
        logits: Raw model outputs [batch_size, vocab_size]
        k: Number of top tokens to consider
        temperature: Sampling temperature (higher = more random)
        
    Returns:
        Sampled token indices
    """
    # Apply temperature
    logits = logits / temperature
    
    # Get top-k values and indices
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # Create a mask for top-k tokens
    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(-1, top_k_indices, top_k_values)
    
    # Sample from the filtered distribution
    probs = F.softmax(mask, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


def top_p_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """Apply top-p (nucleus) sampling to logits.
    
    Args:
        logits: Raw model outputs [batch_size, vocab_size]
        p: Cumulative probability threshold
        temperature: Sampling temperature
        
    Returns:
        Sampled token indices
    """
    # Apply temperature
    logits = logits / temperature
    
    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Create mask for tokens to keep
    sorted_indices_to_remove = cumulative_probs > p
    
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # Set logits to -inf for tokens to remove
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    """Apply greedy sampling (always pick the most likely token).
    
    Args:
        logits: Raw model outputs [batch_size, vocab_size]
        
    Returns:
        Most likely token indices
    """
    return torch.argmax(logits, dim=-1, keepdim=True)


class TextGenerator:
    """Text generation class with various sampling strategies."""
    
    def __init__(self, model: GPTOSS, tokenizer: SimpleTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def generate(self, 
                prompt: str,
                max_new_tokens: int = 100,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                do_sample: bool = True,
                repetition_penalty: float = 1.0,
                pad_token_id: Optional[int] = None,
                eos_token_id: Optional[int] = None,
                seed: Optional[int] = None) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            repetition_penalty: Penalty for repeating tokens
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            seed: Random seed for reproducible generation
            
        Returns:
            Generated text
        """
        if seed is not None:
            set_seed(seed)
        
        # Encode the prompt
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt), 
            dtype=torch.long, 
            device=self.device
        ).unsqueeze(0)  # Add batch dimension
        
        # Generation loop
        generated_tokens = []
        
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs['logits']
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, input_ids, repetition_penalty
                )
            
            # Sample next token
            if do_sample:
                if top_k is not None:
                    next_token = top_k_sampling(next_token_logits, top_k, temperature)
                elif top_p is not None:
                    next_token = top_p_sampling(next_token_logits, top_p, temperature)
                else:
                    # Standard temperature sampling
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = greedy_sampling(next_token_logits)
            
            # Check for end of sequence
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
            
            # Add to generated tokens
            generated_tokens.append(next_token.item())
            
            # Update input_ids for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Truncate if sequence gets too long (to prevent memory issues)
            if input_ids.size(1) > self.model.config.block_size:
                input_ids = input_ids[:, -self.model.config.block_size:]
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, 
                                 input_ids: torch.Tensor, 
                                 penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits.
        
        Args:
            logits: Next token logits
            input_ids: Previous token IDs
            penalty: Repetition penalty factor
            
        Returns:
            Modified logits
        """
        # Get unique tokens in the sequence
        unique_tokens = torch.unique(input_ids)
        
        # Apply penalty to tokens that have appeared before
        for token in unique_tokens:
            if logits[0, token] > 0:
                logits[0, token] /= penalty
            else:
                logits[0, token] *= penalty
        
        return logits
    
    def generate_batch(self, 
                      prompts: List[str],
                      max_new_tokens: int = 100,
                      **kwargs) -> List[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate per prompt
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        results = []
        
        for prompt in prompts:
            generated = self.generate(prompt, max_new_tokens, **kwargs)
            results.append(generated)
        
        return results
    
    def interactive_generation(self):
        """Interactive text generation session."""
        print("=" * 60)
        print("INTERACTIVE TEXT GENERATION")
        print("Type 'quit' to exit, 'help' for commands")
        print("=" * 60)
        
        # Default parameters
        params = {
            'max_new_tokens': 100,
            'temperature': 1.0,
            'top_k': None,
            'top_p': None,
            'do_sample': True
        }
        
        while True:
            try:
                user_input = input("\nPrompt: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self._print_help()
                    continue
                elif user_input.startswith('set '):
                    self._handle_parameter_setting(user_input, params)
                    continue
                elif user_input.lower() == 'params':
                    print(f"Current parameters: {params}")
                    continue
                
                if not user_input:
                    continue
                
                # Generate text
                print("\nGenerating...")
                start_time = time.time()
                
                generated = self.generate(user_input, **params)
                
                end_time = time.time()
                
                print(f"\nGenerated text:")
                print(f"{user_input}{generated}")
                print(f"\nGeneration time: {end_time - start_time:.2f}s")
                print(f"Tokens generated: {len(self.tokenizer.encode(generated))}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _print_help(self):
        """Print help information."""
        print("\nCommands:")
        print("  help - Show this help message")
        print("  quit - Exit the interactive session")
        print("  params - Show current parameters")
        print("  set <param> <value> - Set a parameter")
        print("\nParameters:")
        print("  max_new_tokens - Maximum tokens to generate (int)")
        print("  temperature - Sampling temperature (float)")
        print("  top_k - Top-k sampling (int or None)")
        print("  top_p - Top-p sampling (float or None)")
        print("  do_sample - Use sampling vs greedy (true/false)")
        print("\nExample: set temperature 0.8")
    
    def _handle_parameter_setting(self, user_input: str, params: Dict[str, Any]):
        """Handle parameter setting commands."""
        try:
            parts = user_input.split()
            if len(parts) != 3:
                print("Usage: set <parameter> <value>")
                return
            
            param_name = parts[1]
            param_value = parts[2]
            
            if param_name not in params:
                print(f"Unknown parameter: {param_name}")
                return
            
            # Convert value to appropriate type
            if param_name in ['max_new_tokens', 'top_k']:
                if param_value.lower() == 'none':
                    params[param_name] = None
                else:
                    params[param_name] = int(param_value)
            elif param_name in ['temperature', 'top_p']:
                if param_value.lower() == 'none':
                    params[param_name] = None
                else:
                    params[param_name] = float(param_value)
            elif param_name == 'do_sample':
                params[param_name] = param_value.lower() in ['true', '1', 'yes']
            
            print(f"Set {param_name} = {params[param_name]}")
            
        except ValueError as e:
            print(f"Invalid value: {e}")
        except Exception as e:
            print(f"Error setting parameter: {e}")


def benchmark_generation(model: GPTOSS, tokenizer: SimpleTokenizer, 
                        device: torch.device, num_tokens: int = 100) -> Dict[str, float]:
    """Benchmark text generation speed.
    
    Args:
        model: GPT-OSS model
        tokenizer: Tokenizer
        device: Device to run on
        num_tokens: Number of tokens to generate
        
    Returns:
        Benchmark results
    """
    generator = TextGenerator(model, tokenizer, device)
    
    prompt = "The quick brown fox"
    
    # Warmup
    generator.generate(prompt, max_new_tokens=10, do_sample=False)
    
    # Benchmark
    start_time = time.time()
    generated = generator.generate(prompt, max_new_tokens=num_tokens, do_sample=False)
    end_time = time.time()
    
    total_time = end_time - start_time
    tokens_per_second = num_tokens / total_time
    
    return {
        'total_time': total_time,
        'tokens_generated': num_tokens,
        'tokens_per_second': tokens_per_second,
        'generated_text_length': len(generated)
    }


if __name__ == "__main__":
    # Demo of text generation
    print("=" * 60)
    print("TEXT GENERATION DEMO")
    print("=" * 60)
    
    # Load model and tokenizer
    device = get_device()
    config = get_config('gpt-oss-small')
    
    # Create a simple tokenizer for demo
    chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-'\"\n")
    vocab = {ch: i for i, ch in enumerate(chars)}
    vocab['<UNK>'] = len(vocab)
    vocab['<PAD>'] = len(vocab)
    tokenizer = SimpleTokenizer(vocab)
    
    # Update config with correct vocab size
    config.vocab_size = tokenizer.vocab_size
    
    # Create model
    model = GPTOSS(config)
    
    # Create generator
    generator = TextGenerator(model, tokenizer, device)
    
    # Test different sampling strategies
    prompt = "Hello world"
    
    print(f"Prompt: '{prompt}'")
    print("\nGreedy decoding:")
    result = generator.generate(prompt, max_new_tokens=50, do_sample=False)
    print(f"'{prompt}{result}'")
    
    print("\nTemperature sampling (T=0.8):")
    result = generator.generate(prompt, max_new_tokens=50, temperature=0.8, seed=42)
    print(f"'{prompt}{result}'")
    
    print("\nTop-k sampling (k=10):")
    result = generator.generate(prompt, max_new_tokens=50, top_k=10, seed=42)
    print(f"'{prompt}{result}'")
    
    print("\nTop-p sampling (p=0.9):")
    result = generator.generate(prompt, max_new_tokens=50, top_p=0.9, seed=42)
    print(f"'{prompt}{result}'")
    
    # Benchmark
    print("\nBenchmarking generation speed...")
    benchmark_results = benchmark_generation(model, tokenizer, device, num_tokens=100)
    print(f"Generated {benchmark_results['tokens_generated']} tokens in {benchmark_results['total_time']:.2f}s")
    print(f"Speed: {benchmark_results['tokens_per_second']:.1f} tokens/second")
    
    print("\n" + "=" * 60)
    print("Run with --interactive for interactive generation")
    
    # Uncomment to start interactive session
    # generator.interactive_generation()