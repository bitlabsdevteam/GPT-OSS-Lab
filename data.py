import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Iterator
import os
import pickle
import requests
from pathlib import Path

# ============================================================================
# DATA UTILITIES
# Data loading, preprocessing, and tokenization utilities for GPT-OSS
# ============================================================================

class SimpleTokenizer:
    """A simple character-level tokenizer for demonstration purposes.
    
    In practice, you would use a more sophisticated tokenizer like:
    - tiktoken (OpenAI's tokenizer)
    - SentencePiece
    - Hugging Face tokenizers
    """
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is None:
            # Create a basic vocabulary with common characters
            chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-'\"\n")
            self.vocab = {ch: i for i, ch in enumerate(chars)}
            self.vocab['<UNK>'] = len(self.vocab)
            self.vocab['<PAD>'] = len(self.vocab)
        else:
            self.vocab = vocab
        
        self.inverse_vocab = {i: ch for ch, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.vocab.get(ch, self.vocab['<UNK>']) for ch in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join([self.inverse_vocab.get(id, '<UNK>') for id in token_ids])
    
    def save(self, filepath: str) -> None:
        """Save tokenizer vocabulary."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.vocab, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SimpleTokenizer':
        """Load tokenizer vocabulary."""
        with open(filepath, 'rb') as f:
            vocab = pickle.load(f)
        return cls(vocab)


class TextDataset(Dataset):
    """Dataset for text data with sliding window approach.
    
    This dataset creates training examples by sliding a window of size
    `block_size` over the text data.
    """
    
    def __init__(self, text_data: str, tokenizer: SimpleTokenizer, 
                 block_size: int, stride: Optional[int] = None):
        """
        Args:
            text_data: Raw text data
            tokenizer: Tokenizer instance
            block_size: Maximum sequence length
            stride: Stride for sliding window (defaults to block_size)
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or block_size
        
        # Tokenize the entire text
        self.tokens = tokenizer.encode(text_data)
        
        # Create sliding windows
        self.examples = []
        for i in range(0, len(self.tokens) - block_size, self.stride):
            self.examples.append(self.tokens[i:i + block_size + 1])  # +1 for target
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Input is all tokens except the last one
        input_ids = torch.tensor(example[:-1], dtype=torch.long)
        
        # Target is all tokens except the first one (shifted by 1)
        target_ids = torch.tensor(example[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': target_ids
        }


class TinyShakespeareDataset:
    """Utility class to download and prepare the Tiny Shakespeare dataset.
    
    This is a small dataset commonly used for language modeling experiments.
    """
    
    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    
    @classmethod
    def download(cls, data_dir: str = "data") -> str:
        """Download the Tiny Shakespeare dataset.
        
        Args:
            data_dir: Directory to save the data
            
        Returns:
            Path to the downloaded file
        """
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, "shakespeare.txt")
        
        if not os.path.exists(filepath):
            print("Downloading Tiny Shakespeare dataset...")
            response = requests.get(cls.URL)
            response.raise_for_status()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"Dataset downloaded to {filepath}")
        else:
            print(f"Dataset already exists at {filepath}")
        
        return filepath
    
    @classmethod
    def load_and_split(cls, data_dir: str = "data", 
                      train_split: float = 0.9) -> Tuple[str, str]:
        """Load and split the dataset into train and validation sets.
        
        Args:
            data_dir: Directory containing the data
            train_split: Fraction of data to use for training
            
        Returns:
            Tuple of (train_text, val_text)
        """
        filepath = cls.download(data_dir)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split the data
        split_idx = int(len(text) * train_split)
        train_text = text[:split_idx]
        val_text = text[split_idx:]
        
        print(f"Dataset loaded: {len(text):,} characters")
        print(f"Train: {len(train_text):,} characters")
        print(f"Validation: {len(val_text):,} characters")
        
        return train_text, val_text


def create_data_loaders(train_text: str, val_text: str, 
                       tokenizer: SimpleTokenizer, block_size: int,
                       batch_size: int = 32, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.
    
    Args:
        train_text: Training text data
        val_text: Validation text data
        tokenizer: Tokenizer instance
        block_size: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = TextDataset(train_text, tokenizer, block_size)
    val_dataset = TextDataset(val_text, tokenizer, block_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Train dataset: {len(train_dataset):,} examples")
    print(f"Validation dataset: {len(val_dataset):,} examples")
    print(f"Train batches: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")
    
    return train_loader, val_loader


def prepare_shakespeare_data(data_dir: str = "data", block_size: int = 256,
                           batch_size: int = 32) -> Tuple[DataLoader, DataLoader, SimpleTokenizer]:
    """Convenience function to prepare Shakespeare data for training.
    
    Args:
        data_dir: Directory to store data
        block_size: Maximum sequence length
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader, tokenizer)
    """
    # Load and split data
    train_text, val_text = TinyShakespeareDataset.load_and_split(data_dir)
    
    # Create tokenizer from training data
    # Get unique characters from training data
    unique_chars = sorted(list(set(train_text)))
    vocab = {ch: i for i, ch in enumerate(unique_chars)}
    vocab['<UNK>'] = len(vocab)
    vocab['<PAD>'] = len(vocab)
    
    tokenizer = SimpleTokenizer(vocab)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_text, val_text, tokenizer, block_size, batch_size
    )
    
    return train_loader, val_loader, tokenizer


class DataCollator:
    """Data collator for batching sequences with padding.
    
    This is useful when sequences have different lengths.
    """
    
    def __init__(self, pad_token_id: int = 0, max_length: Optional[int] = None):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples.
        
        Args:
            batch: List of examples from the dataset
            
        Returns:
            Batched and padded tensors
        """
        # Get the maximum length in the batch
        max_len = max(len(example['input_ids']) for example in batch)
        
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
        
        batch_size = len(batch)
        
        # Initialize tensors
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 is ignored in loss
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        
        # Fill the tensors
        for i, example in enumerate(batch):
            seq_len = min(len(example['input_ids']), max_len)
            
            input_ids[i, :seq_len] = example['input_ids'][:seq_len]
            labels[i, :seq_len] = example['labels'][:seq_len]
            attention_mask[i, :seq_len] = True
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def estimate_tokens_per_epoch(text_length: int, block_size: int, stride: Optional[int] = None) -> int:
    """Estimate the number of tokens processed per epoch.
    
    Args:
        text_length: Length of text in characters/tokens
        block_size: Sequence length
        stride: Stride for sliding window
        
    Returns:
        Estimated number of tokens per epoch
    """
    if stride is None:
        stride = block_size
    
    num_examples = max(1, (text_length - block_size) // stride)
    tokens_per_epoch = num_examples * block_size
    
    return tokens_per_epoch


if __name__ == "__main__":
    # Demo of data utilities
    print("=" * 60)
    print("DATA UTILITIES DEMO")
    print("=" * 60)
    
    # Prepare Shakespeare data
    train_loader, val_loader, tokenizer = prepare_shakespeare_data(
        block_size=128, batch_size=4
    )
    
    # Show a sample batch
    print("\nSample batch:")
    batch = next(iter(train_loader))
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    
    # Decode first example
    print("\nFirst example (input):")
    input_text = tokenizer.decode(batch['input_ids'][0].tolist())
    print(repr(input_text[:100] + "..."))
    
    print("\nFirst example (target):")
    target_text = tokenizer.decode(batch['labels'][0].tolist())
    print(repr(target_text[:100] + "..."))
    
    print("\n" + "=" * 60)