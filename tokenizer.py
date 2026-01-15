import re
import string
from typing import List, Dict, Tuple, Optional
import pickle
import json


class BERTTokenizer:
    """
    Custom BERT tokenizer with WordPiece tokenization.
    Implements basic tokenization (lowercasing, punctuation splitting) and
    WordPiece tokenization (subword tokenization).
    """
    
    def __init__(self, vocab_size: int = 30522):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inv_vocab = {}
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"
        
        self.pad_idx = 0
        self.cls_idx = 101
        self.sep_idx = 102
        self.unk_idx = 100
        self.mask_idx = 103
        
        self._init_vocab()
    
    def _init_vocab(self):
        """Initialize vocabulary with special tokens."""
        idx = 0
        
        # Add special tokens
        special_tokens = [
            self.pad_token,      # 0
            "[unused1]",         # 1-99
        ]
        for i in range(1, 100):
            special_tokens.append(f"[unused{i}]")
        
        special_tokens.extend([
            self.unk_token,      # 100
            self.cls_token,      # 101
            self.sep_token,      # 102
            self.mask_token,     # 103
        ])
        
        for token in special_tokens:
            self.vocab[token] = idx
            self.inv_vocab[idx] = token
            idx += 1
        
        # Add common English tokens
        common_words = [
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
            'fake', 'news', 'real', 'true', 'false', 'article', 'report', 'story', 'claim',
            'said', 'according', 'sources', 'police', 'officials', 'government', 'new'
        ]
        
        for word in common_words:
            if word not in self.vocab and idx < self.vocab_size:
                self.vocab[word] = idx
                self.inv_vocab[idx] = word
                idx += 1
    
    def basic_tokenize(self, text: str) -> List[str]:
        """
        Basic tokenization: lowercase, remove accents, split on punctuation.
        """
        # Lowercase
        text = text.lower()
        
        # Add spaces around punctuation
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split on whitespace
        tokens = text.split()
        
        return tokens
    
    def wordpiece_tokenize(self, word: str, max_input_chars_per_word: int = 100) -> List[str]:
        """
        WordPiece tokenization for a single word.
        Greedily matches the longest subword from the vocabulary.
        """
        if len(word) > max_input_chars_per_word:
            return [self.unk_token]
        
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            cur_substr = None
            
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = f"##" + substr
                
                if substr in self.vocab:
                    cur_substr = substr
                    break
                
                end -= 1
            
            if cur_substr is None:
                tokens.append(self.unk_token)
                start += 1
            else:
                tokens.append(cur_substr)
                start = end
        
        return tokens
    
    def tokenize(self, text: str) -> List[str]:
        """
        Full tokenization: basic tokenization + WordPiece tokenization.
        """
        tokens = []
        basic_tokens = self.basic_tokenize(text)
        
        for token in basic_tokens:
            wordpiece_tokens = self.wordpiece_tokenize(token)
            tokens.extend(wordpiece_tokens)
        
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to their IDs."""
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.unk_idx)
        return ids
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs back to tokens."""
        tokens = []
        for idx in ids:
            if idx in self.inv_vocab:
                tokens.append(self.inv_vocab[idx])
            else:
                tokens.append(self.unk_token)
        return tokens
    
    def encode(
        self,
        text: str,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, List[int]]:
        """
        Encode text to token IDs with padding/truncation.
        Returns dict with 'input_ids', 'token_type_ids', 'attention_mask'.
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Add [CLS] and [SEP] tokens
        tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # Truncate if necessary
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [self.sep_token]
        
        # Convert to IDs
        input_ids = self.convert_tokens_to_ids(tokens)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        if padding:
            pad_length = max_length - len(input_ids)
            input_ids += [self.pad_idx] * pad_length
            attention_mask += [0] * pad_length
        
        # Create token type IDs (all 0s for single sentence)
        token_type_ids = [0] * len(input_ids)
        
        return {
            'input_ids': input_ids[:max_length],
            'token_type_ids': token_type_ids[:max_length],
            'attention_mask': attention_mask[:max_length]
        }
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        """
        tokens = self.convert_ids_to_tokens(ids)
        
        # Remove special tokens if requested
        if skip_special_tokens:
            tokens = [t for t in tokens if not t.startswith('[')]
        
        # Remove WordPiece marker (##)
        text = ""
        for token in tokens:
            if token.startswith('##'):
                text += token[2:]
            else:
                if text:
                    text += " "
                text += token
        
        return text.strip()
    
    def save_vocab(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'w') as f:
            for token, idx in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(f"{token}\n")
    
    def load_vocab(self, path: str):
        """Load vocabulary from file."""
        self.vocab = {}
        self.inv_vocab = {}
        
        with open(path, 'r') as f:
            for idx, token in enumerate(f):
                token = token.strip()
                self.vocab[token] = idx
                self.inv_vocab[idx] = token
    
    def add_tokens(self, tokens: List[str]):
        """Add new tokens to vocabulary."""
        start_idx = len(self.vocab)
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = start_idx
                self.inv_vocab[start_idx] = token
                start_idx += 1
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)


class BatchTokenizer:
    """Batch tokenization utility for processing multiple texts."""
    
    def __init__(self, tokenizer: BERTTokenizer):
        self.tokenizer = tokenizer
    
    def tokenize_batch(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: bool = False
    ) -> Dict[str, List]:
        """
        Tokenize a batch of texts.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            return_tensors: Whether to return PyTorch tensors
        
        Returns:
            Dictionary with batch of token IDs, token type IDs, and attention masks
        """
        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_mask = []
        
        for text in texts:
            encoded = self.tokenizer.encode(
                text,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )
            
            batch_input_ids.append(encoded['input_ids'])
            batch_token_type_ids.append(encoded['token_type_ids'])
            batch_attention_mask.append(encoded['attention_mask'])
        
        output = {
            'input_ids': batch_input_ids,
            'token_type_ids': batch_token_type_ids,
            'attention_mask': batch_attention_mask
        }
        
        if return_tensors:
            import torch
            output = {
                'input_ids': torch.tensor(batch_input_ids, dtype=torch.long),
                'token_type_ids': torch.tensor(batch_token_type_ids, dtype=torch.long),
                'attention_mask': torch.tensor(batch_attention_mask, dtype=torch.float32)
            }
        
        return output
