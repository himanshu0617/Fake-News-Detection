import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np

# ============================================================================
# CUDA Attention Module (PyTorch wrapper for custom kernels)
# ============================================================================

class CUDAScaledDotProductAttention(nn.Module):
    """
    Custom CUDA-accelerated scaled dot-product attention.
    Implements: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """
    
    def __init__(self, d_k: int, use_cuda: bool = True):
        super().__init__()
        self.d_k = d_k
        self.scale = 1.0 / math.sqrt(d_k)
        self.use_cuda = use_cuda and torch.cuda.is_available()
    
    def forward(
        self,
        Q: torch.Tensor,  # [batch_size, num_heads, seq_len, d_k]
        K: torch.Tensor,  # [batch_size, num_heads, seq_len, d_k]
        V: torch.Tensor,  # [batch_size, num_heads, seq_len, d_v]
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q, K, V: Query, Key, Value tensors
            mask: Optional attention mask for masking positions
        
        Returns:
            output: Attention output
            attention_weights: Softmax attention weights
        """
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided (for causal masking in decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Apply dropout (optional)
        attention_weights = F.dropout(attention_weights, p=0.1, training=self.training)
        
        # Multiply by values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


# ============================================================================
# Multi-Head Attention
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = CUDAScaledDotProductAttention(self.d_k)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = Q.shape[0]
        
        # Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Reshape for multi-head: [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads: [batch_size, num_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        output = self.dropout(output)
        
        return output, attn_weights


# ============================================================================
# Feed-Forward Network
# ============================================================================

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


# ============================================================================
# Transformer Encoder Layer
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with multi-head attention and feed-forward."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Multi-head attention with residual connection and layer normalization
        attn_output, _ = self.mha(x, x, x, mask)
        x = self.ln1(x + attn_output)
        
        # Feed-forward with residual connection and layer normalization
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        
        return x


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding using sine and cosine functions."""
    
    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1], :]


# ============================================================================
# BERT Transformer Model
# ============================================================================

class BERTTransformer(nn.Module):
    """
    BERT-like transformer model built from scratch in PyTorch.
    Implements bidirectional encoder representations from transformers.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Segment embedding (for distinguishing between sentences A and B)
        self.segment_embedding = nn.Embedding(2, d_model)
        
        # Embedding layer normalization and dropout
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_ln = nn.LayerNorm(d_model)
        
        # Transformer encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head (pooler)
        self.pooler_dense = nn.Linear(d_model, d_model)
        self.pooler_activation = nn.Tanh()
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.constant_(module.weight[module.padding_idx], 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len] - Token IDs
            segment_ids: [batch_size, seq_len] - Segment IDs (0 or 1)
            attention_mask: [batch_size, seq_len] - Attention mask (1 for real tokens, 0 for padding)
        
        Returns:
            logits: [batch_size, num_classes] - Classification logits
            pooled_output: [batch_size, d_model] - Pooled representation
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize segment IDs if not provided
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)
        
        # Create attention mask for padding tokens if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).float()
        
        # Token embedding + positional encoding + segment embedding
        token_embeddings = self.token_embedding(input_ids)
        segment_embeddings = self.segment_embedding(segment_ids)
        embeddings = token_embeddings + segment_embeddings
        embeddings = self.positional_encoding(embeddings)
        embeddings = self.embedding_ln(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Convert attention mask to proper format for transformer
        # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Pass through transformer encoder layers
        hidden_states = embeddings
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, attention_mask)
        
        # Pooling: take [CLS] token representation
        pooled_output = hidden_states[:, 0]  # [batch_size, d_model]
        pooled_output = self.pooler_dense(pooled_output)
        pooled_output = self.pooler_activation(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        return logits, pooled_output


# ============================================================================
# Model Loading and Saving Utilities
# ============================================================================

def create_bert_model(
    vocab_size: int,
    d_model: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    num_classes: int = 2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> BERTTransformer:
    """Create and return a BERT model."""
    model = BERTTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes
    )
    return model.to(device)


def save_model(model: BERTTransformer, path: str):
    """Save model weights and config."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'num_layers': len(model.encoder_layers),
            'num_heads': model.encoder_layers[0].mha.num_heads if model.encoder_layers else 4,
            'd_ff': model.encoder_layers[0].ffn.linear1.out_features if model.encoder_layers else 1024,
            'num_classes': model.num_classes,
            'max_seq_length': model.max_seq_length
        }
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> BERTTransformer:
    """Load model weights and config."""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    
    model = BERTTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 4),
        d_ff=config.get('d_ff', 1024),
        num_classes=config['num_classes'],
        max_seq_length=config['max_seq_length']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)
