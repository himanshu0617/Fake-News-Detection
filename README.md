# Fake News Detection - Custom BERT Transformer with CUDA Kernels

## Project Overview

This project implements a **state-of-the-art fake news detection system** using a custom BERT transformer architecture built entirely from scratch in PyTorch. Instead of relying on pre-trained models or scikit-learn's TF-IDF approach, this implementation features:

1. **Custom BERT Transformer** - Full implementation from scratch with:
   - Multi-head self-attention mechanism
   - Positional encoding using sine/cosine functions
   - Transformer encoder layers with feed-forward networks
   - Classification head for binary fake/real news classification

2. **Custom CUDA Kernels** - Optimized attention mechanism with:
   - Scaled dot-product attention kernels
   - Softmax normalization in CUDA
   - Forward and backward pass implementations
   - Efficient memory management on GPU

3. **Custom BERT Tokenizer** - WordPiece tokenization with:
   - Basic tokenization (lowercasing, punctuation handling)
   - Subword tokenization for unknown words
   - Vocabulary management (30K+ tokens)
   - Batch processing utilities

## Project Structure

```
├── bert_model.py              # BERT transformer implementation from scratch
├── tokenizer.py               # Custom BERT tokenizer with WordPiece
├── trainer.py                 # Training pipeline, data loading, evaluation
├── fake_news_detection.py     # Main application with CLI interface
├── attention_kernel.cu        # CUDA kernels for attention mechanism
├── requirements.txt           # Python dependencies
├── Fake.csv                   # Fake news dataset
├── True.csv                   # Real news dataset
└── README.md                  # This file
```

## Architecture Details

### BERT Model (`bert_model.py`)

The model consists of:

1. **Embedding Layer**
   - Token embeddings (vocab_size × d_model)
   - Positional encodings (sine/cosine)
   - Segment embeddings (distinguishing sentences)
   - Layer normalization and dropout

2. **Transformer Encoder Layers** (12 layers)
   - **Multi-Head Attention** (12 heads, d_k=64)
     - Query, Key, Value projections
     - Scaled dot-product attention: `Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V`
     - Dropout and residual connections
   
   - **Feed-Forward Networks**
     - Linear layer (d_model → d_ff)
     - GELU activation
     - Linear layer (d_ff → d_model)
     - Dropout and residual connections

3. **Classification Head**
   - Pooling on [CLS] token
   - Dense layer with tanh activation
   - Dropout layers
   - Output softmax for binary classification

### CUDA Kernels (`attention_kernel.cu`)

Optimized GPU kernels for:
- **Scaled Dot-Product Attention Forward Pass**
  - Computes Q @ K^T / √d_k efficiently
  - Memory-coalesced access patterns
  
- **Softmax Normalization**
  - Numerically stable implementation (max subtraction)
  - Efficient parallel reduction
  
- **Output Computation**
  - Attention scores × Values
  - Atomic operations for thread-safe accumulation
  
- **Backward Pass**
  - Gradient computation for all inputs
  - Support for efficient backpropagation

### Tokenizer (`tokenizer.py`)

Two-stage tokenization:
1. **Basic Tokenization**
   - Lowercase conversion
   - Punctuation splitting (spaces added around special chars)
   - Whitespace tokenization

2. **WordPiece Tokenization**
   - Greedy longest-match-first algorithm
   - Subword prefix notation (`##` for continuations)
   - Unknown token handling
   - Maximum input length enforcement

Special tokens:
- `[PAD]` (0): Padding token
- `[CLS]` (101): Classification token (sequence start)
- `[SEP]` (102): Separator token
- `[UNK]` (100): Unknown token
- `[MASK]` (103): Mask token (for pretraining)

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ GPU memory recommended (can run on CPU with reduced batch size)

### Setup

1. **Clone/Download the project**
```bash
cd "Brainwave aiml"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**
Download `Fake.csv` and `True.csv` from:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place them in the project directory.

## Usage

### Training

```bash
# Basic training (default: 3 epochs, batch size 16, lr 2e-5)
python fake_news_detection.py train

# Training with custom parameters
python fake_news_detection.py train --epochs 5 --batch-size 32 --lr 1e-5
```

**Training Output:**
- Trained model saved to `fake_news_bert_model.pt`
- Training history plots saved to `training_history.png`
- Confusion matrix saved to `confusion_matrix.png`
- Test set metrics printed to console

### Inference

```bash
# Test on sample articles
python fake_news_detection.py infer

# Interactive mode (enter articles one by one)
python fake_news_detection.py interactive
```

**Interactive Mode:**
```
Enter news article text: NASA discovers water on Mars
Result: REAL NEWS
  Fake probability: 15.23%
  Real probability: 84.77%

Enter news article text: quit
```

## Model Configuration

Default configuration in `fake_news_detection.py`:

```python
VOCAB_SIZE = 30522          # Vocabulary size
D_MODEL = 768               # Model dimension
NUM_LAYERS = 12             # Number of transformer layers
NUM_HEADS = 12              # Number of attention heads
D_FF = 3072                 # Feed-forward hidden dimension
MAX_SEQ_LENGTH = 512        # Maximum sequence length
NUM_CLASSES = 2             # Binary classification
BATCH_SIZE = 16             # Training batch size
LEARNING_RATE = 2e-5        # Adam learning rate
NUM_EPOCHS = 3              # Number of training epochs
```

## Performance

### Model Size
- **Total Parameters**: ~109 million
- **Model File Size**: ~417 MB
- **Training Memory**: ~6-8 GB (with batch size 16)

### Expected Results
- **Test Accuracy**: 85-95% (depending on data quality)
- **Training Time**: ~2-4 hours on single GPU
- **Inference Time**: ~50-100ms per article

## Technical Highlights

### 1. No HuggingFace AutoModel
- Entire BERT architecture implemented from scratch
- No reliance on pre-trained weights
- Full control over model internals

### 2. Custom CUDA Kernels
- Optimized attention computation
- GPU memory-efficient implementations
- Backward pass support for training

### 3. From Scratch Implementation
- Custom tokenizer (no external tokenizers)
- No scikit-learn (replaced with PyTorch)
- No TF-IDF (replaced with learned embeddings)
- Random Forest replaced with Deep Learning

### 4. Production-Ready Features
- Gradient clipping for stability
- Learning rate scheduling (cosine annealing)
- Early stopping mechanism
- Comprehensive metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Training history plots

## Data Statistics

**Fake News Dataset (Kaggle)**:
- ~23,000 fake news articles
- ~21,000 real news articles
- **Total**: ~44,000 articles
- **Features**: Title, text, subject, date

**Train/Val/Test Split**:
- Training: 70% (~30,800 samples)
- Validation: 10% (~4,400 samples)
- Testing: 20% (~8,800 samples)

## Comparison with Original Implementation

| Aspect | Original | Upgraded |
|--------|----------|----------|
| ML Framework | Scikit-learn | PyTorch |
| Feature Extraction | TF-IDF (650 dims) | BERT Embeddings (768 dims) |
| Model Type | Random Forest | Transformer (12 layers) |
| GPU Acceleration | No | Yes (CUDA kernels) |
| Model Parameters | ~150K | ~109M |
| Training Time | ~5 minutes | ~2-4 hours |
| Inference Speed | ~10ms | ~50-100ms |
| Expected Accuracy | ~90% | ~85-95% |
| Model Interpretability | Feature importance | Attention weights |

## Advanced Features

### 1. Attention Visualization
Examine attention weights to understand which parts of the text the model focuses on:
```python
logits, pooled_output = model(input_ids, token_type_ids, attention_mask)
# Attention weights available in intermediate layers
```

### 2. Fine-tuning Strategies
Recommended for domain-specific improvements:
- Reduce learning rate to 1e-5 or lower
- Freeze early layers, train only later layers
- Use task-specific data for fine-tuning

### 3. Mixed Precision Training
For faster training with less memory:
```python
# Enable in trainer.py for potential 2x speedup
from torch.cuda.amp import autocast, GradScaler
```

### 4. Distributed Training
Scale to multiple GPUs:
```python
model = nn.DataParallel(model)
# or use DistributedDataParallel for multi-node
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` (try 8 or 4)
- Use `device='cpu'` for testing
- Enable gradient checkpointing in trainer

### Slow Training
- Ensure CUDA is being used: Check GPU utilization with `nvidia-smi`
- Compile CUDA kernels with `-O3` optimization
- Use `num_workers > 0` in DataLoader for data preprocessing

### Poor Accuracy
- Increase training epochs
- Adjust learning rate
- Ensure data files are correctly loaded
- Check for class imbalance in data

## References

1. **Original BERT Paper**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)
2. **Transformer Architecture**: "Attention is All You Need" (Vaswani et al., 2017)
3. **CUDA Programming**: NVIDIA CUDA C Programming Guide
4. **Dataset**: Kaggle Fake and Real News Dataset

## Future Improvements

1. **Multi-task Learning**: Joint classification and claim verification
2. **Explainability**: LIME/SHAP integration for model interpretability
3. **Active Learning**: Uncertainty sampling for efficient labeling
4. **Ensemble Methods**: Combine with other detectors
5. **Multilingual Support**: Extend to other languages
6. **Real-time Pipeline**: API deployment with FastAPI
7. **Adversarial Robustness**: Defense against adversarial examples

## License

This project is provided for educational and research purposes.

## Contact

For questions or issues, please open an issue on the project repository.

---

**Project Status**: Complete and production-ready
**Last Updated**: January 2026
**Python Version**: 3.8+
**PyTorch Version**: 2.0+
