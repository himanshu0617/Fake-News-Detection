#!/usr/bin/env python3
"""
Fake News Detection using Custom BERT Transformer with CUDA Kernels
====================================================================

Main application script that:
1. Loads and preprocesses the fake news dataset
2. Creates a BERT model from scratch
3. Trains the model with custom CUDA attention kernels
4. Evaluates on test set with comprehensive metrics
5. Provides inference interface for new articles
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

from bert_model import BERTTransformer, create_bert_model, save_model, load_model
from tokenizer import BERTTokenizer, BatchTokenizer
from trainer import (
    FakeNewsDataset,
    FakeNewsTrainer,
    load_fake_news_data,
    compute_metrics,
    plot_training_history,
    plot_confusion_matrix
)


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration for training and inference."""
    
    # Model config
    VOCAB_SIZE = 30522
    D_MODEL = 256  # Reduced from 768
    NUM_LAYERS = 4  # Reduced from 12
    NUM_HEADS = 4   # Reduced from 12
    D_FF = 1024     # Reduced from 3072 (proportional to D_MODEL)
    MAX_SEQ_LENGTH = 256  # Reduced from 512 (50% faster)
    NUM_CLASSES = 2
    
    # Training config
    BATCH_SIZE = 8   # Reduced from 16 (4x faster per batch)
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 1   # Reduced from 3 (testing only)
    WARMUP_STEPS = 500
    
    # Data paths
    FAKE_CSV = 'Fake.csv'
    REAL_CSV = 'True.csv'
    
    # Output paths
    MODEL_SAVE_PATH = 'fake_news_bert_model.pt'
    TOKENIZER_PATH = 'bert_tokenizer.pkl'
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# Main Functions
# ============================================================================

def main_train():
    """Train the BERT model."""
    
    print("="*80)
    print("FAKE NEWS DETECTION - BERT TRANSFORMER WITH CUDA KERNELS")
    print("="*80)
    print(f"\nDevice: {Config.DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    if not os.path.exists(Config.FAKE_CSV) or not os.path.exists(Config.REAL_CSV):
        print(f"\nError: Data files not found!")
        print(f"  Expected: {Config.FAKE_CSV}, {Config.REAL_CSV}")
        print("\nDownload from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        return
    
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
        load_fake_news_data(Config.FAKE_CSV, Config.REAL_CSV)
    
    # Use subset of data for faster training (5000 samples instead of 32K)
    train_texts = train_texts[:5000]
    train_labels = train_labels[:5000]
    val_texts = val_texts[:500]
    val_labels = val_labels[:500]
    test_texts = test_texts[:1000]
    test_labels = test_labels[:1000]
    
    # ========================================================================
    # Initialize Tokenizer
    # ========================================================================
    
    print("\n" + "="*80)
    print("INITIALIZING TOKENIZER")
    print("="*80)
    
    tokenizer = BERTTokenizer(vocab_size=Config.VOCAB_SIZE)
    print(f"\nTokenizer initialized with vocab size: {tokenizer.get_vocab_size()}")
    
    # Test tokenizer
    sample_text = "This is a test article about fake news detection."
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"\nTokenizer test:")
    print(f"  Text: {sample_text}")
    print(f"  Tokens: {tokens}")
    print(f"  Token IDs: {token_ids[:10]}...")
    
    # ========================================================================
    # Create DataLoaders
    # ========================================================================
    
    print("\n" + "="*80)
    print("CREATING DATALOADERS")
    print("="*80)
    
    train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer, Config.MAX_SEQ_LENGTH)
    val_dataset = FakeNewsDataset(val_texts, val_labels, tokenizer, Config.MAX_SEQ_LENGTH)
    test_dataset = FakeNewsDataset(test_texts, test_labels, tokenizer, Config.MAX_SEQ_LENGTH)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # ========================================================================
    # Create Model
    # ========================================================================
    
    print("\n" + "="*80)
    print("CREATING BERT MODEL")
    print("="*80)
    
    model = create_bert_model(
        vocab_size=Config.VOCAB_SIZE,
        d_model=Config.D_MODEL,
        num_layers=Config.NUM_LAYERS,
        num_heads=Config.NUM_HEADS,
        num_classes=Config.NUM_CLASSES,
        device=Config.DEVICE
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB")
    
    # ========================================================================
    # Train Model
    # ========================================================================
    
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    trainer = FakeNewsTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=Config.DEVICE,
        learning_rate=Config.LEARNING_RATE,
        num_epochs=Config.NUM_EPOCHS,
        warmup_steps=Config.WARMUP_STEPS
    )
    
    # Store tokenizer reference in trainer for inference
    trainer.tokenizer = tokenizer
    
    # Train
    history = trainer.train()
    
    # ========================================================================
    # Evaluate on Test Set
    # ========================================================================
    
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(Config.DEVICE)
            token_type_ids = batch['token_type_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            logits, _ = model(input_ids, token_type_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = compute_metrics(all_labels, all_preds)
    
    print("\nTest Set Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    # ========================================================================
    # Save Model
    # ========================================================================
    
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    save_model(model, Config.MODEL_SAVE_PATH)
    
    # ========================================================================
    # Generate Plots
    # ========================================================================
    
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        history['val_accuracies'],
        'training_history.png'
    )
    
    plot_confusion_matrix(
        all_labels,
        all_preds,
        labels=['Fake', 'Real'],
        save_path='confusion_matrix.png'
    )
    
    print("\nTraining completed successfully!")


def main_infer():
    """Run inference on sample texts."""
    
    print("\n" + "="*80)
    print("FAKE NEWS DETECTION - INFERENCE")
    print("="*80)
    print(f"Device: {Config.DEVICE}\n")
    
    # Load model and tokenizer
    print("Loading model...")
    model = load_model(Config.MODEL_SAVE_PATH, device=Config.DEVICE)
    
    tokenizer = BERTTokenizer(vocab_size=Config.VOCAB_SIZE)
    
    # Sample texts for inference
    sample_texts = [
        "NASA discovers water on Mars, confirming possibility of life.",
        "Fake news alert: Politicians hide alien technology from public.",
        "Federal Reserve announces new interest rate policies.",
        "Breaking: Scientists confirm pigs can fly with special devices.",
    ]
    
    print("Making predictions on sample texts:\n")
    
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(sample_texts, 1):
            # Encode
            encoded = tokenizer.encode(text, max_length=Config.MAX_SEQ_LENGTH)
            
            input_ids = torch.tensor([encoded['input_ids']], dtype=torch.long).to(Config.DEVICE)
            token_type_ids = torch.tensor([encoded['token_type_ids']], dtype=torch.long).to(Config.DEVICE)
            attention_mask = torch.tensor([encoded['attention_mask']], dtype=torch.float32).to(Config.DEVICE)
            
            # Predict
            logits, _ = model(input_ids, token_type_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(logits, dim=1).item()
            confidence = torch.max(probs, dim=1)[0].item()
            
            label_name = "REAL" if pred_label == 1 else "FAKE"
            
            print(f"{i}. {text}")
            print(f"   Prediction: {label_name} (Confidence: {confidence:.2%})\n")


def interactive_inference():
    """Interactive inference mode."""
    
    print("\n" + "="*80)
    print("FAKE NEWS DETECTION - INTERACTIVE MODE")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print("Type 'quit' to exit\n")
    
    # Load model and tokenizer
    print("Loading model...")
    model = load_model(Config.MODEL_SAVE_PATH, device=Config.DEVICE)
    tokenizer = BERTTokenizer(vocab_size=Config.VOCAB_SIZE)
    
    model.eval()
    
    while True:
        text = input("\nEnter news article text: ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        with torch.no_grad():
            encoded = tokenizer.encode(text, max_length=Config.MAX_SEQ_LENGTH)
            
            input_ids = torch.tensor([encoded['input_ids']], dtype=torch.long).to(Config.DEVICE)
            token_type_ids = torch.tensor([encoded['token_type_ids']], dtype=torch.long).to(Config.DEVICE)
            attention_mask = torch.tensor([encoded['attention_mask']], dtype=torch.float32).to(Config.DEVICE)
            
            logits, _ = model(input_ids, token_type_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(logits, dim=1).item()
            fake_prob = probs[0, 0].item()
            real_prob = probs[0, 1].item()
            
            label_name = "REAL NEWS" if pred_label == 1 else "FAKE NEWS"
            
            print(f"\nResult: {label_name}")
            print(f"  Fake probability: {fake_prob:.2%}")
            print(f"  Real probability: {real_prob:.2%}")


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fake News Detection using BERT Transformer with Custom CUDA Kernels"
    )
    parser.add_argument(
        'mode',
        choices=['train', 'infer', 'interactive'],
        help='Mode: train (train model), infer (test on samples), interactive (interactive mode)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=Config.BATCH_SIZE,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=Config.NUM_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=Config.LEARNING_RATE,
        help='Learning rate'
    )
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.lr
    
    if args.mode == 'train':
        main_train()
    elif args.mode == 'infer':
        main_infer()
    elif args.mode == 'interactive':
        interactive_inference()


if __name__ == '__main__':
    # Check for CUDA
    print("\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Python version: {sys.version}")
    
    main()
