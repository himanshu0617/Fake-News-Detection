import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from bert_model import BERTTransformer, create_bert_model, save_model, load_model
from tokenizer import BERTTokenizer, BatchTokenizer


# ============================================================================
# Dataset Class
# ============================================================================

class FakeNewsDataset(Dataset):
    """PyTorch Dataset for fake news detection."""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BERTTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoded = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(encoded['token_type_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# Training Pipeline
# ============================================================================

class FakeNewsTrainer:
    """Trainer class for fake news detection model."""
    
    def __init__(
        self,
        model: BERTTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        warmup_steps: int = 500
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        
        # Loss function (weighted for imbalanced data)
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Move model to device
        self.model.to(device)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            logits, _ = self.model(input_ids, token_type_ids, attention_mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on validation or test set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Evaluating")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                logits, _ = self.model(input_ids, token_type_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train(self) -> Dict:
        """Train the model for multiple epochs."""
        best_val_accuracy = 0.0
        best_model_state = None
        patience = 3
        patience_counter = 0
        
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, val_accuracy = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print("✓ Best model updated!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def predict(self, texts: List[str], batch_size: int = 32) -> Tuple[List[int], List[float]]:
        """Make predictions on new texts."""
        self.model.eval()
        predictions = []
        confidences = []
        
        # Create temporary dataset
        dataset = FakeNewsDataset(
            texts,
            labels=[0] * len(texts),  # Dummy labels
            tokenizer=self.tokenizer,
            max_length=512
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits, _ = self.model(input_ids, token_type_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                
                preds = torch.argmax(logits, dim=1)
                conf = torch.max(probs, dim=1)[0]
                
                predictions.extend(preds.cpu().numpy())
                confidences.extend(conf.cpu().numpy())
        
        return predictions, confidences


# ============================================================================
# Metrics and Visualization
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    save_path: str = 'training_history.png'
):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(val_accuracies, label='Val Accuracy', marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ['Real', 'Fake'],
    save_path: str = 'confusion_matrix.png'
):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix plot saved to {save_path}")
    plt.close()


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_fake_news_data(
    fake_csv: str,
    real_csv: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Load fake news dataset from CSV files.
    
    Returns:
        (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
    """
    # Load data
    fake_df = pd.read_csv(fake_csv)
    real_df = pd.read_csv(real_csv)
    
    # Extract text column (handle different column names)
    fake_texts = fake_df['text'].values if 'text' in fake_df.columns else fake_df.iloc[:, -1].values
    real_texts = real_df['text'].values if 'text' in real_df.columns else real_df.iloc[:, -1].values
    
    # Combine and create labels
    texts = list(fake_texts) + list(real_texts)
    labels = [0] * len(fake_texts) + [1] * len(real_texts)  # 0: Fake, 1: Real
    
    # Shuffle
    np.random.seed(random_state)
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Split: train, val, test
    total = len(texts)
    test_split = int(total * test_size)
    val_split = int((total - test_split) * val_size)
    
    test_texts = texts[:test_split]
    test_labels = labels[:test_split]
    
    remaining_texts = texts[test_split:]
    remaining_labels = labels[test_split:]
    
    val_texts = remaining_texts[:val_split]
    val_labels = remaining_labels[:val_split]
    
    train_texts = remaining_texts[val_split:]
    train_labels = remaining_labels[val_split:]
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Val: {len(val_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels
