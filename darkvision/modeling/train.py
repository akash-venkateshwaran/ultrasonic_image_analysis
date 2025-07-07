from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from darkvision.config import MODELS_DIR, PROCESSED_DATA_DIR, RANDOM_STATE
from darkvision.modeling.model import SimpleFlawCNN
from darkvision.dataset import get_dataloader

app = typer.Typer()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Training some model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Modeling training complete.")
    # -----------------------------------------



@app.command()
def train(
    train_dir: Path = PROCESSED_DATA_DIR / "train",
    test_dir: Path = PROCESSED_DATA_DIR / "val",
    model_path: Path = MODELS_DIR / "cnn_flaw.pt",
    batch_size: int = 16,
    epochs: int = 100,  # Increased to 100
    lr: float = 1e-3,
    seed: int = RANDOM_STATE,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 15,  # Early stopping patience
    min_delta: float = 1e-4,  # Minimum change to qualify as improvement
    save_every: int = 25,  # Save checkpoint every N epochs
    print_every: int = 10,  # Print every 10th epoch
):
    """
    Train the CNN model for flaw size regression with enhanced features.
    """
    set_seed(seed)
    
    # Data loaders
    train_loader = get_dataloader(train_dir, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(test_dir, batch_size=batch_size, shuffle=False)
    
    # Model setup
    model = SimpleFlawCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    # Training tracking
    best_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    learning_rates = []
    
    logger.info(f"Starting training for {epochs} epochs on {device}")
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(test_loader.dataset)}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_count = 0
        
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            train_count += X.size(0)
        
        train_loss /= train_count
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_count = 0
        
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss = criterion(preds, y.squeeze())
                val_loss += loss.item() * X.size(0)
                val_count += X.size(0)
        
        val_loss /= val_count
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Print progress every 10th epoch
        if (epoch + 1) % print_every == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{epochs}: Train Loss: {train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
            }, model_path)
            logger.success(f"New best model saved! Val Loss: {val_loss:.6f}")
        else:
            epochs_without_improvement += 1
        
        # Save periodic checkpoints
        if (epoch + 1) % save_every == 0:
            checkpoint_path = model_path.parent / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs. "
                       f"No improvement for {patience} epochs.")
            break
    
    # Training completion
    logger.success(f"Training completed! Best validation loss: {best_loss:.6f}")
    
    # Save training history
    history_path = model_path.parent / "training_history.json"
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_loss': float(best_loss),
        'total_epochs': len(train_losses),
    }
    
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training history saved to {history_path}")
    
    return {
        'best_loss': best_loss,
        'total_epochs': len(train_losses),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }


if __name__ == "__main__":
    app()
