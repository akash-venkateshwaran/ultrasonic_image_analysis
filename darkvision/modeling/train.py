from pathlib import Path
import json
import subprocess
import threading
import time
import socket

from loguru import logger
from tqdm import tqdm
import typer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from darkvision.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, RANDOM_STATE, TENSORBOARD_PORT
from darkvision.modeling.model import SimpleFlawCNN
from darkvision.dataset import get_dataloader

app = typer.Typer()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def check_port_available(port: int):
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except socket.error:
            return False


def start_tensorboard(log_dir: Path, port: int = TENSORBOARD_PORT):
    """Start TensorBoard server in a separate thread."""
    
    # Check if port is available
    if not check_port_available(port):
        logger.warning(f"Port {port} is already in use, TensorBoard may not start properly")
    
    def run_tensorboard():
        try:
            subprocess.run([
                'tensorboard', '--logdir', str(log_dir), 
                '--port', str(port)], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"TensorBoard failed to start: {e}")
    
    # Start TensorBoard in background thread
    tb_thread = threading.Thread(target=run_tensorboard, daemon=True)
    tb_thread.start()
    
    # Wait a moment for TensorBoard to start
    time.sleep(2)
    
    return port


def setup_tensorboard(experiment_name: str = None):
    """Setup TensorBoard logging."""
    if experiment_name is None:
        experiment_name = f"experiment_{int(time.time())}"
    
    log_dir = REPORTS_DIR / "logs" / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    # Start TensorBoard server
    port = start_tensorboard(REPORTS_DIR / "logs")
    logger.info(f"TensorBoard Dashboard available at: http://localhost:{port}")
    
    return writer, log_dir


def log_model_graph(writer, model, sample_input):
    """Log model architecture to TensorBoard."""
    try:
        writer.add_graph(model, sample_input)
        logger.info("Model architecture logged to TensorBoard")
    except Exception as e:
        logger.warning(f"Failed to log model graph: {e}")


def log_sample_images(writer, dataloader, num_samples=8):
    """Log sample images from dataset to TensorBoard."""
    try:
        # Get a batch of images
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        
        # Create a grid of images
        img_grid = vutils.make_grid(images[:num_samples], normalize=True, nrow=4)
        
        # Log to TensorBoard
        writer.add_image('Sample_Images', img_grid, 0)
        logger.info(f"Sample images logged to TensorBoard")
    except Exception as e:
        logger.warning(f"Failed to log sample images: {e}")


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
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = RANDOM_STATE,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    patience: int = 15,
    min_delta: float = 1e-4,
    save_every: int = 25,
    print_every: int = 10,
    exp_name: str = None,
    log_every: int = 100,  # Log to TensorBoard every N batches
):
    """
    Train the CNN model for flaw size regression with TensorBoard logging.
    """
    set_seed(seed)
    
    # Setup TensorBoard
    writer, log_dir = setup_tensorboard(exp_name)
    
    # Data loaders
    train_loader = get_dataloader(train_dir, batch_size=batch_size, shuffle=True)
    test_loader = get_dataloader(test_dir, batch_size=batch_size, shuffle=False)
    
    # Log sample images
    log_sample_images(writer, train_loader)
    
    # Model setup
    model = SimpleFlawCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    # Log model architecture
    sample_batch = next(iter(train_loader))
    sample_input = sample_batch[0][:1].to(device)
    log_model_graph(writer, model, sample_input)
    
    # Training tracking
    best_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    learning_rates = []
    global_step = 0
    
    logger.info(f"Starting training for {epochs} epochs on {device}")
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(test_loader.dataset)}")
    logger.info(f"Logs saved to: {log_dir}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_count = 0
        running_loss = 0.0
        
        for batch_idx, (X, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y.squeeze())
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss * X.size(0)
            train_count += X.size(0)
            running_loss += batch_loss
            global_step += 1
            
            # Log to TensorBoard every log_every batches
            if batch_idx % log_every == 0 and batch_idx > 0:
                avg_loss = running_loss / log_every
                writer.add_scalar('Loss/Train_Running', avg_loss, global_step)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                running_loss = 0.0
        
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

        # Log epoch metrics to TensorBoard
        if (epoch + 1) % print_every == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:3d}/{epochs}: Train Loss: {train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")
            writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            writer.add_scalar('Loss/Validation_Epoch', val_loss, epoch)
            
            # Log histograms of model parameters
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'Parameters/{name}', param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        
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
            
            # Log best model metrics
            writer.add_scalar('Best_Loss/Validation', best_loss, epoch)
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
    
    # Log final metrics
    
    writer.add_hparams(
            {
                'lr': lr,
                'batch_size': batch_size,
                'epochs': len(train_losses),
                'patience': patience,
            },
            {
                'best_val_loss': best_loss,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1],
            }
        )
    
    # Save training history
    history_path = REPORTS_DIR / "logs" / exp_name / "training_history.json"
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_loss': float(best_loss),
        'total_epochs': len(train_losses),
        'tensorboard_log_dir': str(log_dir),
    }
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training history saved to {history_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    return {
        'best_loss': best_loss,
        'total_epochs': len(train_losses),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'tensorboard_log_dir': log_dir,
    }


if __name__ == "__main__":
    app()