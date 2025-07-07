"""
Prediction utilities for loading models and making predictions.
"""

import torch
from pathlib import Path
from typing import Union, Tuple
import numpy as np
from loguru import logger

from darkvision.modeling.model import SimpleFlawCNN


def load_model(
    model_path: Union[str, Path],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> SimpleFlawCNN:
    """
    Load a trained SimpleFlawCNN model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint (.pt file)
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Loaded SimpleFlawCNN model ready for inference
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Initialize model
        model = SimpleFlawCNN().to(device)
        
        # Load checkpoint
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Log additional checkpoint info if available
        if 'epoch' in checkpoint:
            logger.info(f"Model trained for {checkpoint['epoch']} epochs")
        if 'best_loss' in checkpoint:
            logger.info(f"Best validation loss: {checkpoint['best_loss']:.6f}")
            
        logger.success("Model loaded successfully")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def predict(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model: SimpleFlawCNN,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on test data using a loaded model.
    
    Args:
        X_test: Test input tensor of shape (N, C, H, W)
        y_test: Test target tensor of shape (N, 1) or (N,)
        model: Loaded SimpleFlawCNN model
        device: Device to run predictions on
        
    Returns:
        Tuple of (predictions, ground_truth) as numpy arrays
        
    Raises:
        RuntimeError: If prediction fails
    """
    try:
        # Ensure model is in eval mode
        model.eval()
        
        # Move data to device
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        
        logger.info(f"Making predictions on {X_test.shape[0]} samples")
        
        # Make predictions
        with torch.no_grad():
            y_pred = model(X_test)
            
        # Convert to numpy arrays
        predictions = y_pred.cpu().numpy().flatten()
        ground_truth = y_test.cpu().numpy().flatten()
        
        logger.info(f"Predictions completed. Shape: {predictions.shape}")
        
        return predictions, ground_truth
        
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")


def predict_batch(
    dataloader: torch.utils.data.DataLoader,
    model: SimpleFlawCNN,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on a DataLoader using a loaded model.
    
    Args:
        dataloader: DataLoader containing test data
        model: Loaded SimpleFlawCNN model
        device: Device to run predictions on
        
    Returns:
        Tuple of (predictions, ground_truth) as numpy arrays
        
    Raises:
        RuntimeError: If prediction fails
    """
    try:
        # Ensure model is in eval mode
        model.eval()
        
        all_predictions = []
        all_ground_truth = []
        
        logger.info(f"Making batch predictions on {len(dataloader)} batches")
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Make predictions
                y_pred = model(X_batch)
                
                # Collect results
                all_predictions.append(y_pred.cpu().numpy().flatten())
                all_ground_truth.append(y_batch.cpu().numpy().flatten())
        
        # Concatenate all batches
        predictions = np.concatenate(all_predictions)
        ground_truth = np.concatenate(all_ground_truth)
        
        logger.info(f"Batch predictions completed. Total samples: {len(predictions)}")
        
        return predictions, ground_truth
        
    except Exception as e:
        raise RuntimeError(f"Batch prediction failed: {str(e)}")