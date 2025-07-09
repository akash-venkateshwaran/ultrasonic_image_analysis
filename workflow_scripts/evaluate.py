#!/usr/bin/env python3
"""
Model evaluation CLI script for calculating metrics and generating reports.
"""

import click
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from loguru import logger
import sys

# Import your prediction utilities
from darkvision.modeling.predict import load_model, predict_batch
from darkvision.dataset import get_dataloader
from darkvision.config import REPORTS_DIR, PROCESSED_DATA_DIR


def setup_logging(log_level: str = "INFO"):
    """Setup loguru logging configuration."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level
    )


def save_results(mse: float, mae: float, output_dir: Path, model_name: str):
    """Save evaluation results to text file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{model_name}_{timestamp}.txt"
    output_path = output_dir / filename
    
    results_text = f"""Model Evaluation Results
========================
Model: {model_name}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Metrics:
--------
Mean Squared Error (MSE): {mse:.6f}
Mean Absolute Error (MAE): {mae:.6f}

Additional Statistics:
---------------------
RMSE: {np.sqrt(mse):.6f}
"""
    
    with open(output_path, 'w') as f:
        f.write(results_text)
    
    logger.success(f"Results saved to: {output_path}")
    return output_path


@click.command()
@click.option(
    "--model-path",
    "-m",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the trained model checkpoint (.pt file)"
)
@click.option(
    "--test-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    default=PROCESSED_DATA_DIR / "test",
    help=f"Path to test data directory (default: {PROCESSED_DATA_DIR / 'test'})"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=REPORTS_DIR,
    help=f"Output directory for results (default: {REPORTS_DIR})"
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "auto"]),
    default="auto",
    help="Device to use for inference (default: auto)"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="Logging level (default: INFO)"
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for evaluation (default: None)"
)
def evaluate_model(
    model_path: Path,
    test_dir: Path,
    output_dir: Path,
    device: str,
    log_level: str,
    batch_size: int
):
        """
        Evaluate a trained model on test data and generate metrics report.
        
        This script loads a trained model, makes predictions on test data,
        calculates MSE and MAE metrics, and saves results to a text file.
        """
        # Setup logging
        setup_logging(log_level)
        
        logger.info("Starting model evaluation")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Test data directory: {test_dir}")
        logger.info(f"Output directory: {output_dir}")
    

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {device}")
        
        # Load model
        logger.info("Loading model...")
        model = load_model(model_path, device)
        model_name = model_path.stem
        
        # Get test dataloader
        logger.info("Creating test dataloader...")
        test_loader = get_dataloader(test_dir, batch_size=batch_size, shuffle=False)
        
        # Make batch predictions
        logger.info("Making batch predictions...")
        predictions, ground_truth = predict_batch(test_loader, model, device)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        
        # Log metrics
        logger.info(f"Mean Squared Error (MSE): {mse:.6f}")
        logger.info(f"Mean Absolute Error (MAE): {mae:.6f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
        
        # Save results
        output_path = save_results(mse, mae, output_dir, model_name)
        
        logger.success("Model evaluation completed successfully!")


if __name__ == "__main__":
    evaluate_model()