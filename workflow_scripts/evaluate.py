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
from darkvision.modeling.predict import load_model, predict, predict_batch
from darkvision.config import REPORTS_DIR


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
    "--test-data",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to test data (.pt file containing X_test and y_test tensors)"
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
    "--batch-mode",
    is_flag=True,
    help="Use batch prediction mode (expects DataLoader instead of tensors)"
)
def evaluate_model(
    model_path: Path,
    test_data: Path,
    output_dir: Path,
    device: str,
    log_level: str,
    batch_mode: bool
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
    logger.info(f"Test data path: {test_data}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {device}")
        
        # Load model
        logger.info("Loading model...")
        model = load_model(model_path, device)
        model_name = model_path.stem
        
        # Load test data
        logger.info("Loading test data...")
        test_data_dict = torch.load(test_data, map_location=device)
        
        if batch_mode:
            # Assume test_data contains a DataLoader
            if 'dataloader' not in test_data_dict:
                raise ValueError("Batch mode requires 'dataloader' key in test data file")
            
            dataloader = test_data_dict['dataloader']
            logger.info("Making batch predictions...")
            predictions, ground_truth = predict_batch(dataloader, model, device)
            
        else:
            # Assume test_data contains X_test and y_test tensors
            if 'X_test' not in test_data_dict or 'y_test' not in test_data_dict:
                raise ValueError("Test data file must contain 'X_test' and 'y_test' keys")
            
            X_test = test_data_dict['X_test']
            y_test = test_data_dict['y_test']
            
            logger.info("Making predictions...")
            predictions, ground_truth = predict(X_test, y_test, model, device)
        
        # Calculate metrics
        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        rmse = np.sqrt(mse)
        
        output_path = save_results(mse, mae, output_dir, model_name)
        
        logger.success("Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    evaluate_model()