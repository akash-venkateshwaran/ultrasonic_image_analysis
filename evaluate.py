
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from darkvision.dataset import FlawDataset
from darkvision.modeling.model import FlawDetector
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model_path: str, data_dir: str, output_dir: str):
    """
    Evaluates a model and saves the metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlawDetector.load_from_checkpoint(model_path).to(device)
    model.eval()

    dataset = FlawDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for i, (image, target) in enumerate(dataloader):
            image = image.to(device)
            pred = model(image)
            predictions.append(pred.cpu().numpy())
            ground_truth.append(target.numpy())
            if i >= 100: # Limit evaluation to 100 samples for speed
                break

    predictions = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)

    mse = mean_squared_error(ground_truth, predictions)
    mae = mean_absolute_error(ground_truth, predictions)

    model_name = Path(model_path).stem
    output_path = Path(output_dir) / f"{model_name}_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({"mse": mse, "mae": mae}, f, indent=4)

    print(f"Evaluated {model_name}: MSE={mse:.4f}, MAE={mae:.4f}")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to the processed data directory.")
    parser.add_argument("--output_dir", type=str, default="reports", help="Path to the output directory for metrics.")
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.output_dir)
