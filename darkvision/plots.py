from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from darkvision.dataset import DataProcessorRaw
import numpy as np

from loguru import logger
from tqdm import tqdm
import click
import random
import torch
from darkvision.config import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, DEFAULT_FILENAME, MODELS_DIR, RANDOM_STATE
from darkvision.modeling.predict import load_model, predict_batch
from darkvision.dataset import get_dataloader
from typing import Optional


def plot_slice_common(slice_info, filename, slice_num=None, slice_idx=None, 
                     figsize=(8, 6), show_title=True, show_axes=False, 
                     save_path=None, show_plot=False, dpi=150):
    """
    Common function for plotting a single slice with defect information.
    
    Args:
        slice_info: Dictionary containing slice information from processor
        filename: Base filename for display
        slice_num: 1-based slice number for display (optional)
        slice_idx: 0-based slice index for display (optional)
        figsize: Figure size tuple
        show_title: Whether to show title
        show_axes: Whether to show axes
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        dpi: DPI for saving
        
    Returns:
        tuple: (fig, ax, im) - matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the image
    im = ax.imshow(slice_info['image'], cmap="gray")
    
    # Configure title if requested
    if show_title:
        display_slice = slice_num if slice_num is not None else (slice_idx if slice_idx is not None else "Unknown")
        defect_status = "DEFECT DETECTED" if slice_info['has_defect'] else "NO DEFECT"
        title = f"{filename} - Slice {display_slice}\n{defect_status} | Flaw Size: {slice_info['equivalent_flawsize']:.3f}"
        ax.set_title(title, fontsize=12, pad=20)
    
    # Configure axes
    if not show_axes:
        ax.axis("off")
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig, ax, im


def print_slice_info(slice_info, filename, slice_num=None, slice_idx=None, save_path=None):
    """
    Common function for printing slice information.
    
    Args:
        slice_info: Dictionary containing slice information
        filename: Base filename
        slice_num: 1-based slice number (optional)
        slice_idx: 0-based slice index (optional)
        save_path: Save path to display (optional)
    """
    display_slice = slice_num if slice_num is not None else slice_idx
    print(f"\n=== Slice {display_slice} Information ===")
    print(f"Filename: {filename}")
    if slice_idx is not None:
        print(f"Slice Index: {slice_idx} (0-based)")
    print(f"Has Defect: {slice_info['has_defect']}")
    print(f"Equivalent Flaw Size: {slice_info['equivalent_flawsize']}")
    print(f"Label: {slice_info['label']}")
    if save_path:
        print(f"Plot saved to: {save_path}")


def create_animation_figure():
    """
    Create and configure the figure and axes for animation.
    
    Returns:
        tuple: (fig, ax, im, title_text, defect_text, flaw_text)
    """
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create placeholder image (will be updated in animation)
    placeholder = np.zeros((100, 100))  # Temporary placeholder
    im = ax.imshow(placeholder, cmap='gray', animated=True, aspect='auto')
    
    # Remove all margins and padding to fill entire figure
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])  # Fill entire figure
    ax.axis('off')  # Remove axes for cleaner look
    
    # Set up text annotations with adjusted positions
    title_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        fontsize=16, fontweight='bold', 
                        verticalalignment='top', color='white',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    defect_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, 
                         fontsize=14, fontweight='bold', verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    flaw_text = ax.text(0.02, 0.82, '', transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top', color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    return fig, ax, im, title_text, defect_text, flaw_text


@click.group()
def cli():
    """Data visualization tool for darkvision dataset."""
    pass


@cli.command()
@click.option('--filename', '-f', default=DEFAULT_FILENAME, 
              help=f'Base filename (without extension). Default: {DEFAULT_FILENAME}')
@click.option('--slice-idx', '-s', default=4, 
              help='Slice index to inspect. Default: 4')
@click.option('--output-path', '-o', 
              type=click.Path(path_type=Path), 
              default=FIGURES_DIR / "plot.png",
              help='Output path for the plot')
def inspect(filename, slice_idx, output_path):
    """
    Inspect a .bins file, plot a slice, and display info from .jsons and .labels.
    """
    processor = DataProcessorRaw(filename)
    slice_info = processor.get_slice_info(slice_idx)
    
    # Print basic array information
    array_data = processor.load_bin()
    print(f"Array shape: {array_data.shape}, dtype: {array_data.dtype}")
    print(f"Slice info keys: {slice_info.keys()}")
    print(f"Has defect: {slice_info['has_defect']}")
    print(f"Equivalent flaw size: {slice_info['equivalent_flawsize']}")

    # Use common plotting function
    fig, ax, im = plot_slice_common(
        slice_info=slice_info,
        filename=filename,
        slice_idx=slice_idx,
        figsize=(6, 6),
        save_path=output_path,
        show_plot=True
    )
    
    # Print detailed information using common function
    print_slice_info(slice_info, filename, slice_idx=slice_idx, save_path=output_path)
    
    # Print additional detailed info
    print(f"JSON Data: {slice_info['json_data']}")


@cli.command()
@click.option('--filename', '-f', default=DEFAULT_FILENAME,
              help=f'Base filename (without extension). Default: {DEFAULT_FILENAME}')
@click.option('--slice-num', '-s', type=click.IntRange(1, 100), default=1,
              help='Slice number to plot (1-100). Default: 1')
@click.option('--output-path', '-o', 
              type=click.Path(path_type=Path), 
              default=FIGURES_DIR / "single_slice.png",
              help='Output path for the plot')
def plot_slice(filename, slice_num, output_path):
    """
    Plot a single slice from a file.
    
    Args:
        filename: Base filename (without extension)
        slice_num: Slice number (1-100, converted to 0-based index)
        output_path: Path to save the plot
    """
    # Convert 1-based slice number to 0-based index
    slice_idx = slice_num - 1
    
    processor = DataProcessorRaw(filename)
    slice_info = processor.get_slice_info(slice_idx)
    
    # Use common plotting function
    fig, ax, im = plot_slice_common(
        slice_info=slice_info,
        filename=filename,
        slice_num=slice_num,
        slice_idx=slice_idx,
        save_path=output_path,
        show_plot=True
    )
    
    # Print information using common function
    print_slice_info(slice_info, filename, slice_num=slice_num, 
                    slice_idx=slice_idx, save_path=output_path)


@cli.command()
@click.option('--filename', '-f', default=DEFAULT_FILENAME,
              help=f'Base filename (without extension). Default: {DEFAULT_FILENAME}')
@click.option('--output-path', '-o', 
              type=click.Path(path_type=Path), 
              default=FIGURES_DIR / "animation.gif",
              help='Output path for the animation')
@click.option('--interval', '-i', default=200,
              help='Time between frames in milliseconds. Default: 200')
@click.option('--start-slice', default=0,
              help='Starting slice index. Default: 0')
@click.option('--end-slice', default=None, type=int,
              help='Ending slice index (None for all slices)')
def animate(filename, output_path, interval, start_slice, end_slice):
    """
    Create an animation cycling through slices with defect information.
    
    Args:
        filename: Base filename (without extension)
        output_path: Path to save the animation
        interval: Time between frames in milliseconds
        start_slice: Starting slice index
        end_slice: Ending slice index (None for all slices)
    """
    # Initialize data processor
    processor = DataProcessorRaw(filename)
    slice_range = processor.get_slice_range(start_slice, end_slice)
    
    logger.info(f"Creating animation for slices {start_slice} to {max(slice_range)}")

    # Create figure using the separate function
    fig, ax, im, title_text, defect_text, flaw_text = create_animation_figure()
    
    # Get first slice info to initialize properly
    first_slice_info = processor.get_slice_info(start_slice)
    
    # Update image properties with actual data and set proper extent
    im.set_array(first_slice_info['image'])
    im.set_extent([0, first_slice_info['image'].shape[1], 
                   first_slice_info['image'].shape[0], 0])
    
    # Set axis limits to match image dimensions exactly
    ax.set_xlim(0, first_slice_info['image'].shape[1])
    ax.set_ylim(first_slice_info['image'].shape[0], 0)

    def animate_frame(frame_idx):
        slice_idx = slice_range[frame_idx]
        
        # Get slice information from processor
        slice_info = processor.get_slice_info(slice_idx)
        
        # Update image
        im.set_array(slice_info['image'])
        
        # Update text
        title_text.set_text(f'Slice: {slice_idx}')
        
        defect_status = "DEFECT DETECTED" if slice_info['has_defect'] else "NO DEFECT"
        defect_color = 'red' if slice_info['has_defect'] else 'green'
        defect_text.set_text(f'Status: {defect_status}')
        defect_text.set_bbox(dict(boxstyle='round', facecolor=defect_color, alpha=0.8))
        
        flaw_text.set_text(f'Equivalent Flaw Size: {slice_info["equivalent_flawsize"]:.3f}')
        
        # Update colorbar limits for better contrast
        vmin, vmax = np.percentile(slice_info['image'], [1, 99])
        im.set_clim(vmin, vmax)
        
        return [im, title_text, defect_text, flaw_text]

    # Create animation
    logger.info("Creating animation frames...")
    anim = animation.FuncAnimation(
        fig, animate_frame, frames=len(slice_range),
        interval=interval, blit=True, repeat=True
    )
    
    # Save animation
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving animation to {output_path}")
    
    # Use PillowWriter for GIF or FFMpegWriter for MP4
    if output_path.suffix.lower() == '.gif':
        writer = animation.PillowWriter(fps=1000//interval)
    else:
        writer = animation.FFMpegWriter(fps=1000//interval, bitrate=1800)
    
    with tqdm(total=len(slice_range), desc="Saving animation") as pbar:
        def progress_callback(frame, total):
            pbar.update(1)
        
        anim.save(output_path, writer=writer, progress_callback=progress_callback)
    
    plt.close(fig)  # Close figure to free memory
    logger.info(f"Animation saved successfully to {output_path}!")




@cli.command()
@click.option('--test-dir', '-t',
              type=click.Path(exists=True, path_type=Path),
              default=PROCESSED_DATA_DIR / "test",
              help='Path to test data directory')
@click.option('--model-path', '-m',
              type=click.Path(exists=True, path_type=Path),
              default=MODELS_DIR / "checkpoint_epoch_100.pt",
              help='Path to trained model')
@click.option('--n-samples', '-n',
              type=int,
              default=20,
              help='Number of samples to plot')
@click.option('--max-samples',
              type=int,
              default=None,
              help='Maximum number of samples to consider (None = use all)')
@click.option('--output-path', '-o',
              type=click.Path(path_type=Path),
              default=FIGURES_DIR / "predictions_plot.png",
              help='Output path for the plot')
@click.option('--device',
              type=str,
              default="cuda" if torch.cuda.is_available() else "cpu",
              help='Device to use for inference')
@click.option('--seed',
              type=int,
              default=RANDOM_STATE,
              help='Random seed for reproducible sampling')
@click.option('--figsize',
              type=str,
              default="12,8",
              help='Figure size as "width,height"')
def plot_predictions(
    test_dir: Path,
    model_path: Path,
    n_samples: int,
    max_samples: Optional[int],
    output_path: Path,
    device: str,
    seed: int,
    figsize: str
):
    """
    Plot predictions vs ground truth for randomly selected test samples.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Parse figure size
    try:
        fig_width, fig_height = map(float, figsize.split(','))
    except ValueError:
        logger.error("Invalid figsize format. Use 'width,height' (e.g., '12,8')")
        return
    
    # Load model using the new function
    model = load_model(model_path, device)
    
    # Load test data
    logger.info(f"Loading test data from {test_dir}")
    test_loader = get_dataloader(test_dir, batch_size=1, shuffle=False)
    
    # Get all predictions first
    logger.info("Getting all predictions...")
    all_predictions, all_ground_truths = predict_batch(test_loader, model, device)
    
    # Apply max_samples limit if specified
    if max_samples is not None and len(all_predictions) > max_samples:
        logger.info(f"Limiting to first {max_samples} samples")
        all_predictions = all_predictions[:max_samples]
        all_ground_truths = all_ground_truths[:max_samples]
    
    total_available = len(all_predictions)
    logger.info(f"Total available samples: {total_available}")
    
    if n_samples > total_available:
        logger.warning(f"Requested {n_samples} samples but only {total_available} available. Using all {total_available} samples.")
        n_samples = total_available
    
    # Randomly select n samples
    selected_indices = random.sample(range(total_available), n_samples)
    logger.info(f"Randomly selected {n_samples} samples for plotting")
    
    # Get selected predictions and ground truths
    predictions = all_predictions[selected_indices]
    ground_truths = all_ground_truths[selected_indices]
    
    # Create the plot
    plt.figure(figsize=(fig_width, fig_height))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(ground_truths, predictions, alpha=0.7, s=50)
    
    # Perfect prediction line
    min_val = min(min(ground_truths), min(predictions))
    max_val = max(max(ground_truths), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Ground Truth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(ground_truths, predictions)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # Residuals plot
    plt.subplot(2, 2, 2)
    residuals = predictions - ground_truths
    plt.scatter(ground_truths, residuals, alpha=0.7, s=50)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Ground Truth')
    plt.ylabel('Residuals (Pred - Truth)')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # Histogram of residuals
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=min(15, n_samples//3), alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    plt.text(0.05, 0.95, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    
    # Sample indices plot
    plt.subplot(2, 2, 4)
    sample_indices = np.arange(n_samples)
    plt.plot(sample_indices, ground_truths, 'o-', label='Ground Truth', alpha=0.8)
    plt.plot(sample_indices, predictions, 's-', label='Predictions', alpha=0.8)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Sample-wise Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate and display metrics
    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals**2)
    rmse = np.sqrt(mse)
    
    # Overall title with metrics
    plt.suptitle(f'Model Predictions Analysis (n={n_samples})\n'
                f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {correlation**2:.4f}', 
                fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.success(f"Plot saved to {output_path}")
    
    # Print summary statistics
    logger.info("Summary Statistics:")
    logger.info(f"  Mean Absolute Error (MAE): {mae:.4f}")
    logger.info(f"  Root Mean Square Error (RMSE): {rmse:.4f}")
    logger.info(f"  Correlation coefficient: {correlation:.4f}")
    logger.info(f"  R-squared: {correlation**2:.4f}")
    logger.info(f"  Mean residual: {mean_residual:.4f}")
    logger.info(f"  Std residual: {std_residual:.4f}")
    
    plt.show()



if __name__ == "__main__":
    cli()