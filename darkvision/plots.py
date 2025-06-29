from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from darkvision.dataset import DataProcessorRaw
import numpy as np

from loguru import logger
from tqdm import tqdm
import typer

from darkvision.config import FIGURES_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()



@app.command()
def inspect(
    filename: str = "0A4A7A0F-A8A4-40A5-95A3-2D15AEC422E3",
    slice_idx: int = 4,
    output_path: Path = FIGURES_DIR / "plot.png",
):
    """
    Inspect a .bins file, plot a slice, and display info from .jsons and .labels.
    """
    processor = DataProcessorRaw(filename)
    slice_info = processor.get_slice_info(slice_idx)
    
    print(f"Array shape: {processor.load_bin().shape}, dtype: {processor.load_bin().dtype}")
    print(f"Slice info keys: {slice_info.keys()}")
    print(f"Has defect: {slice_info['has_defect']}")
    print(f"Equivalent flaw size: {slice_info['equivalent_flawsize']}")

    # Plot a slice
    plt.figure(figsize=(6, 6))
    plt.imshow(slice_info['image'], cmap="gray")
    plt.title(f"{filename} - Slice {slice_idx}")
    plt.axis("off")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.show()

    # Print detailed info
    print("\n=== Slice Information ===")
    print(f"Slice: {slice_info['slice_idx']}")
    print(f"Label: {slice_info['label']}")
    print(f"Has Defect: {slice_info['has_defect']}")
    print(f"Equivalent Flaw Size: {slice_info['equivalent_flawsize']}")
    print(f"JSON Data: {slice_info['json_data']}")

@app.command()
def animate(
    filename: str = "0A4A7A0F-A8A4-40A5-95A3-2D15AEC422E3",
    output_path: Path = FIGURES_DIR / "animation.gif",
    interval: int = 200,
    start_slice: int = 0,
    end_slice: int = None,
):
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

    # Set up the figure and axis - don't show during processing
    plt.ioff()  # Turn off interactive mode
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get first slice info to initialize
    first_slice_info = processor.get_slice_info(start_slice)
    
    # Create initial image
    im = ax.imshow(first_slice_info['image'], cmap='gray', animated=True)
    ax.set_xlim(0, first_slice_info['image'].shape[1])
    ax.set_ylim(first_slice_info['image'].shape[0], 0)  # Flip y-axis for proper image orientation
    ax.axis('off')  # Remove axes for cleaner look
    
    # Set up text annotations
    title_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        fontsize=16, fontweight='bold', 
                        verticalalignment='top', color='white',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    defect_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, 
                         fontsize=14, fontweight='bold', verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    flaw_text = ax.text(0.02, 0.78, '', transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top', color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

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

@app.command()
def animate_mp4(
    filename: str = "0A4A7A0F-A8A4-40A5-95A3-2D15AEC422E3",
    output_path: Path = FIGURES_DIR / "animation.mp4",
    fps: int = 5,
    start_slice: int = 0,
    end_slice: int = None,
):
    """
    Create an MP4 animation cycling through slices with defect information.
    """
    # Change output path to MP4 and call animate with converted parameters
    animate(
        filename=filename,
        output_path=output_path,
        interval=1000//fps,  # Convert fps to interval
        start_slice=start_slice,
        end_slice=end_slice
    )

if __name__ == "__main__":
    app()