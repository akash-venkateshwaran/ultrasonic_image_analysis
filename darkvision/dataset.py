from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import os
import shutil
import numpy as np
import json
import gdown
import py7zr
import random
import torch
from torch.utils.data import Dataset, DataLoader

from darkvision.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, FILE_ID, RANDOM_STATE

app = typer.Typer()

class DataProcessorRaw:
    """
    A class to handle loading and processing of raw data files (.bins, .jsons, .labels).
    """
    
    def __init__(self, filename: str, data_dir: Path = RAW_DATA_DIR):
        """
        Initialize the data processor with a filename.
        
        Args:
            filename: Base filename without extension
            data_dir: Directory containing the data files
        """
        self.filename = filename
        self.data_dir = data_dir
        self.bin_path = data_dir / f"{filename}.bins"
        self.json_path = data_dir / f"{filename}.jsons"
        self.labels_path = data_dir / f"{filename}.labels"
        
        # Data containers
        self._array = None
        self._data_dict = None
        
    def load_bin(self) -> np.ndarray:
        """
        Load a .bins file as a 3D numpy array of shape (100, 256, 256) and dtype uint16.
        """
        if self._array is None:
            logger.info(f"Loading {self.bin_path}")
            arr = np.fromfile(self.bin_path, dtype=np.uint16)
            if arr.size != 256 * 256 * 100:
                raise ValueError(f"Unexpected .bins file size: {arr.size}")
            self._array = arr.reshape((100, 256, 256))
            logger.info(f"Loaded array shape: {self._array.shape}, dtype: {self._array.dtype}")
        return self._array
    
    def load_json(self) -> list:
        """
        Load a .jsons file and return the parsed JSON object.
        """
        logger.info(f"Loading {self.json_path}")
        with open(self.json_path, "r") as f:
            content = f.read()
            if content.startswith("{") and content.endswith("}"):
                # Try to fix concatenated JSON objects
                content = "[" + content.replace("}{", "},{") + "]"
                data = json.loads(content)
                return data
            else:
                # fallback: try to load line by line
                data = [json.loads(line) for line in f]
                return data
    
    def load_labels(self) -> list:
        """
        Load a .labels file and return the lines as a list.
        """
        logger.info(f"Loading {self.labels_path}")
        with open(self.labels_path, "r") as f:
            return [line.strip() for line in f.readlines()]
    
    def load_all_data(self) -> dict:
        """
        Load all data and combine into a comprehensive dictionary.
        Returns a dict with slice indices as keys and slice info as values.
        """
        if self._data_dict is None:
            # Load all data
            array = self.load_bin()
            json_data = self.load_json()
            labels = self.load_labels()
            
            # Create comprehensive data dictionary
            self._data_dict = {}
            
            for slice_idx in range(array.shape[0]):
                slice_info = {
                    'slice_idx': slice_idx,
                    'image': array[slice_idx],
                    'json_data': json_data[slice_idx] if slice_idx < len(json_data) else {},
                    'label': labels[slice_idx] if slice_idx < len(labels) else "N/A",
                    'has_defect': False,
                    'equivalent_flawsize': 0.0
                }
                
                # Extract flaw information
                flaws = slice_info['json_data'].get('flaws', [])
                if flaws:
                    equivalent_flawsize = float(flaws[0].get('equivalent_flawsize', '0.0'))
                    slice_info['equivalent_flawsize'] = equivalent_flawsize
                    slice_info['has_defect'] = equivalent_flawsize > 0.0
                
                self._data_dict[slice_idx] = slice_info
            
            logger.info(f"Loaded complete dataset with {len(self._data_dict)} slices")
        
        return self._data_dict
    
    def get_slice_info(self, slice_idx: int) -> dict:
        """
        Get information for a specific slice.
        
        Args:
            slice_idx: Index of the slice
            
        Returns:
            Dictionary containing slice information
        """
        data_dict = self.load_all_data()
        return data_dict.get(slice_idx, {})
    
    def get_slice_range(self, start_slice: int = 0, end_slice: int = None) -> range:
        """
        Get a range of slice indices.
        
        Args:
            start_slice: Starting slice index
            end_slice: Ending slice index (None for all slices)
            
        Returns:
            Range object for slice indices
        """
        array = self.load_bin()
        if end_slice is None:
            end_slice = array.shape[0]
        end_slice = min(end_slice, array.shape[0])
        return range(start_slice, end_slice)
    

class DataDownloader:
    """
    A class to handle downloading, extracting, and organizing raw data files.
    """
    
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.archive_path = self.raw_data_dir / "raw_data.7z"
    
    def download_file(self, file_id: str, dest: Path) -> None:
        """
        Download a file from Google Drive using its file ID via gdown.
        Args:
            file_id (str): The Google Drive file ID.
            dest (Path): The destination file path.
        Raises:
            Exception: If the download fails.
        """
        try:
            url = f'https://drive.google.com/uc?id={file_id}'
            logger.info(f"Downloading from {url} to {dest}...")
            gdown.download(url, str(dest), quiet=False)
            logger.success(f"Downloaded file to {dest}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise
    
    def extract_7z(self, archive_path: Path, extract_to: Path) -> None:
        """
        Extract a .7z archive to a directory.
        Args:
            archive_path (Path): Path to the .7z archive.
            extract_to (Path): Directory to extract to.
        Raises:
            Exception: If extraction fails.
        """
        try:
            logger.info(f"Extracting {archive_path} to {extract_to}...")
            with py7zr.SevenZipFile(archive_path, mode='r') as archive:
                archive.extractall(path=extract_to)
            logger.success(f"Extraction complete.")
        except py7zr.Bad7zFile:
            logger.error(f"Error: '{archive_path}' is not a valid 7z archive or is corrupted.")
            raise
        except Exception as e:
            logger.error(f"An error occurred during extraction: {e}")
            raise
    
    def remove_file(self, file_path: Path) -> None:
        """
        Remove a file from the filesystem.
        Args:
            file_path (Path): Path to the file to remove.
        Raises:
            Exception: If removal fails.
        """
        try:
            file_path.unlink()
            logger.info(f"Deleted {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            raise
    
    def reorganize_extracted_files(self) -> None:
        """
        Reorganize files after extraction:
        - Move all files from RAW_DATA_DIR/raw_data/data to RAW_DATA_DIR
        - Move README from RAW_DATA_DIR/raw_data to RAW_DATA_DIR
        """
        try:
            # Define source paths
            data_source_dir = self.raw_data_dir / "raw_data" / "data"
            readme_source = self.raw_data_dir / "raw_data" / "README.md"
            
            logger.info("Starting file reorganization...")
            
            # Move all files from data directory to RAW_DATA_DIR
            if data_source_dir.exists():
                logger.info(f"Moving files from {data_source_dir} to {self.raw_data_dir}")
                for item in data_source_dir.iterdir():
                    dest_path = self.raw_data_dir / item.name
                    if item.is_file():
                        shutil.move(str(item), str(dest_path))
                        logger.info(f"Moved file: {item.name}")
                    elif item.is_dir():
                        if dest_path.exists():
                            shutil.rmtree(dest_path)
                        shutil.move(str(item), str(dest_path))
                        logger.info(f"Moved directory: {item.name}")
            else:
                logger.warning(f"Data directory {data_source_dir} does not exist")
            
            # Move README file
            if readme_source.exists():
                readme_dest = self.raw_data_dir / "README.md"
                if readme_dest.exists():
                    readme_dest.unlink()  # Remove existing README if it exists
                shutil.move(str(readme_source), str(readme_dest))
                logger.info(f"Moved README to {readme_dest}")
            else:
                logger.warning(f"README file {readme_source} does not exist")
            
            # Clean up empty raw_data directory structure
            raw_data_extracted_dir = self.raw_data_dir / "raw_data"
            if raw_data_extracted_dir.exists():
                # Remove the data directory if it's empty
                if data_source_dir.exists() and not any(data_source_dir.iterdir()):
                    data_source_dir.rmdir()
                    logger.info(f"Removed empty directory: {data_source_dir}")
                
                # Remove the raw_data directory if it's empty
                if not any(raw_data_extracted_dir.iterdir()):
                    raw_data_extracted_dir.rmdir()
                    logger.info(f"Removed empty directory: {raw_data_extracted_dir}")
            
            logger.success("File reorganization complete.")
            
        except Exception as e:
            logger.error(f"Error during file reorganization: {e}")
            raise
    
    def download_and_extract(self) -> None:
        """
        Download and extract the dataset from Google Drive (.7z archive),
        then reorganize the extracted files.
        """
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if RAW_DATA_DIR is not empty
        if self.raw_data_dir.exists() and any(self.raw_data_dir.iterdir()):
            logger.info(f"RAW_DATA_DIR ({self.raw_data_dir}) is not empty. Skipping download and extraction.")
            return
        
        try:
            # Download the archive
            self.download_file(FILE_ID, self.archive_path)
            
            # Extract the archive
            self.extract_7z(self.archive_path, self.raw_data_dir)
            
            # Reorganize the extracted files
            self.reorganize_extracted_files()
            
        finally:
            # Always clean up the archive file
            if self.archive_path.exists():
                self.remove_file(self.archive_path)

@app.command()
def download_and_extract() -> None:
    """
    Download and extract the dataset from Google Drive (.7z archive).
    """
    processor = DataDownloader()
    processor.download_and_extract()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    """
    Example dataset processing function.
    """
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")

@app.command()
def split_raw_to_train_test(
    test_size: float = 0.2,
    max_samples: int = 1000,
    seed: int = RANDOM_STATE
) -> None:
    """
    Split raw data into train and test sets. Each bin file has 100 slices, each slice is a sample.
    Export X_i (image) and y_i (flaw size) as .npy files in data/processed/train and data/processed/test.
    """
    np.random.seed(seed)
    random.seed(seed)
    raw_dir = RAW_DATA_DIR
    processed_dir = PROCESSED_DATA_DIR
    train_dir = processed_dir / 'train'
    test_dir = processed_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Gather all .bins files
    bin_files = list(raw_dir.glob('*.bins'))
    all_samples = []
    for bin_file in bin_files:
        base = bin_file.stem.replace('.bins', '')
        proc = DataProcessorRaw(base, data_dir=raw_dir)
        arr = proc.load_bin()
        data_dict = proc.load_all_data()
        for i in range(arr.shape[0]):
            all_samples.append((arr[i], data_dict[i]['equivalent_flawsize']))
            if len(all_samples) >= max_samples:
                break
        if len(all_samples) >= max_samples:
            break
    indices = np.arange(len(all_samples))
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    # Save train
    for idx, i in enumerate(train_idx):
        np.save(train_dir / f'X_{idx}.npy', all_samples[i][0])
        np.save(train_dir / f'y_{idx}.npy', np.array([all_samples[i][1]]))
    # Save test
    for idx, i in enumerate(test_idx):
        np.save(test_dir / f'X_{idx}.npy', all_samples[i][0])
        np.save(test_dir / f'y_{idx}.npy', np.array([all_samples[i][1]]))
    logger.success(f"Exported {len(train_idx)} train and {len(test_idx)} test samples to {processed_dir}")

class FlawDataset(Dataset):
    """
    Custom torch Dataset for flaw size regression.
    """
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.X_files = sorted([f for f in self.folder.glob('X_*.npy')])
        self.y_files = sorted([f for f in self.folder.glob('y_*.npy')])
        assert len(self.X_files) == len(self.y_files)
    def __len__(self):
        return len(self.X_files)
    def __getitem__(self, idx):
        X = np.load(self.X_files[idx]).astype(np.float32) / 65535.0  # normalize
        X = np.expand_dims(X, 0)  # (1, 256, 256)
        y = np.load(self.y_files[idx]).astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)

# Example CLI for DataLoader usage
def get_dataloader(folder: Path, batch_size: int = 16, shuffle: bool = True):
    ds = FlawDataset(folder)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

@app.command()
def test_dataloader(
    folder: Path = PROCESSED_DATA_DIR / 'train',
    batch_size: int = 8
):
    """
    Test the custom DataLoader and print batch shapes.
    """
    loader = get_dataloader(folder, batch_size=batch_size)
    for X, y in loader:
        print(f"Batch X: {X.shape}, Batch y: {y.shape}")
        break

if __name__ == "__main__":
    app()