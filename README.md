# ultrasonic_image_analysis

![Animation](reports/figures/animation.gif)
![Prediction Plot](reports/figures/predictions_plot.png)

A project for automated flaw detection and regression using ultrasonic non-destructive testing (NDT) data. This repository provides workflows for image analysis, raw ultrasonic data processing, and predictive model creation using deep learning.

## WHAT
- End-to-end pipeline for flaw detection and flaw size regression from ultrasonic images
- Data preprocessing, splitting, and augmentation utilities
- Custom PyTorch Dataset and DataLoader for 3D ultrasonic data
- Simple CNN model for flaw size regression
- CLI tools for data download, processing, training, and prediction
- Visualization tools for inspection and model evaluation

## INSTRUCTION
1. **Download and extract the data:**
   ```sh
   python -m darkvision.dataset download-and-extract
   ```
2. **Split raw data into train/test sets:**
   ```sh
   python -m darkvision.dataset split-raw-data --test-size 0.2 --val-size 0.1 --max-samples 10000 --num-cpus 8 --seed 42
   ```
3. **Test the DataLoader:**
   ```sh
   python -m darkvision.dataset test-dataloader --folder data/processed/train --batch-size 8
   ```
4. **Train the CNN model:**
   ```sh
   python -m darkvision.modeling.train train --batch-size 16 --epochs 10 --lr 0.001 --device cuda
   ```
5. **Plot predictions vs ground truth:**
   ```sh
   python -m darkvision.plots plot-predictions --test-dir data/processed/test --model-path models/checkpoint_epoch_100.pt --n-samples 20
   ```
6. **Inspect and animate slices:**
   ```sh
   python -m darkvision.plots inspect --filename <UUID> --slice-idx 4
   python -m darkvision.plots animate --filename <UUID> --output-path reports/figures/animation.gif
   ```

## DATA
Please refer to https://arxiv.org/abs/1903.11399 for details.

**Contents:**
- The directory `data` contains ultrasonic data sets, containing various flaws. Each batch file is named with a UUID and contains:
  - `.bins` file: raw data (UInt16, 256 x 256 x 100)
  - `.meta` file: documents the raw data format
  - `.jsons` file: json-formatted meta-data for each binary file (flaw locations, source flaw size, equivalent size)
  - `.labels` file: tab-separated data for flaw existence (0/1) and equivalent flaw size
- The directory `src` contains python code to train a deep CNN using the data provided. Use `./train.py` to run.
- For inference, consult the sample code in `src/inference.py`.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         darkvision and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── darkvision   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes darkvision a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

