# Waffle
This repository contains experiments using datasets from `torchvision.datasets`. All experiments are implemented as Jupyter notebooks in the `./notebooks` directory.

## Requirements
Install the required Python packages using: pip install -r requirements.txt

## Data preparation 
Before running the experiments, download the dataset using torchvision.datasets and save the images in the following structure:
./data/dataset_name/
    ├── train/
    │   └── x{i}_y{label}.png
    └── test/
        └── x{i}_y{label}.png

•	dataset_name can be, for example, cifar10 or fashion_mnist
•	Images must be in .png format
•	i is a unique index
•	label is the class label

The data/ folder should be at the same level as the src/ folder.

## Running Experiments
All experiments are located in the ./notebooks folder. After setting up the data and installing the requirements, notebooks can be run directly.