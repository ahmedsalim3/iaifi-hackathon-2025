# Per-channel z-score normalization
# =================================
# 
# This script computes the per-channel mean and standard deviation of an image dataset.
# It is used to normalize the images to have a mean of 0 and a standard deviation of 1.
# 
# The formula for z-score normalization is:
# 
# Z-score normalization means that for each RGB channel (c):
# z = (x - μ_c) / σ_c
# 
# where:
# - x is the pixel value,
# - μ_c is the mean pixel value of channel (c) across the entire dataset,
# - σ_c is the standard deviation of channel (c) across the entire dataset.

from pathlib import Path
from typing import Optional
import json

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def compute_dataset_mean_std(dataset):
    """
    Compute mean/std over the samples already loaded in a GalaxyDataset
    """
    to_tensor = transforms.ToTensor()

    n_images = 0
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)

    for img, _ in tqdm(dataset, desc="Computing mean/std"):
        tensor = to_tensor(img)  # [C,H,W]
        n_images += 1
        channel_sum += tensor.mean(dim=(1, 2))
        channel_sum_sq += (tensor ** 2).mean(dim=(1, 2))

    mean = channel_sum / n_images
    std = (channel_sum_sq / n_images - mean ** 2).sqrt()
    return mean, std

def _compute_dataset_mean_std(images_path: Path, ext: str = "*.png", save: Optional[bool] = False):
    """
    Compute per-channel mean and standard deviation for an image dataset and save the results to a JSON file.

    Args:
        images_path (Path): Path to the folder containing images.
        ext (str): File extension pattern (default: '*.png').
        save (bool): Whether to save the results to a JSON file.
    Returns:
        tuple: (mean, std), both torch.Tensor of shape (3,)
    """

    # Transform to tensor (0..1)
    to_tensor = transforms.ToTensor()
    image_files = list(images_path.glob(ext))

    if not image_files:
        raise ValueError(f"No images found in {images_path} with extension {ext}")
    
    # Accumulators
    n_images = 0
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)

    for img_file in tqdm(image_files, desc="Computing mean/std"):
        img = Image.open(img_file).convert("RGB")
        tensor = to_tensor(img)  # shape: [C, H, W]

        n_images += 1
        channel_sum += tensor.mean(dim=(1, 2))
        channel_sum_sq += (tensor ** 2).mean(dim=(1, 2))

    # Mean over dataset
    mean = channel_sum / n_images
    std = (channel_sum_sq / n_images - mean ** 2).sqrt()

    if save:
        result = {"mean": mean.tolist(), "std": std.tolist()}
        with open(images_path.parent / "mean_std.json", "w") as f:
            json.dump(result, f, indent=4)

    return mean, std
