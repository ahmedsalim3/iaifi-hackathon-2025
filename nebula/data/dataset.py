from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import json
import random

from .base import BaseGalaxyDataset


@dataclass
class DatasetConfig:
    """Configuration dataclass for galaxy datasets."""
    root: str
    name: str
    split: str = "train"


class GalaxyDataset(BaseGalaxyDataset, ABC):
    """Abstract dataset class handling common galaxy dataset logic."""
    
    LABEL_MAPPING = {"elliptical": 0, "spiral": 1, "irregular": 2}
    
    def __init__(
        self,
        data_root: str,
        transform: Optional[Callable] = None,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: Optional[int] = 42,
        train_ratio: float = 0.8,
    ):
        self.seed = seed
        self.split = split
        self.train_ratio = train_ratio
        
        cfg = self._create_config(data_root, split)
        super().__init__(cfg, transform, self.LABEL_MAPPING, max_samples)
        
        self._load_and_split_data()
    
    @abstractmethod
    def _create_config(self, data_root: str, split: str) -> DatasetConfig:
        """Create dataset-specific configuration."""
        pass
    
    @abstractmethod
    def _get_json_path(self) -> Path:
        """Get path to JSON file containing dataset labels."""
        pass
    
    def _process_item(self, item: dict) -> dict:
        """Process individual data item. Override for dataset-specific quirks."""
        return {
            "im_path": item["image_path"],
            "label": self.label2idx[item["classification"]],
        }
    
    def _load_and_split_data(self):
        """Load data from JSON and handle train/test splitting."""
        json_path = self._get_json_path()
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Process all items
        samples = [self._process_item(item) for item in data]
        
        # Shuffle with seed for reproducibility
        random.seed(self.seed)
        random.shuffle(samples)
        
        # Split into train/test
        n_train = int(self.train_ratio * len(samples))
        if self.split == "train":
            samples = samples[:n_train]
        elif self.split == "test":  # test
            samples = samples[n_train:]
        else:
            samples = samples # full dataset
        
        # for dev, apply max_samples limit
        if self.max_samples:
            samples = samples[:self.max_samples]
        
        self.samples = samples


class SourceDataset(GalaxyDataset):
    """Dataset class for IllustrisTNG source domain data."""
    
    def _create_config(self, data_root: str, split: str) -> DatasetConfig:
        return DatasetConfig(root=data_root, name="source", split=split)
    
    def _get_json_path(self) -> Path:
        return Path(self.cfg.root) / "source" / "labels_master.json"


class TargetDataset(GalaxyDataset):
    """Dataset class for Galaxy Zoo 2 target domain data."""
    
    def _create_config(self, data_root: str, split: str) -> DatasetConfig:
        return DatasetConfig(root=data_root, name="target", split=split)
    
    def _get_json_path(self) -> Path:
        return Path(self.cfg.root) / "target" / "labels_master_top_n.json"
    
    def _process_item(self, item: dict) -> dict:
        """Process target dataset items with path normalization."""
        # Convert absolute path to relative path
        image_path = item["image_path"]
        if image_path.startswith("data/target/"):
            image_path = image_path[len("data/target/"):]
        
        return {
            "im_path": image_path,
            "label": self.label2idx[item["classification"]],
        }
