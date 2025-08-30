import json
import random
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, random_split
from PIL import Image

from nebula.commons import Logger

logger = Logger()

class GalaxyDataset(Dataset):
    """Galaxy dataset for source and target domains."""

    LABEL_MAPPING = {"elliptical": 0, "spiral": 1, "irregular": 2}

    def __init__(
        self,
        data_root: str,
        domain_type: str,  # 'source' or 'target'
        transform: Optional[Callable] = None,
        split: str = "full",  # 'train', 'test', or 'full'
        train_ratio: float = 0.8,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        self.domain_type = domain_type
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.max_samples = max_samples
        self.seed = seed

        # Set paths based on domain type
        if domain_type == "source":
            self.data_path = self.data_root / "source"
            self.json_file = "labels_master.json"
        elif domain_type == "target":
            self.data_path = self.data_root / "target"
            self.json_file = "labels_master_top_n.json"
        else:
            raise ValueError(f"domain_type must be 'source' or 'target', got {domain_type}")

        # Load and process data
        self.samples = self._load_data()

    def _load_data(self):
        json_path = self.data_path / self.json_file
        with open(json_path, 'r') as f:
            data = json.load(f)

        samples = []
        for item in data:
            image_path = item["image_path"]
            if self.domain_type == "target" and image_path.startswith("data/target/"):
                image_path = image_path[len("data/target/"):]
            samples.append({
                "image_path": image_path,
                "label": self.LABEL_MAPPING[item["classification"]]
            })

        random.seed(self.seed)
        random.shuffle(samples)

        if self.split != "full":
            n_train = int(self.train_ratio * len(samples))
            if self.split == "train":
                samples = samples[:n_train]
            elif self.split == "test":
                samples = samples[n_train:]

        if self.max_samples:
            samples = samples[:self.max_samples]

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = self.data_path / sample["image_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)  # Transform applied here, on the original PIL image
        label = torch.tensor(sample["label"], dtype=torch.long)
        return img, label

    def get_class_distribution(self):
        class_counts = {}
        for sample in self.samples:
            label = sample["label"]
            class_counts[label] = class_counts.get(label, 0) + 1

        total = len(self.samples)
        distribution = {}
        idx_to_name = {v: k for k, v in self.LABEL_MAPPING.items()}
        for label_idx, count in class_counts.items():
            distribution[idx_to_name[label_idx]] = {
                "count": count,
                "percentage": (count / total) * 100
            }
        return distribution

    def __repr__(self):
        return (f"GalaxyDataset(domain={self.domain_type}, split={self.split}, "
                f"samples={len(self.samples)}, classes={len(self.LABEL_MAPPING)})")


def split_dataset(dataset, val_size=0.2, train_transform=None, val_transform=None, seed=42):
    torch.manual_seed(seed)
    val_len = int(len(dataset) * val_size)
    train_len = len(dataset) - val_len
    train_subset, val_subset = random_split(dataset, [train_len, val_len])
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    return train_subset, val_subset


def SourceDataset(data_root: str, **kwargs):
    return GalaxyDataset(data_root, domain_type="source", **kwargs)


def TargetDataset(data_root: str, **kwargs):
    return GalaxyDataset(data_root, domain_type="target", **kwargs)


if __name__ == "__main__":
    from torchvision import transforms
    
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    src_dataset = SourceDataset(data_root="data", split="full")
    train_set, val_set = split_dataset(src_dataset, val_size=0.2,
                                       train_transform=train_transform,
                                       val_transform=val_transform)

    logger.info("Train samples:", len(train_set))
    logger.info("Validation samples:", len(val_set))
    logger.info("Class distribution:", src_dataset.get_class_distribution())
