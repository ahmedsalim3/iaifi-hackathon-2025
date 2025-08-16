from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class BaseGalaxyDataset(Dataset):
    """Base dataset with shared config, label mapping, and utility methods."""

    LABELS = {"elliptical": 0, "spiral": 1, "irregular": 2}

    def __init__(
        self,
        cfg,
        transform: Optional[Callable] = None,
        label_mapping: Optional[dict[str, int]] = None,
        max_samples: Optional[int] = None,
        seed: Optional[int] = 42,
    ):
        self.cfg = cfg
        self.root = Path(cfg.root) / cfg.name
        self.transform = transform
        self.samples = []
        self.max_samples = max_samples

        if label_mapping:
            self.label2idx, self.idx2label = self.get_label_maps(label_mapping)
        else:
            self.label2idx, self.idx2label = {}, {}

    @staticmethod
    def get_label_maps(mapping: dict[str, int]):
        return mapping, {v: k for k, v in mapping.items()}

    @staticmethod
    def numpy_to_pil(img_array: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image."""
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1 else img_array.astype(np.uint8)
        if img_array.ndim == 2:
            return Image.fromarray(img_array, "L")
        if img_array.ndim == 3:
            channels = img_array.shape[2]
            if channels == 1: return Image.fromarray(img_array.squeeze(2), "L")
            if channels == 3: return Image.fromarray(img_array, "RGB")
            if channels == 4: return Image.fromarray(img_array, "RGBA")
        raise ValueError(f"Unsupported shape: {img_array.shape}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample = deepcopy(self.samples[idx])
        if "im_path" in sample:
            img = Image.open(self.root / sample["im_path"]).convert("RGB")
        elif "image" in sample:
            img = self.numpy_to_pil(sample["image"])
        else:
            raise KeyError("Sample missing 'im_path' or 'image'")

        if self.transform:
            img = self.transform(img)

        label = sample.get("label", -1)
        if hasattr(self, "label_map_override"):
            label = self.label_map_override.get(label, label)

        return img, torch.tensor(label, dtype=torch.long)

    @property
    def num_samples(self): return len(self.samples)

    @property
    def num_ids(self):
        return len({s["label"] for s in self.samples}) if self.samples and "label" in self.samples[0] else -1

    def get_class_distribution(self):
        """Get detailed class distribution with counts and percentages."""
        if not self.samples or "label" not in self.samples[0]:
            return {}
        
        class_counts = {}
        for sample in self.samples:
            label = sample["label"]
            if hasattr(self, "label_map_override"):
                label = self.label_map_override.get(label, label)
            class_counts[label] = class_counts.get(label, 0) + 1
        
        total = sum(class_counts.values())
        class_info = {}
        for label_idx, count in class_counts.items():
            label_name = self.idx2label.get(label_idx, f"class_{label_idx}")
            class_info[label_name] = {
                "index": label_idx,
                "count": count,
                "percentage": (count / total) * 100
            }
        
        return class_info

    def get_data_info(self):
        """Get information about data source and structure."""
        info = {
            "dataset_type": self.__class__.__name__,
            "total_samples": len(self.samples),
            "data_source": "file_paths" if self.samples and "im_path" in self.samples[0] else "arrays",
            "has_transforms": self.transform is not None,
            "root_path": str(self.root),
        }
        
        if hasattr(self, 'max_samples') and self.max_samples:
            info["max_samples_limit"] = self.max_samples
            
        if hasattr(self, 'label_map_override'):
            info["has_label_remapping"] = True
            info["label_mapping"] = getattr(self, 'label_map_override', {})
        else:
            info["has_label_remapping"] = False
            
        return info

    @property
    def summary(self):
        # class distribution
        class_counts = {}
        if self.samples and "label" in self.samples[0]:
            for sample in self.samples:
                label = sample["label"]
                if hasattr(self, "label_map_override"):
                    label = self.label_map_override.get(label, label)
                class_counts[label] = class_counts.get(label, 0) + 1
        
        # image dimensions from first sample
        img_dims = "Unknown"
        if self.samples:
            try:
                if "im_path" in self.samples[0]:
                    img = Image.open(self.root / self.samples[0]["im_path"])
                    img_dims = f"{img.size[0]}x{img.size[1]}"
                elif "image" in self.samples[0]:
                    img_array = self.samples[0]["image"]
                    if img_array.ndim >= 2:
                        img_dims = f"{img_array.shape[1]}x{img_array.shape[0]}"
                        if img_array.ndim == 3:
                            img_dims += f"x{img_array.shape[2]}"
            except Exception:
                img_dims = "Error reading dims"
        
        # class distribution string
        class_dist_str = ""
        if class_counts:
            total = sum(class_counts.values())
            dist_items = []
            for label_idx in sorted(class_counts.keys()):
                count = class_counts[label_idx]
                percentage = (count / total) * 100
                label_name = self.idx2label.get(label_idx, f"class_{label_idx}")
                dist_items.append(f"{label_name}: {count} ({percentage:.1f}%)")
            class_dist_str = ", ".join(dist_items)
        
        # build summary lines
        lines = [
            "=" * 25,
            self.__class__.__name__,
            "=" * 25,
            f"split: {getattr(self.cfg, 'split', 'N/A')}",
            f"root: {self.root}",
            f"# images: {self.num_samples}",
            f"# classes: {self.num_ids}",
            f"image dims: {img_dims}",
            f"transform: {'Yes' if self.transform else 'None'}",
        ]
        
        # max_samples info if applicable
        if hasattr(self, 'max_samples') and self.max_samples:
            lines.append(f"max_samples: {self.max_samples}")
        
        # label mapping info if applicable
        if hasattr(self, 'label_map_override') and self.label_map_override:
            lines.append(f"label mapping: {self.label_map_override}")
        
        # class distribution
        if class_dist_str:
            lines.append(f"class distribution: {class_dist_str}")
        
        lines.append("")  # empty line at end
        
        return "\n".join(lines)
