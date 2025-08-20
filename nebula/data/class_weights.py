import torch
import numpy as np


def compute_class_weights(
    labels,
    method: str = "effective",
    beta: float = 0.9999,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
):
    """Compute per-class weights for imbalanced datasets.

    Supports:
    - method="balanced": inverse-frequency weighting (same average weight of 1 over present classes)
    - method="effective": Class-Balanced weights based on Effective Number of Samples
      https://arxiv.org/abs/1901.05555, CVPR'19.
      https://github.com/vandit15/Class-balanced-loss-pytorch/


    Args:
        labels: 1D array-like of integer class labels (NumPy array or torch tensor).
        method: "balanced" or "effective".
        beta: Beta parameter for the effective number formula (only used when method="effective").
        normalize: If True, rescale weights so the mean over present classes equals 1.
        dtype: Torch dtype of the returned tensor.
        device: Optional torch device for the returned tensor.

    Returns:
        torch.Tensor of shape [C], where C = max(labels) + 1.
    """

    # Convert labels to a flat NumPy integer array
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    else:
        labels = np.asarray(labels)
    labels = labels.reshape(-1)
    assert np.issubdtype(labels.dtype, np.integer), "Labels must be integers."

    if labels.size == 0:
        return torch.zeros(0, dtype=dtype, device=device)

    # Determine number of classes from labels
    inferred_num_classes = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=inferred_num_classes).astype(np.int64)

    present_mask = counts > 0

    if method.lower() == "balanced":
        # Standard balanced weights
        total = labels.size
        num_present = int(present_mask.sum())
        weights = np.empty_like(counts, dtype=np.float64)
        weights[present_mask] = total / (max(num_present, 1) * counts[present_mask])

    elif method.lower() == "effective":
        # Effective number of samples
        weights = np.zeros_like(counts, dtype=np.float64)
        if np.any(present_mask):
            effective_num = 1.0 - np.power(beta, counts[present_mask])
            weights[present_mask] = (1.0 - beta) / np.maximum(effective_num, np.finfo(np.float64).eps)

    # normalize so the average weight over present classes is 1.0
    if normalize and np.any(present_mask):
        present_sum = float(weights[present_mask].sum())
        if present_sum > 0.0:
            scale_target = int(present_mask.sum())
            weights = weights * (scale_target / present_sum)

    return torch.tensor(weights, dtype=dtype, device=device)
