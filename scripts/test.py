import argparse
import os
from datetime import datetime

# core libs
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import yaml
import numpy as np


from sklearn.metrics import confusion_matrix, silhouette_score, accuracy_score, classification_report, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from scipy.stats import entropy

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['figure.dpi'] = 300

# uncomment if needed to access nebula
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nebula.models import cnn_nebula_galaxy, enn_nebula_galaxy
from nebula.data.normalization import compute_dataset_mean_std
from nebula.commons import set_all_seeds

from dataset import GalaxyDataset, SourceDataset, TargetDataset


def rbf_kernel(X, Y, sigma=1.0):
    """
    Compute RBF kernel between two matrices
    https://en.wikipedia.org/wiki/Radial_basis_function_kernel

    """
    X_sqnorms = np.sum(X**2, axis=1, keepdims=True)
    Y_sqnorms = np.sum(Y**2, axis=1, keepdims=True)
    XY = np.dot(X, Y.T)
    distances = X_sqnorms - 2*XY + Y_sqnorms.T
    return np.exp(-distances / (2 * sigma**2))


def compute_mmd(X, Y, sigma=1.0):
    """Compute Maximum Mean Discrepancy between two distributions"""
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    return Kxx.mean() + Kyy.mean() - 2*Kxy.mean()


def safe_load(path: str) -> np.ndarray | None:
    try:
        return np.load(path)
    except FileNotFoundError:
        return None

def load_model(config: dict, config_path: str, device: str) -> torch.nn.Module:
    model_name = config["model"]["name"]
    input_size = tuple(config["model"]["input_size"])
    N = config["model"].get("N", 4)
    dihedral = config["model"].get("dihedral", True)

    if model_name == "cnn":
        model = cnn_nebula_galaxy(input_size=input_size)
    elif model_name == "enn":
        model = enn_nebula_galaxy(input_size=input_size, N=N, dihedral=dihedral)
    else:
        raise ValueError("invalid model name")
    ckpt = os.path.join(os.path.dirname(os.path.abspath(config_path)), "best_model_val_acc.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)
    
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_eval_loaders(config: dict):
    if config['data']['normalize']:
        src_full = SourceDataset(data_root=config['data']['data_root'], split="full")
        mean, std = compute_dataset_mean_std(src_full)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    # apply transforms
    transform = transforms.Compose([
        transforms.Resize(tuple(config["model"]["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    src_ds = SourceDataset(data_root=config['data']['data_root'], split="full", transform=transform)
    tgt_ds = TargetDataset(data_root=config['data']['data_root'], split="full", transform=transform)
    bs = int(config["parameters"].get("batch_size", 32))
    return (
        DataLoader(src_ds, batch_size=bs, shuffle=False),
        DataLoader(tgt_ds, batch_size=bs, shuffle=False),
        mean,
        std,
    )


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features, labels, predictions, and probabilities from a model."""
    feats, labels, preds, probs = [], [], [], []

    for imgs, y in tqdm(loader, unit="batch", total=len(loader), desc=f"Extracting {loader.dataset.__class__.__name__} features"):
        imgs = imgs.to(device).float()
        out = model(imgs)
        if isinstance(out, tuple):
            f, logits = out
        else:
            f, logits = None, out
        if f is None:
            f = logits
        p = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1)
        feats.append(f.detach().cpu().numpy())
        labels.append(y.numpy())
        preds.append(pred.detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())

    feats = np.vstack(feats)
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    probs = np.vstack(probs)
    return feats, labels, preds, probs


def plot_confusion_matrices(
    src_y: np.ndarray,
    src_p: np.ndarray,
    tgt_y: np.ndarray,
    tgt_p: np.ndarray,
    class_names: list[str],
    save_dir: str,
):
    """Plot confusion matrices for both source and target domains with proper legends."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Source domain confusion matrix
    cm_src = confusion_matrix(src_y, src_p)
    sns.heatmap(cm_src, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Source Domain\nAccuracy: {accuracy_score(src_y, src_p):.3f}")
    
    # Target domain confusion matrix
    cm_tgt = confusion_matrix(tgt_y, tgt_p)
    sns.heatmap(cm_tgt, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"Target Domain\nAccuracy: {accuracy_score(tgt_y, tgt_p):.3f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrices.png"), dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_latent_space(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    class_names: list[str],
    save_dir: str,
    method: str = "tsne",
    suffix: str = "",
):
    """Latent space visualization with proper legends and domain adaptation focus."""
    n = min(len(source_features), len(target_features), 1500)
    if len(source_features) > n:
        idx = np.random.choice(len(source_features), n, replace=False)
        source_features = source_features[idx]
        source_labels = source_labels[idx]
    if len(target_features) > n:
        idx = np.random.choice(len(target_features), n, replace=False)
        target_features = target_features[idx]
        target_labels = target_labels[idx]

    X = np.vstack([source_features, target_features])
    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    Z = reducer.fit_transform(X)
    sZ = Z[: len(source_features)]
    tZ = Z[len(source_features) :]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Domain separation
    axes[0,0].scatter(sZ[:,0], sZ[:,1], c='blue', alpha=0.6, s=15, marker='^', label='Source', edgecolors='darkblue', linewidth=0.5)
    axes[0,0].scatter(tZ[:,0], tZ[:,1], c='red', alpha=0.6, s=15, marker='v', label='Target', edgecolors='darkred', linewidth=0.5)
    axes[0,0].legend(fontsize=12)
    axes[0,0].set_title(f"Domain Separation ({method.upper()})", fontsize=14, fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # Source by class
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    for i, name in enumerate(class_names):
        m = source_labels == i
        if np.any(m):
            axes[0,1].scatter(sZ[m,0], sZ[m,1], c=[colors[i]], s=20, alpha=0.7, label=name.capitalize(), edgecolors='black', linewidth=0.5)
    axes[0,1].set_title("Source Domain by Galaxy Type", fontsize=14, fontweight='bold')
    axes[0,1].legend(fontsize=10)
    axes[0,1].grid(True, alpha=0.3)

    # Target by class
    for i, name in enumerate(class_names):
        m = target_labels == i
        if np.any(m):
            axes[1,0].scatter(tZ[m,0], tZ[m,1], c=[colors[i]], s=20, alpha=0.7, label=name.capitalize(), edgecolors='black', linewidth=0.5)
    axes[1,0].set_title("Target Domain by Galaxy Type", fontsize=14, fontweight='bold')
    axes[1,0].legend(fontsize=10)
    axes[1,0].grid(True, alpha=0.3)
    
    # Combined view with class alignment
    for i, name in enumerate(class_names):
        # Source samples of this class
        m_src = source_labels == i
        if np.any(m_src):
            axes[1,1].scatter(sZ[m_src,0], sZ[m_src,1], c=[colors[i]], s=20, alpha=0.7, 
                            marker='^', label=f'{name.capitalize()} (Source)' if i == 0 else "", edgecolors='black', linewidth=0.5)
        
        # Target samples of this class
        m_tgt = target_labels == i
        if np.any(m_tgt):
            axes[1,1].scatter(tZ[m_tgt,0], tZ[m_tgt,1], c=[colors[i]], s=20, alpha=0.7, 
                            marker='v', label=f'{name.capitalize()} (Target)' if i == 0 else "", edgecolors='black', linewidth=0.5)
    
    legend_elements = [
        matplotlib.lines.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Source'),
        matplotlib.lines.Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', markersize=8, label='Target')
    ]
    axes[1,1].legend(handles=legend_elements, fontsize=10, loc='upper right')
    axes[1,1].set_title("Class Alignment Across Domains", fontsize=14, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f"latent_{method}{suffix}.png" if suffix else f"latent_{method}.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_domain_adaptation_metrics(
    source_features: np.ndarray,
    target_features: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
    src_p: np.ndarray,
    tgt_p: np.ndarray,
    class_names: list[str],
    save_dir: str,
):
    """domain adaptation metrics visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Class-wise centroid distances
    centroid_distances = []
    for i in range(len(class_names)):
        sm = source_labels == i
        tm = target_labels == i
        if np.any(sm) and np.any(tm):
            sc = source_features[sm].mean(0)
            tc = target_features[tm].mean(0)
            centroid_distances.append(np.linalg.norm(sc - tc))
        else:
            centroid_distances.append(0)
    
    bars = axes[0,0].bar(range(len(class_names)), centroid_distances, 
                        color=['skyblue', 'lightcoral', 'lightgreen'][:len(class_names)])
    axes[0,0].set_xticks(range(len(class_names)))
    axes[0,0].set_xticklabels([name.capitalize() for name in class_names], rotation=0)
    axes[0,0].set_title("Class-wise Centroid Distances", fontweight='bold')
    axes[0,0].set_ylabel("Distance")
    axes[0,0].grid(True, alpha=0.3)
    for bar, dist in zip(bars, centroid_distances):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{dist:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Domain discriminability
    domain_labels = np.concatenate([np.zeros(len(source_features)), np.ones(len(target_features))])
    combined_features = np.vstack([source_features, target_features])
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(combined_features, domain_labels)
    domain_acc = clf.score(combined_features, domain_labels)
    a_distance = 2 * (1 - 2 * (1 - domain_acc))
    
    domain_sil = silhouette_score(combined_features, domain_labels)
    
    metrics = ['Domain Accuracy', 'A-Distance', 'Domain Silhouette']
    values = [domain_acc, a_distance, domain_sil]
    colors_metrics = ['orange', 'purple', 'green']
    
    bars = axes[0,1].bar(metrics, values, color=colors_metrics, alpha=0.7)
    axes[0,1].set_title("Domain Separability Metrics", fontweight='bold')
    axes[0,1].set_ylabel("Score")
    axes[0,1].set_xticklabels(metrics, rotation=15)
    axes[0,1].grid(True, alpha=0.3)
    
    for bar, val in zip(bars, values):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Per-class accuracy comparison
    src_acc_per_class = []
    tgt_acc_per_class = []
    for i in range(len(class_names)):
        sm = source_labels == i
        tm = target_labels == i
        if np.any(sm):
            # Compute actual source per-class accuracy using predictions
            src_acc_per_class.append(accuracy_score(source_labels[sm], src_p[sm]))
        else:
            src_acc_per_class.append(0.0)
        if np.any(tm):
            # Compute actual target per-class accuracy using predictions
            tgt_acc_per_class.append(accuracy_score(target_labels[tm], tgt_p[tm]))
        else:
            tgt_acc_per_class.append(0.0)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[0,2].bar(x - width/2, src_acc_per_class, width, label='Source', color='skyblue', alpha=0.8)
    axes[0,2].bar(x + width/2, tgt_acc_per_class, width, label='Target', color='lightcoral', alpha=0.8)
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels([name.capitalize() for name in class_names])
    axes[0,2].set_title("Per-Class Performance", fontweight='bold')
    axes[0,2].set_ylabel("Accuracy")
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Feature distribution comparison
    src_norms = np.linalg.norm(source_features, axis=1)
    tgt_norms = np.linalg.norm(target_features, axis=1)
    
    axes[1,0].hist(src_norms, bins=30, alpha=0.7, label='Source', color='skyblue', density=True)
    axes[1,0].hist(tgt_norms, bins=30, alpha=0.7, label='Target', color='lightcoral', density=True)
    axes[1,0].set_title("Feature Magnitude Distribution", fontweight='bold')
    axes[1,0].set_xlabel("L2 Norm")
    axes[1,0].set_ylabel("Density")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Class balance comparison
    src_class_counts = np.bincount(source_labels, minlength=len(class_names))
    tgt_class_counts = np.bincount(target_labels, minlength=len(class_names))
    
    src_percentages = src_class_counts / src_class_counts.sum() * 100
    tgt_percentages = tgt_class_counts / tgt_class_counts.sum() * 100
    
    x = np.arange(len(class_names))
    axes[1,1].bar(x - width/2, src_percentages, width, label='Source', color='skyblue', alpha=0.8)
    axes[1,1].bar(x + width/2, tgt_percentages, width, label='Target', color='lightcoral', alpha=0.8)
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels([name.capitalize() for name in class_names])
    axes[1,1].set_title("Class Distribution Comparison", fontweight='bold')
    axes[1,1].set_ylabel("Percentage (%)")
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. MMD across classes
    mmd_per_class = []
    for i in range(len(class_names)):
        sm = source_labels == i
        tm = target_labels == i
        if np.any(sm) and np.any(tm):
            mmd_val = compute_mmd(source_features[sm], target_features[tm])
            mmd_per_class.append(mmd_val)
        else:
            mmd_per_class.append(0)
    
    bars = axes[1,2].bar(range(len(class_names)), mmd_per_class, 
                        color=['gold', 'lightsteelblue', 'lightpink'][:len(class_names)])
    axes[1,2].set_xticks(range(len(class_names)))
    axes[1,2].set_xticklabels([name.capitalize() for name in class_names])
    axes[1,2].set_title("MMD Distance by Class", fontweight='bold')
    axes[1,2].set_ylabel("MMD")
    axes[1,2].grid(True, alpha=0.3)
    
    for bar, mmd_val in zip(bars, mmd_per_class):
        axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{mmd_val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "domain_adaptation_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    return fig

def compute_da_metrics(source_features, target_features, source_labels, target_labels):
    """Compute domain adaptation metrics."""
    
    # Domain discriminability
    domain_labels = np.concatenate([np.zeros(len(source_features)), np.ones(len(target_features))])
    combined = np.vstack([source_features, target_features])
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(combined, domain_labels)
    domain_acc = clf.score(combined, domain_labels)
    a_distance = 2 * (1 - 2 * (1 - domain_acc))
    
    # Overall MMD
    overall_mmd = compute_mmd(source_features, target_features)
    
    # Domain silhouette
    domain_sil = silhouette_score(combined, domain_labels)
    
    # Per-class analysis
    unique_classes = np.unique(source_labels)
    class_metrics = {}
    
    for c in unique_classes:
        sm = source_labels == c
        tm = target_labels == c
        
        if np.any(sm) and np.any(tm):
            src_class_features = source_features[sm]
            tgt_class_features = target_features[tm]
            
            # Centroids
            src_centroid = src_class_features.mean(0)
            tgt_centroid = tgt_class_features.mean(0)
            centroid_distance = np.linalg.norm(src_centroid - tgt_centroid)
            
            # Class-specific MMD
            class_mmd = compute_mmd(src_class_features, tgt_class_features)
            
            # Intra-class variance
            src_variance = np.var(src_class_features, axis=0).mean()
            tgt_variance = np.var(tgt_class_features, axis=0).mean()
            
            class_metrics[int(c)] = {
                "centroid_distance": float(centroid_distance),
                "class_mmd": float(class_mmd),
                "source_variance": float(src_variance),
                "target_variance": float(tgt_variance),
                "variance_ratio": float(tgt_variance / (src_variance + 1e-8)),
                "source_samples": int(sm.sum()),
                "target_samples": int(tm.sum())
            }
    
    return {
        "overall_metrics": {
            "MMD": float(overall_mmd),
            "A_distance": float(a_distance),
            "domain_accuracy": float(domain_acc),
            "domain_silhouette": float(domain_sil),
        },
        "class_metrics": class_metrics,
        "summary": {
            "avg_centroid_distance": float(np.mean([class_metrics[c]["centroid_distance"] for c in class_metrics])),
            "avg_class_mmd": float(np.mean([class_metrics[c]["class_mmd"] for c in class_metrics])),
            "total_source_samples": int(len(source_features)),
            "total_target_samples": int(len(target_features))
        }
    }


def compute_expected_calibration_error(proba, true_labels, pred_labels, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    confidences = proba.max(axis=1)
    accuracies = (pred_labels == true_labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def plot_domain_classifier_analysis(source_features, target_features, save_dir):
    """Analyze domain classifier performance with ROC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Prepare data
    domain_labels = np.concatenate([np.zeros(len(source_features)), np.ones(len(target_features))])
    combined_features = np.vstack([source_features, target_features])
    
    # Train domain classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(combined_features, domain_labels)
    
    # Get prediction probabilities
    domain_probs = clf.predict_proba(combined_features)[:, 1]
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(domain_labels, domain_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Domain Classifier ROC Curve', fontweight='bold')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # Plot prediction distribution
    src_probs = domain_probs[:len(source_features)]
    tgt_probs = domain_probs[len(source_features):]
    
    axes[1].hist(src_probs, bins=30, alpha=0.7, label='Source (actual=0)', 
                 color='skyblue', density=True)
    axes[1].hist(tgt_probs, bins=30, alpha=0.7, label='Target (actual=1)', 
                 color='lightcoral', density=True)
    axes[1].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision boundary')
    axes[1].set_xlabel('Domain Prediction Probability')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Domain Classifier Predictions', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "domain_classifier_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc


def plot_domain_adaptation_summary(source_features, target_features, source_labels, target_labels, 
                                 src_proba, tgt_proba, src_p, tgt_p, class_names, save_dir):
    """Create domain adaptation visual summary"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Domain Adaptation Performance Summary', fontsize=18, fontweight='bold')
    
    # 1. TSNE visualization (top-left)
    n = min(len(source_features), len(target_features), 1000)
    if len(source_features) > n:
        idx = np.random.choice(len(source_features), n, replace=False)
        src_f_sub = source_features[idx]
        src_l_sub = source_labels[idx]
    else:
        src_f_sub = source_features
        src_l_sub = source_labels
        
    if len(target_features) > n:
        idx = np.random.choice(len(target_features), n, replace=False)
        tgt_f_sub = target_features[idx]
        tgt_l_sub = target_labels[idx]
    else:
        tgt_f_sub = target_features
        tgt_l_sub = target_labels
    
    X = np.vstack([src_f_sub, tgt_f_sub])
    reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    Z = reducer.fit_transform(X)
    sZ = Z[:len(src_f_sub)]
    tZ = Z[len(src_f_sub):]
    
    axes[0,0].scatter(sZ[:,0], sZ[:,1], c='blue', alpha=0.6, s=20, marker='^', 
                     label='Source', edgecolors='darkblue', linewidth=0.5)
    axes[0,0].scatter(tZ[:,0], tZ[:,1], c='red', alpha=0.6, s=20, marker='v', 
                     label='Target', edgecolors='darkred', linewidth=0.5)
    axes[0,0].legend()
    axes[0,0].set_title("Latent Space (TSNE)", fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Class-wise MMD (top-right)
    mmd_per_class = []
    for i in range(len(class_names)):
        sm = source_labels == i
        tm = target_labels == i
        if np.any(sm) and np.any(tm):
            mmd_val = compute_mmd(source_features[sm], target_features[tm])
            mmd_per_class.append(mmd_val)
        else:
            mmd_per_class.append(0)
    
    bars = axes[0,1].bar(range(len(class_names)), mmd_per_class, 
                        color=['gold', 'lightsteelblue', 'lightpink'][:len(class_names)])
    axes[0,1].set_xticks(range(len(class_names)))
    axes[0,1].set_xticklabels([name.capitalize() for name in class_names])
    axes[0,1].set_title("Class-wise MMD Distance", fontweight='bold')
    axes[0,1].set_ylabel("MMD")
    axes[0,1].grid(True, alpha=0.3)
    
    for bar, mmd_val in zip(bars, mmd_per_class):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{mmd_val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Confidence gap per class (bottom-left)
    src_conf = src_proba.max(axis=1)
    tgt_conf = tgt_proba.max(axis=1)
    
    conf_gap_per_class = []
    for i in range(len(class_names)):
        src_mask = source_labels == i
        tgt_mask = target_labels == i
        
        src_conf_class = src_conf[src_mask].mean() if src_mask.sum() > 0 else 0
        tgt_conf_class = tgt_conf[tgt_mask].mean() if tgt_mask.sum() > 0 else 0
        conf_gap_per_class.append(abs(src_conf_class - tgt_conf_class))
    
    bars = axes[1,0].bar(range(len(class_names)), conf_gap_per_class, 
                        color=['orange', 'lightgreen', 'lightcyan'][:len(class_names)])
    axes[1,0].set_xticks(range(len(class_names)))
    axes[1,0].set_xticklabels([name.capitalize() for name in class_names])
    axes[1,0].set_title("Confidence Gap Between Domains", fontweight='bold')
    axes[1,0].set_ylabel("Absolute Difference")
    axes[1,0].grid(True, alpha=0.3)
    
    for bar, gap in zip(bars, conf_gap_per_class):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                      f'{gap:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Overall DA metrics (bottom-right)
    src_acc = accuracy_score(source_labels, src_p)
    tgt_acc = accuracy_score(target_labels, tgt_p)
    domain_gap = src_acc - tgt_acc
    
    overall_mmd = compute_mmd(source_features, target_features)
    src_ece = compute_expected_calibration_error(src_proba, source_labels, src_p)
    tgt_ece = compute_expected_calibration_error(tgt_proba, target_labels, tgt_p)
    
    metrics = ['Domain\nGap', 'Overall\nMMD', 'Source\nECE', 'Target\nECE']
    values = [domain_gap, overall_mmd, src_ece, tgt_ece]
    colors = ['lightcoral', 'gold', 'lightblue', 'lightpink']
    
    bars = axes[1,1].bar(metrics, values, color=colors, alpha=0.8)
    axes[1,1].set_title("Key DA Metrics", fontweight='bold')
    axes[1,1].set_ylabel("Score")
    axes[1,1].grid(True, alpha=0.3)
    
    for bar, val in zip(bars, values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "domain_adaptation_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_confidence_analysis(src_proba, tgt_proba, src_y, tgt_y, src_p, tgt_p, class_names, save_dir):
    """Analyze prediction confidence patterns across domains."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Compute ECE for both domains
    src_ece = compute_expected_calibration_error(src_proba, src_y, src_p)
    tgt_ece = compute_expected_calibration_error(tgt_proba, tgt_y, tgt_p)
    
    # Confidence distribution comparison
    src_conf = src_proba.max(axis=1)
    tgt_conf = tgt_proba.max(axis=1)
    
    axes[0,0].hist(src_conf, bins=30, alpha=0.7, label=f'Source (ECE: {src_ece:.4f})', color='skyblue', density=True)
    axes[0,0].hist(tgt_conf, bins=30, alpha=0.7, label=f'Target (ECE: {tgt_ece:.4f})', color='lightcoral', density=True)
    axes[0,0].set_title("Prediction Confidence Distribution", fontweight='bold')
    axes[0,0].set_xlabel("Maximum Probability")
    axes[0,0].set_ylabel("Density")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Confidence vs Accuracy (Calibration)
    def plot_calibration(proba, true_labels, pred_labels, ax, domain_name, color):
        confidences = proba.max(axis=1)
        accuracies = (pred_labels == true_labels).astype(int)
        
        bins = np.linspace(0, 1, 11)
        bin_centers, bin_accs, bin_confs = [], [], []
        
        for i in range(len(bins)-1):
            mask = (confidences >= bins[i]) & (confidences < bins[i+1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i+1]) / 2)
                bin_accs.append(accuracies[mask].mean())
                bin_confs.append(confidences[mask].mean())
        
        if bin_centers:
            ax.scatter(bin_confs, bin_accs, label=domain_name, s=50, alpha=0.8, color=color)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    
    plot_calibration(src_proba, src_y, src_p, axes[0,1], 'Source', 'blue')
    plot_calibration(tgt_proba, tgt_y, tgt_p, axes[0,1], 'Target', 'red')
    axes[0,1].set_title("Confidence vs Accuracy", fontweight='bold')
    axes[0,1].set_xlabel("Average Confidence")
    axes[0,1].set_ylabel("Accuracy")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Per-class confidence comparison
    src_conf_per_class = []
    tgt_conf_per_class = []
    
    for i in range(len(class_names)):
        src_mask = src_y == i
        tgt_mask = tgt_y == i
        
        if src_mask.sum() > 0:
            src_conf_per_class.append(src_proba[src_mask].max(axis=1).mean())
        else:
            src_conf_per_class.append(0)
            
        if tgt_mask.sum() > 0:
            tgt_conf_per_class.append(tgt_proba[tgt_mask].max(axis=1).mean())
        else:
            tgt_conf_per_class.append(0)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[0,2].bar(x - width/2, src_conf_per_class, width, label='Source', color='skyblue', alpha=0.8)
    axes[0,2].bar(x + width/2, tgt_conf_per_class, width, label='Target', color='lightcoral', alpha=0.8)
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels([name.capitalize() for name in class_names])
    axes[0,2].set_title("Average Confidence by Class", fontweight='bold')
    axes[0,2].set_ylabel("Average Confidence")
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Entropy analysis
    src_entropy = entropy(src_proba.T)
    tgt_entropy = entropy(tgt_proba.T)
    
    axes[1,0].hist(src_entropy, bins=30, alpha=0.7, label='Source', color='skyblue', density=True)
    axes[1,0].hist(tgt_entropy, bins=30, alpha=0.7, label='Target', color='lightcoral', density=True)
    axes[1,0].set_title("Prediction Entropy Distribution", fontweight='bold')
    axes[1,0].set_xlabel("Entropy")
    axes[1,0].set_ylabel("Density")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Misclassification confidence
    src_correct = (src_p == src_y)
    tgt_correct = (tgt_p == tgt_y)
    
    src_correct_conf = src_conf[src_correct]
    src_wrong_conf = src_conf[~src_correct]
    tgt_correct_conf = tgt_conf[tgt_correct]
    tgt_wrong_conf = tgt_conf[~tgt_correct]
    
    data_to_plot = []
    labels = []
    if len(src_correct_conf) > 0:
        data_to_plot.append(src_correct_conf)
        labels.append('Source Correct')
    if len(src_wrong_conf) > 0:
        data_to_plot.append(src_wrong_conf)
        labels.append('Source Wrong')
    if len(tgt_correct_conf) > 0:
        data_to_plot.append(tgt_correct_conf)
        labels.append('Target Correct')
    if len(tgt_wrong_conf) > 0:
        data_to_plot.append(tgt_wrong_conf)
        labels.append('Target Wrong')
    
    if data_to_plot:
        box_plot = axes[1,1].boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightcyan', 'lightcoral', 'mistyrose']
        for patch, color in zip(box_plot['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
    
    axes[1,1].set_title("Confidence by Prediction Correctness", fontweight='bold')
    axes[1,1].set_ylabel("Confidence")
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    # Domain gap in confidence
    conf_gap_per_class = []
    for i in range(len(class_names)):
        src_mask = src_y == i
        tgt_mask = tgt_y == i
        
        src_conf_class = src_proba[src_mask].max(axis=1).mean() if src_mask.sum() > 0 else 0
        tgt_conf_class = tgt_proba[tgt_mask].max(axis=1).mean() if tgt_mask.sum() > 0 else 0
        conf_gap_per_class.append(abs(src_conf_class - tgt_conf_class))
    
    bars = axes[1,2].bar(range(len(class_names)), conf_gap_per_class, 
                        color=['gold', 'lightsteelblue', 'lightpink'][:len(class_names)])
    axes[1,2].set_xticks(range(len(class_names)))
    axes[1,2].set_xticklabels([name.capitalize() for name in class_names])
    axes[1,2].set_title("Confidence Gap Between Domains", fontweight='bold')
    axes[1,2].set_ylabel("Absolute Difference")
    axes[1,2].grid(True, alpha=0.3)
    
    for bar, gap in zip(bars, conf_gap_per_class):
        axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                      f'{gap:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confidence_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain adaptation evaluation")
    parser.add_argument("--config", required=True, help="Path to training-generated config.yaml")
    args = parser.parse_args()

    print("Starting evaluation...")
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    set_all_seeds(config.get("seed"))
    model_name = config["model"]["name"]
    report_interval = int(config["parameters"].get("report_interval", 1))
    warmup = int(config["parameters"].get("warmup", 0))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results_dir = os.path.join(os.path.dirname(os.path.abspath(args.config)), "results")
    os.makedirs(results_dir, exist_ok=True)    
    model = load_model(config, args.config, device)
    src_loader, tgt_loader, mean, std = build_eval_loaders(config)
    
    # features and predictions
    src_f, src_y, src_p, src_proba = extract_features(model, src_loader, device)
    tgt_f, tgt_y, tgt_p, tgt_proba = extract_features(model, tgt_loader, device)
    
    # class names
    idx2label = {v: k for k, v in GalaxyDataset.LABEL_MAPPING.items()}
    class_names = [idx2label[i] for i in range(len(idx2label))]
    
    print(f"Class names: {[name.capitalize() for name in class_names]}")
    print(f"Source samples: {len(src_f)}, Target samples: {len(tgt_f)}")
    
    # accuracies
    src_acc = accuracy_score(src_y, src_p)
    tgt_acc = accuracy_score(tgt_y, tgt_p)
    print(f"Source accuracy: {src_acc:.3f}, Target accuracy: {tgt_acc:.3f}")
        
    plot_confusion_matrices(src_y, src_p, tgt_y, tgt_p, class_names, results_dir)
    
    # Latent space with true labels
    plot_latent_space(src_f, tgt_f, src_y, tgt_y, class_names, results_dir, method="tsne")
    plot_latent_space(src_f, tgt_f, src_y, tgt_y, class_names, results_dir, method="pca")

    # Latent space with predicted labels
    src_labels_pred = src_p  # predicted by model
    tgt_labels_pred = tgt_p  # predicted by model
    plot_latent_space(
        source_features=src_f,
        target_features=tgt_f,
        source_labels=src_labels_pred,   # predicted labels
        target_labels=tgt_labels_pred,   # predicted labels
        class_names=class_names,
        save_dir=results_dir,
        method="tsne",
        suffix="_pred"
    )
    plot_latent_space(
        source_features=src_f,
        target_features=tgt_f,
        source_labels=src_labels_pred,   # predicted labels
        target_labels=tgt_labels_pred,   # predicted labels
        class_names=class_names,
        save_dir=results_dir,
        method="pca",
        suffix="_pred"
    )
    plot_domain_adaptation_metrics(src_f, tgt_f, src_y, tgt_y, src_p, tgt_p, class_names, results_dir)
    plot_prediction_confidence_analysis(src_proba, tgt_proba, src_y, tgt_y, src_p, tgt_p, class_names, results_dir)    
    domain_roc_auc = plot_domain_classifier_analysis(src_f, tgt_f, results_dir)    
    plot_domain_adaptation_summary(src_f, tgt_f, src_y, tgt_y, src_proba, tgt_proba, 
                                  src_p, tgt_p, class_names, results_dir)
    
    da_metrics = compute_da_metrics(src_f, tgt_f, src_y, tgt_y)    
    da_metrics["domain_classifier"] = {"roc_auc": float(domain_roc_auc)}
    
    with open(os.path.join(results_dir, "da_metrics.yaml"), "w") as f:
        yaml.safe_dump(da_metrics, f, default_flow_style=False, indent=2)
