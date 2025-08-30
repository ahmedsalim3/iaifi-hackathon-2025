import argparse
import os
from datetime import datetime

# core libs
from geomloss import SamplesLoss
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

# uncomment if needed to access nebula
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nebula.data.normalization import compute_dataset_mean_std
from nebula.data.class_weights import compute_class_weights
from nebula.models import cnn_nebula_galaxy, enn_nebula_galaxy
from nebula.commons import Logger, set_all_seeds

from dataset import SourceDataset, TargetDataset, split_dataset


# ------ losses/helpers ------
def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)
    return torch.sum(p * torch.log(p / q), dim=-1)

def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd

def jensen_shannon_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    jsd = torch.clamp(jensen_shannon_divergence(p, q), min=0.0)
    return torch.sqrt(jsd)

def focal_loss(
    inputs: torch.Tensor, targets: torch.Tensor,
    alpha: torch.Tensor = None, gamma: float = 2.0,
    reduction: str = 'mean',
    weights: torch.Tensor = None) -> torch.Tensor:

    ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=weights)
    pt = torch.exp(-ce_loss)

    if alpha is not None:
        # alpha is [C], targets is [N]
        alpha_weights = alpha[targets]
        fl = alpha_weights * (1 - pt) ** gamma * ce_loss
    else:
        fl = (1 - pt) ** gamma * ce_loss

    if reduction == 'mean':
        return fl.mean()
    elif reduction == 'sum':
        return fl.sum()
    else:
        return fl

def sinkhorn_loss(
    x: torch.Tensor, y: torch.Tensor,
    blur: float,) -> torch.Tensor:
    """returns a callable loss"""

    loss = SamplesLoss("sinkhorn", blur=blur, scaling=0.9)
    return loss(x, y)

# ------ data / model loaders ------
def load_data(config: dict):
    data_root = config["data"]["data_root"]
    val_size = config["data"]["val_size"]
    batch_size = config["parameters"]["batch_size"]
    normalize = bool(config["data"]["normalize"])
    input_size = tuple(config["model"]["input_size"])

    src_dataset = SourceDataset(data_root=data_root, split="full")
    tgt_dataset = TargetDataset(data_root=data_root, split="full")

    if normalize:
        # compute the mean and std of the dataset
        mean, std = compute_dataset_mean_std(src_dataset)
    else:
        # use the default mean and std
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    
    # apply augmentation to the train set
    train_transform = transforms.Compose(
        [
            transforms.Resize(input_size[0]),
            transforms.RandomRotation(180),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(input_size[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    source_train, source_val = split_dataset(
        src_dataset,
        val_size=val_size,
        train_transform=train_transform,
        val_transform=val_transform,
        seed=config["seed"],
    )
    target_train, target_val = split_dataset(
        tgt_dataset,
        val_size=val_size,
        train_transform=train_transform,
        val_transform=val_transform,
        seed=config["seed"],
    )

    src_train_dataloader = DataLoader(source_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    src_val_dataloader = DataLoader(source_val, batch_size=batch_size, shuffle=False, pin_memory=True)
    tgt_train_dataloader = DataLoader(target_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    tgt_val_dataloader = DataLoader(target_val, batch_size=batch_size, shuffle=False, pin_memory=True)

    return src_train_dataloader, src_val_dataloader, tgt_train_dataloader, tgt_val_dataloader

def load_model(config: dict):
    model_name = config["model"]["name"]
    input_size = tuple(config["model"]["input_size"])
    device = config["device"]
    N = config["model"].get("N", 4)
    dihedral = config["model"].get("dihedral", True)
    lr = float(config["parameters"]["lr"])
    weight_decay = float(config["parameters"]["weight_decay"])
    milestones = config["parameters"]["milestones"]
    lr_decay = config["parameters"]["lr_decay"]

    if model_name == "cnn":
        model = cnn_nebula_galaxy(input_size=input_size).to(device)
    elif model_name == "enn":
        model = enn_nebula_galaxy(input_size=input_size, N=N, dihedral=dihedral).to(device)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay)

    return model, optimizer, scheduler

# ------ train loop ------
def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    target_dataloader: DataLoader,
    target_val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    model_name: str,
    scheduler: optim.lr_scheduler = None,
    epochs: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints",
    early_stopping_patience: int = 10,
    report_interval: int = 1,
    warmup: int = 0,
    lambda_DA: float = 0.005,
    class_weights: torch.Tensor | None = None,
    alpha_weights: torch.Tensor | None = None,
    loss_type: str = "cross_entropy",
):
    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    best_val_acc = 0.0
    best_total_val_loss = float("inf")
    best_classification_loss = float("inf")
    best_DA_loss = float("inf")
    no_improvement_count = 0
    losses, steps = [], []
    train_classification_losses, train_DA_losses = [], []
    val_losses, val_classification_losses, val_DA_losses = [], [], []
    val_accs = []
    max_distances, js_distances, blur_vals = [], [], []

    if class_weights is not None:
        class_weights = class_weights.to(device)
    if alpha_weights is not None:
        alpha_weights = alpha_weights.to(device)

    num_train_steps = min(len(train_dataloader), len(target_dataloader))
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        classification_losses, DA_losses = [], []

        it = enumerate(zip(train_dataloader, target_dataloader))
        # wrap with tqdm sized to minimum of dataloaders
        pbar = tqdm(range(num_train_steps), desc=f"Epoch {epoch+1}/{epochs}", unit="it")
        for step_idx in pbar:
            try:
                (source_batch, target_batch) = next(it)[1]
            except StopIteration:
                break

            source_inputs, source_outputs = source_batch
            source_inputs, source_outputs = source_inputs.to(device).float(), source_outputs.to(device)
            target_inputs, _ = target_batch
            target_inputs = target_inputs.to(device).float()

            optimizer.zero_grad()

            if epoch < warmup:
                _, model_outputs = model(source_inputs)
                if loss_type == "focal":
                    classification_loss = focal_loss(model_outputs, source_outputs, alpha=alpha_weights, weights=class_weights)
                else:
                    classification_loss = F.cross_entropy(model_outputs, source_outputs, weight=class_weights)
                loss = classification_loss
                DA_loss = None
            else:
                concatenated_inputs = torch.cat((source_inputs, target_inputs), dim=0)
                batch_size = source_inputs.size(0)

                features, model_outputs = model(concatenated_inputs)
                source_features = features[:batch_size]
                target_features = features[batch_size:]
                source_model_outputs = model_outputs[:batch_size]

                if loss_type == "focal":
                    classification_loss = focal_loss(source_model_outputs, source_outputs, alpha=alpha_weights, weights=class_weights)
                else:
                    classification_loss = F.cross_entropy(source_model_outputs, source_outputs, weight=class_weights)

                # pairwise distances
                pairwise_distances = torch.cdist(source_features, target_features, p=2)
                flattened_distances = pairwise_distances.view(-1)
                if flattened_distances.numel() == 0:
                    max_distance = torch.tensor(0.0, device=device)
                else:
                    max_distance = torch.max(flattened_distances)
                max_distances.append(max_distance.detach().cpu().item())

                # make sizes equal for JS and DA
                sf, tf = source_features, target_features
                if sf.size(0) != tf.size(0):
                    if sf.size(0) < tf.size(0):
                        idx = torch.randperm(tf.size(0), device=tf.device)[: sf.size(0)]
                        tf = tf[idx]
                    else:
                        idx = torch.randperm(sf.size(0), device=sf.device)[: tf.size(0)]
                        sf = sf[idx]
                sf_p = torch.softmax(sf, dim=-1)
                tf_p = torch.softmax(tf, dim=-1)
                jsd_val = jensen_shannon_distance(sf_p, tf_p)
                js_distances.append(float(jsd_val.nanmean().item()) if torch.isfinite(jsd_val).any() else 0.0)

                dynamic_blur_val = float(0.05 * max_distance.detach().cpu().item())
                blur_vals.append(dynamic_blur_val)

                # prepare DA loss inputs (again ensure sizes match)
                sf_da, tf_da = source_features, target_features
                if sf_da.size(0) != tf_da.size(0):
                    if sf_da.size(0) < tf_da.size(0):
                        idx = torch.randperm(tf_da.size(0), device=tf_da.device)[: sf_da.size(0)]
                        tf_da = tf_da[idx]
                    else:
                        idx = torch.randperm(sf_da.size(0), device=sf_da.device)[: tf_da.size(0)]
                        sf_da = sf_da[idx]

                DA_loss = sinkhorn_loss(sf_da, tf_da, blur=max(dynamic_blur_val, 0.01))
                
                # ramp for first 5 epochs after warmup
                ramp_steps = 5.0
                lambda_DA_ramped = lambda_DA if epoch - warmup + 1 >= ramp_steps else lambda_DA * max(0.0, float(epoch + 1 - warmup) / ramp_steps)
                loss = classification_loss + lambda_DA_ramped * DA_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
            classification_losses.append(classification_loss.item())
            if epoch >= warmup and DA_loss is not None:
                DA_losses.append(DA_loss.item())

            pbar.set_postfix({"train_loss": f"{(train_loss/(step_idx+1)):.4f}"})

        # epoch-level bookkeeping
        train_loss /= max(len(train_dataloader), 1)
        train_classification_loss = float(np.mean(classification_losses)) if len(classification_losses) > 0 else 0.0
        train_DA_loss = float(np.mean(DA_losses)) if len(DA_losses) > 0 else 0.0

        losses.append(train_loss)
        train_classification_losses.append(train_classification_loss)
        train_DA_losses.append(train_DA_loss)
        steps.append(epoch + 1)

        logger.info(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.6f}")
        if epoch < warmup:
            logger.info(f"Epoch: {epoch + 1}, Classification Loss: {train_classification_loss:.6f}")
        else:
            logger.info(f"Epoch: {epoch + 1}, Classification Loss: {train_classification_loss:.6f}, DA Loss: {train_DA_loss:.6f}")

        # periodic check for collapse
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_dataloader))
                sample_inputs, _ = sample_batch
                sample_inputs = sample_inputs.to(device).float()
                _, sample_outputs = model(sample_inputs)
                sample_preds = torch.softmax(sample_outputs, dim=1)
                pred_classes = torch.argmax(sample_preds, dim=1)
                unique_preds, counts = torch.unique(pred_classes, return_counts=True)
                logger.info(f"Epoch {epoch + 1} - Predicted class distribution: {dict(zip(unique_preds.cpu().numpy(), counts.cpu().numpy()))}")
            model.train()

        if scheduler is not None:
            scheduler.step()

        # validation & early stopping
        if (epoch + 1) % report_interval == 0:
            model.eval()
            source_correct, source_total = 0, 0
            val_loss = 0.0
            val_classification_loss = 0.0
            val_DA_loss = 0.0

            with torch.no_grad():
                for (batch, target_batch) in zip(val_dataloader, target_val_dataloader):
                    source_inputs, source_outputs = batch
                    source_inputs, source_outputs = source_inputs.to(device).float(), source_outputs.to(device)
                    target_inputs, target_outputs = target_batch
                    target_inputs, target_outputs = target_inputs.to(device).float(), target_outputs.to(device)

                    if epoch < warmup:
                        _, source_preds = model(source_inputs)
                        if loss_type == "focal":
                            classification_loss_ = focal_loss(source_preds, source_outputs, alpha=alpha_weights, weights=class_weights)
                        else:
                            classification_loss_ = F.cross_entropy(source_preds, source_outputs, weight=class_weights)
                        combined_loss = classification_loss_
                        DA_loss_ = torch.tensor(0.0, device=device)
                    else:
                        concatenated_inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        batch_size = source_inputs.size(0)

                        features, preds = model(concatenated_inputs)
                        source_features = features[:batch_size]
                        target_features = features[batch_size:]
                        source_preds = preds[:batch_size]

                        if loss_type == "focal":
                            classification_loss_ = focal_loss(source_preds, source_outputs, alpha=alpha_weights, weights=class_weights)
                        else:
                            classification_loss_ = F.cross_entropy(source_preds, source_outputs, weight=class_weights)

                        # pairwise distances -> DA loss
                        pairwise_distances = torch.cdist(source_features, target_features, p=2)
                        flattened_distances = pairwise_distances.view(-1)
                        max_distance = torch.max(flattened_distances) if flattened_distances.numel() > 0 else torch.tensor(0.0, device=device)
                        dynamic_blur_val = float(0.05 * max_distance.detach().cpu().item())

                        sf, tf = source_features, target_features
                        if sf.size(0) != tf.size(0):
                            if sf.size(0) < tf.size(0):
                                idx = torch.randperm(tf.size(0), device=tf.device)[: sf.size(0)]
                                tf = tf[idx]
                            else:
                                idx = torch.randperm(sf.size(0), device=sf.device)[: tf.size(0)]
                                sf = sf[idx]

                        DA_loss_ = sinkhorn_loss(sf, tf, blur=max(dynamic_blur_val, 0.01))

                        lambda_DA = 0.005
                        lambda_DA_ramped = min(lambda_DA, lambda_DA * float(epoch + 1 - warmup) / 5.0) if epoch >= warmup else 0.0
                        combined_loss = classification_loss_ + lambda_DA_ramped * DA_loss_

                    val_loss += combined_loss.item()
                    val_classification_loss += classification_loss_.item()
                    if epoch >= warmup:
                        val_DA_loss += DA_loss_.item()

                    _, source_predicted = torch.max(source_preds.data, 1)
                    source_total += source_outputs.size(0)
                    source_correct += (source_predicted == source_outputs).sum().item()

            source_val_acc = 100.0 * source_correct / max(source_total, 1)
            val_loss /= max(len(val_dataloader), 1)
            val_classification_loss /= max(len(val_dataloader), 1)
            if epoch >= warmup:
                val_DA_loss /= max(len(val_dataloader), 1)

            val_losses.append(val_loss)
            val_classification_losses.append(val_classification_loss)
            val_DA_losses.append(val_DA_loss)
            val_accs.append(source_val_acc)

            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch: {epoch + 1}, Total Validation Loss: {val_loss:.6f}, Source Validation Accuracy: {source_val_acc:.2f}%, Learning rate: {lr}"
            )

            # save based on criteria
            if val_loss < best_total_val_loss and epoch >= warmup:
                best_total_val_loss = val_loss
                best_total_val_epoch = epoch + 1
                path = os.path.join(save_dir, "best_model_total_val_loss.pt")
                torch.save(model.eval().state_dict(), path)
                logger.info(f"Saved best total validation loss model at epoch {best_total_val_epoch}")

            if source_val_acc >= best_val_acc:
                best_val_acc = source_val_acc
                best_val_epoch = epoch + 1
                path = os.path.join(save_dir, "best_model_val_acc.pt")
                torch.save(model.eval().state_dict(), path)
                logger.info(f"Saved best validation accuracy model at epoch {best_val_epoch}")

            if val_classification_loss <= best_classification_loss and epoch >= warmup:
                best_classification_loss = val_classification_loss
                best_classification_loss_epoch = epoch + 1
                path = os.path.join(save_dir, "best_model_classification_loss.pt")
                torch.save(model.eval().state_dict(), path)
                logger.info(f"Saved lowest classification loss model at epoch {best_classification_loss_epoch}")

            if val_DA_loss <= best_DA_loss and epoch >= warmup:
                best_DA_loss = val_DA_loss
                best_DA_epoch = epoch + 1
                path = os.path.join(save_dir, "best_model_DA_loss.pt")
                torch.save(model.eval().state_dict(), path)
                logger.info(f"Saved lowest DA loss model at epoch {best_DA_epoch}")

            # early stopping on accuracy plateaus
            if 'best_val_epoch' in locals():
                improved = (source_val_acc >= best_val_acc)
            else:
                improved = True
            if not improved:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping after {early_stopping_patience} epochs without improvement in accuracy.")
                break

    # final save
    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.eval().state_dict(), final_path)

    # save losses / metadata
    loss_dir = os.path.join(save_dir, "losses")
    os.makedirs(loss_dir, exist_ok=True)
    np.save(os.path.join(loss_dir, f"losses-{model_name}.npy"), np.array(losses))
    np.save(os.path.join(loss_dir, f"train_classification_losses-{model_name}.npy"), np.array(train_classification_losses))
    np.save(os.path.join(loss_dir, f"train_DA_losses-{model_name}.npy"), np.array(train_DA_losses))
    np.save(os.path.join(loss_dir, f"val_losses-{model_name}.npy"), np.array(val_losses))
    np.save(os.path.join(loss_dir, f"val_classification_losses-{model_name}.npy"), np.array(val_classification_losses))
    np.save(os.path.join(loss_dir, f"val_DA_losses-{model_name}.npy"), np.array(val_DA_losses))
    np.save(os.path.join(loss_dir, f"val_accs-{model_name}.npy"), np.array(val_accs))
    np.save(os.path.join(loss_dir, f"steps-{model_name}.npy"), np.array(steps))
    np.save(os.path.join(loss_dir, f"max_distances-{model_name}.npy"), np.array(max_distances))
    np.save(os.path.join(loss_dir, f"blur_vals-{model_name}.npy"), np.array(blur_vals))
    np.save(os.path.join(loss_dir, f"js_distances-{model_name}.npy"), np.array(js_distances))
    # loss plots
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle(f'Training Analysis - {model_name.upper()}', fontsize=24, fontweight='bold', y=0.98)
    
    colors = {
        'total_loss': '#1f77b4',
        'classification': '#ff7f0e', 
        'da_loss': '#2ca02c',
        'accuracy': '#9467bd',
        'js_distance': '#d62728',
        'blur': '#9467bd',
        'distance': '#8c564b'
    }
    
    # 1. Training Losses
    ax = axes[0, 0]
    ax.plot(steps, losses, label="Total Loss", linewidth=3, color=colors['total_loss'], alpha=0.9)
    ax.plot(steps, train_classification_losses, label="Classification Loss", linewidth=3, color=colors['classification'], alpha=0.9)
    if len(train_DA_losses) > 0 and np.any(np.array(train_DA_losses) > 0):
        ax.plot(steps, train_DA_losses, label="Domain Adaptation Loss", linewidth=3, color=colors['da_loss'], alpha=0.9)
    if warmup > 0:
        ax.axvline(x=warmup, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Warmup End (epoch {warmup})')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Losses', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 2. Validation Losses/Accuracy
    ax = axes[0, 1]
    validation_steps = steps[::max(1, report_interval)][:len(val_losses)]
    ax.plot(validation_steps, val_losses, label="Total Val Loss", linewidth=3, color=colors['total_loss'], alpha=0.9)
    ax.plot(validation_steps, val_classification_losses, label="Val Classification Loss", linewidth=3, color=colors['classification'], alpha=0.9)
    if len(val_DA_losses) > 0 and np.any(np.array(val_DA_losses) > 0):
        ax.plot(validation_steps, val_DA_losses, label="Val DA Loss", linewidth=3, color=colors['da_loss'], alpha=0.9)
    if warmup > 0:
        ax.axvline(x=warmup, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Warmup End')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Validation Losses & Accuracy', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=10)
    if val_accs is not None and len(val_accs) > 0:
        ax2 = ax.twinx()
        ax2.plot(validation_steps[:len(val_accs)], val_accs, label='Val Accuracy (%)', 
                 color=colors['accuracy'], linewidth=3, linestyle='--', alpha=0.9)
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color=colors['accuracy'])
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor=colors['accuracy'], labelsize=10)
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, framealpha=0.9, loc='upper right')
    else:
        ax.legend(fontsize=10, framealpha=0.9)
    
    # 3. Jensen-Shannon Distance
    ax = axes[0, 2]
    if js_distances is not None and len(js_distances) > 0:
        js_array = np.array(js_distances)
        valid_mask = ~np.isnan(js_array) & np.isfinite(js_array)
        if np.any(valid_mask):
            valid_steps = np.array(steps)[valid_mask]
            valid_js = js_array[valid_mask]
            ax.plot(valid_steps, valid_js, linewidth=3, color=colors['js_distance'], alpha=0.9)
            if warmup > 0:
                ax.axvline(x=warmup, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Jensen-Shannon Distance', fontsize=12, fontweight='bold')
            ax.set_title('Domain Similarity\n(Lower = Better Adaptation)', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=10)
        else:
            ax.text(0.5, 0.5, 'No valid JS Distance data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, style='italic')
            ax.set_title('Jensen-Shannon Distance', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'JS Distance data\nnot available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12, style='italic')
        ax.set_title('Jensen-Shannon Distance', fontsize=14, fontweight='bold')
    
    # 4. Training Progress Summary
    ax = axes[1, 0]
    data = [
        ['Metric', 'Value'],
        ['Best Val Acc', f'{best_val_acc:.2f}%' if 'best_val_acc' in locals() else 'N/A'],
        ['Best Val Loss', f'{best_total_val_loss:.6f}' if 'best_total_val_loss' in locals() else 'N/A'],
        ['Final Train Loss', f'{losses[-1]:.6f}' if len(losses) > 0 else 'N/A'],
        ['Warmup Epochs', str(warmup)],
        ['Total Epochs', str(len(steps))]
    ]
    
    table = ax.table(cellText=data[1:], colLabels=data[0], 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(data)):
        for j in range(len(data[0])):
            if i == 0:
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#E8F5E8')
    
    ax.set_title('Training Summary', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 5. Sinkhorn Blur Parameter
    ax = axes[1, 1]
    if blur_vals is not None and len(blur_vals) > 0:
        # Convert to numpy array and handle any invalid values
        blur_array = np.array(blur_vals)
        valid_mask = ~np.isnan(blur_array) & np.isfinite(blur_array)
        
        if np.any(valid_mask):
            valid_steps = np.array(steps)[valid_mask]
            valid_blur = blur_array[valid_mask]
            
            ax.plot(valid_steps, valid_blur, linewidth=3, color=colors['blur'], alpha=0.9)
            ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Min Blur Threshold')
            
            if warmup > 0:
                ax.axvline(x=warmup, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Blur Value', fontsize=12, fontweight='bold')
            ax.set_title('Sinkhorn Blur Parameter', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.legend(fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=10)
        else:
            ax.text(0.5, 0.5, 'No valid blur values data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, style='italic')
            ax.set_title('Sinkhorn Blur Parameter', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Blur values data\nnot available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12, style='italic')
        ax.set_title('Sinkhorn Blur Parameter', fontsize=14, fontweight='bold')
    
    # 6. Max Feature Distance
    ax = axes[1, 2]
    if max_distances is not None and len(max_distances) > 0:
        dist_array = np.array(max_distances)
        valid_mask = ~np.isnan(dist_array) & np.isfinite(dist_array)
        
        if np.any(valid_mask):
            valid_steps = np.array(steps)[valid_mask]
            valid_dist = dist_array[valid_mask]
            
            ax.plot(valid_steps, valid_dist, linewidth=3, color=colors['distance'], alpha=0.9)
            
            if warmup > 0:
                ax.axvline(x=warmup, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel('Max Pairwise Distance', fontsize=12, fontweight='bold')
            ax.set_title('Feature Space Distances', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=10)
        else:
            ax.text(0.5, 0.5, 'No valid distance data', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, style='italic')
            ax.set_title('Feature Space Distances', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Distance data\nnot available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12, style='italic')
        ax.set_title('Feature Space Distances', fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(save_dir, f"training_plots_{model_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Training plots saved to: {plot_path}")
    return {
        "best_val_epoch": locals().get("best_val_epoch", None),
        "best_val_acc": best_val_acc,
        "best_total_val_epoch": locals().get("best_total_val_epoch", None),
        "best_total_val_loss": best_total_val_loss,
        "best_classification_loss_epoch": locals().get("best_classification_loss_epoch", None),
        "best_classification_loss": best_classification_loss,
        "best_DA_epoch": locals().get("best_DA_epoch", None),
        "best_DA_loss": best_DA_loss,
        "final_loss": losses[-1] if len(losses) > 0 else None,
    }

# ---------------------------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Train with Sinkhorn domain adaptation")
    parser.add_argument("--config", metavar="config", required=True, help="path to the config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_all_seeds(config.get("seed", 42))
    config["device"] = device
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = (
        f"{config['output_dir'].rstrip('/')}/"
        f"{config['model']['name']}"
        f"_N{config['model']['N']}"
        f"_ep{config['parameters']['epochs']}"
        f"_seed{config['seed']}"
        f"_{config['parameters']['loss_type']}"
        f"_{timestamp}"
    )

    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(str(save_dir + "logs.log"))
    src_train_dataloader, src_val_dataloader, tgt_train_dataloader, tgt_val_dataloader = load_data(config)
    model, optimizer, scheduler = load_model(config)
    if config["data"].get("use_class_weights", False):
        src_full = SourceDataset(
            data_root=config["data"]["data_root"],
            split="full",
        )
        src_labels = [sample["label"] for sample in src_full.samples]
        class_weights = compute_class_weights(src_labels, method="balanced", device=device)
    else:
        class_weights = None

    # loss type and alpha
    loss_type = config["parameters"].get("loss_type", "cross_entropy")
    alpha_weights = None
    if loss_type == "focal":
        # If config provides alpha use it; else use class_weights normalized
        if "alpha" in config["parameters"] and config["parameters"]["alpha"] is not None:
            alpha_arr = np.array(config["parameters"]["alpha"], dtype=float)
            alpha_weights = torch.tensor(alpha_arr, dtype=torch.float32, device=device)
        elif class_weights is not None:
            # normalize class_weights to sum 1 as alpha prior
            alpha_weights = class_weights.to(device)
            alpha_weights = alpha_weights / torch.sum(alpha_weights)
        else:
            # uniform alpha
            # need number of classes -> infer from dataset
            num_classes = config["model"].get("num_classes", 3)
            if num_classes is None:
                s = SourceDataset(data_root=config["data"]["data_root"], split="full")
                labels = [sample["label"] for sample in s.samples]
                num_classes = int(max(labels) + 1)
            alpha_weights = torch.ones(num_classes, device=device) / float(num_classes)

    # run training
    results = train(
        model=model,
        train_dataloader=src_train_dataloader,
        val_dataloader=src_val_dataloader,
        target_dataloader=tgt_train_dataloader,
        target_val_dataloader=tgt_val_dataloader,
        optimizer=optimizer,
        model_name=config["model"]["name"],
        scheduler=scheduler,
        epochs=config["parameters"]["epochs"],
        device=device,
        save_dir=save_dir,
        early_stopping_patience=config["parameters"]["early_stopping"],
        report_interval=config["parameters"]["report_interval"],
        warmup=config["parameters"].get("warmup", 0),
        lambda_DA=config["parameters"].get("lambda_DA", 0.005),
        class_weights=class_weights,
        alpha_weights=alpha_weights,
        loss_type=loss_type,
    )

    logger.info("Training Done")
    config.update(results)
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)
