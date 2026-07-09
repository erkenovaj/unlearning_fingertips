#!/usr/bin/env python3
"""Visualization and logging utilities for unlearning-trace detection experiments.

Provides training curves, TensorBoard logging, CSV experiment tracking,
PCA/t-SNE activation plots, and score-distribution histograms.

Mirrors the style of reproduction_xpu.py (json, print, io, numpy, torch).
"""

import csv
import io
import json
import os
from datetime import datetime

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 1. MLP training curves — per-epoch metrics
# --------------------------------------------------------------------------- #
class TrainingLogger:
    """Captures per-epoch train/val loss & accuracy during MLP training.

    Usage
    -----
        tlog = TrainingLogger()
        # inside training loop:
        tlog.log_epoch(train_loss, train_acc, val_loss, val_acc)
        tlog.save("metrics.json")
        tlog.plot("metrics.png")
    """

    def __init__(self):
        self.epochs = []

    def log_epoch(self, train_loss, train_acc, val_loss, val_acc):
        self.epochs.append({
            "epoch": len(self.epochs),
            "train_loss": round(float(train_loss), 6),
            "train_acc":  round(float(train_acc),  4),
            "val_loss":   round(float(val_loss),   6),
            "val_acc":    round(float(val_acc),    4),
        })

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with io.open(path, "w", encoding="utf-8") as f:
            json.dump(self.epochs, f, indent=2)
        print(f"  [viz] training metrics -> {path}")

    def plot(self, path):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  [viz] install matplotlib for plots: pip install matplotlib")
            return
        if not self.epochs:
            print("  [viz] no data to plot")
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        epochs = [e["epoch"] for e in self.epochs]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(epochs, [e["train_loss"] for e in self.epochs], label="train")
        ax1.plot(epochs, [e["val_loss"]   for e in self.epochs], label="val")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend(); ax1.set_title("Loss")
        ax2.plot(epochs, [e["train_acc"]  for e in self.epochs], label="train")
        ax2.plot(epochs, [e["val_acc"]    for e in self.epochs], label="val")
        ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.legend(); ax2.set_title("Accuracy")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [viz] training plot -> {path}")


# --------------------------------------------------------------------------- #
# 2. TensorBoard logging
# --------------------------------------------------------------------------- #
def _summary_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir)
    except ImportError:
        print("  [viz] install tensorboard for TensorBoard logs: pip install tensorboard")
        return None


def log_to_tensorboard(log_dir, metrics_list, tag="mlp"):
    """Write per-epoch metrics to TensorBoard.

    Parameters
    ----------
    metrics_list : list[dict]
        Each dict must have keys: epoch, train_loss, train_acc, val_loss, val_acc.
    """
    writer = _summary_writer(log_dir)
    if writer is None:
        return
    for m in metrics_list:
        writer.add_scalar(f"{tag}/train_loss", m["train_loss"], m["epoch"])
        writer.add_scalar(f"{tag}/train_acc",  m["train_acc"],  m["epoch"])
        writer.add_scalar(f"{tag}/val_loss",   m["val_loss"],   m["epoch"])
        writer.add_scalar(f"{tag}/val_acc",    m["val_acc"],    m["epoch"])
    writer.close()
    print(f"  [viz] TensorBoard logs -> {log_dir}")


# --------------------------------------------------------------------------- #
# 3. CSV experiment tracker
# --------------------------------------------------------------------------- #
EXPERIMENTS_CSV_HEADER = [
    "timestamp", "model", "unlearn", "dataset", "feature",
    "num_samples", "normalize", "mix_train", "test_accuracy",
]


def log_experiment_to_csv(csv_path, config, test_accuracy):
    """Append one row to a CSV tracking all experiment runs."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    row = {
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":         config.get("model", ""),
        "unlearn":       config.get("unlearn", ""),
        "dataset":       config.get("dataset", ""),
        "feature":       config.get("feature", ""),
        "num_samples":   config.get("num_samples", ""),
        "normalize":     config.get("normalize", ""),
        "mix_train":     config.get("mix_train", ""),
        "test_accuracy": f"{test_accuracy:.4f}",
    }
    exists = os.path.exists(csv_path)
    with io.open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENTS_CSV_HEADER)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  [viz] experiment logged -> {csv_path}")


# --------------------------------------------------------------------------- #
# 4. Activation visualizations (PCA + t-SNE)
# --------------------------------------------------------------------------- #
def plot_activations(X, labels, output_path, method="both", title=None):
    """PCA and/or t-SNE scatter plot of pre-logit activations.

    Parameters
    ----------
    X : np.ndarray  (n_samples, n_features)
    labels : array-like  (n_samples,)  — 0 = original, 1 = unlearned
    method : "pca" | "tsne" | "both"
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as e:
        print(f"  [viz] install dependencies: pip install matplotlib scikit-learn  ({e})")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    label_colors = ["steelblue", "crimson"]
    label_names = ["original", "unlearned"]
    if title is None:
        title = "Pre-logit activations"

    n_methods = {"pca": 1, "tsne": 1, "both": 2}[method]
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    def _scatter(ax, emb, method_name):
        for lbl in (0, 1):
            mask = np.asarray(labels) == lbl
            ax.scatter(emb[mask, 0], emb[mask, 1], c=label_colors[lbl],
                       label=label_names[lbl], alpha=0.6, s=8, edgecolors="none")
        ax.set_title(f"{title} ({method_name})")
        ax.legend(markerscale=4)

    idx = 0
    if method in ("pca", "both"):
        emb = PCA(n_components=2, random_state=42).fit_transform(X)
        _scatter(axes[idx], emb, "PCA")
        idx += 1
    if method in ("tsne", "both"):
        emb = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1)).fit_transform(X)
        _scatter(axes[idx], emb, "t-SNE")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [viz] activation plot -> {output_path}")


# --------------------------------------------------------------------------- #
# 5. Score distribution histogram
# --------------------------------------------------------------------------- #
def plot_score_distribution(scores, labels, output_path, title=None):
    """Histogram of classifier logits/scores split by ground-truth label."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [viz] install matplotlib for plots: pip install matplotlib")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()
    if title is None:
        title = "Classifier score distribution"

    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, color, name in ((0, "steelblue", "original"), (1, "crimson", "unlearned")):
        subset = scores[labels == lbl]
        ax.hist(subset, bins=40, alpha=0.5, color=color, label=name, density=True)
    ax.set_xlabel("score"); ax.set_ylabel("density"); ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [viz] score distribution -> {output_path}")
