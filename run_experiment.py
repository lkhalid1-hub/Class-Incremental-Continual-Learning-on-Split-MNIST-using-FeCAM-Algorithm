"""Reproducible FeCAM and Experience Replay experiment on Split-MNIST.

The script reproduces the workflow in ``notebooks/cl-notebook.ipynb``. It
downloads MNIST and Fashion-MNIST on first use and writes all generated files
under the selected output directory.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


SEED = 42
CLASS_SPLITS = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
TASK_LABELS = ["(0,1)", "(2,3)", "(4,5)", "(6,7)", "(8,9)"]
NUM_CLASSES = 10
FEATURE_DIM = 128
TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def set_seed(seed: int = SEED) -> None:
    """Seed the experiment's random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_into_tasks(dataset):
    """Divide an MNIST dataset into the five two-class Split-MNIST tasks."""
    targets = dataset.targets.numpy()
    return [
        Subset(dataset, np.concatenate([np.flatnonzero(targets == c) for c in pair]).tolist())
        for pair in CLASS_SPLITS
    ]


class FeatureExtractor(nn.Module):
    """Frozen convolutional encoder used to obtain features for FeCAM."""

    def __init__(self, feature_dim: int = FEATURE_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(128, feature_dim)

    def forward(self, images):
        return self.projection(self.features(images).flatten(1))


def pretrain_feature_extractor(data_dir: Path, checkpoint: Path, device, epochs: int):
    """Train the FeCAM encoder on Fashion-MNIST, or reuse an existing checkpoint."""
    encoder = FeatureExtractor().to(device)
    if checkpoint.exists():
        encoder.load_state_dict(torch.load(checkpoint, map_location=device))
        encoder.eval()
        print(f"Reusing feature extractor: {checkpoint}")
        return encoder

    train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=TRANSFORM)
    loader = DataLoader(train, batch_size=128, shuffle=True)
    head = nn.Linear(FEATURE_DIM, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam([*encoder.parameters(), *head.parameters()], lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Pre-training the feature extractor on Fashion-MNIST ({epochs} epochs)")
    encoder.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_fn(head(encoder(images)), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  epoch {epoch:2d}/{epochs}: loss={running_loss:.3f}")

    torch.save(encoder.state_dict(), checkpoint)
    encoder.eval()
    return encoder


class FeCAM:
    """Class-wise Gaussian classifier with a shrinkage-regularised covariance."""

    def __init__(self, shrinkage: float = 1.0, cov_type: str = "full"):
        if cov_type not in {"full", "diagonal", "identity"}:
            raise ValueError("cov_type must be full, diagonal, or identity")
        self.shrinkage = shrinkage
        self.cov_type = cov_type
        self.means, self.covs = {}, {}

    def _estimate_covariance(self, features):
        dim = features.shape[1]
        if self.cov_type == "identity":
            return np.eye(dim)
        covariance = np.diag(features.var(axis=0)) if self.cov_type == "diagonal" else np.cov(features.T)
        return covariance + self.shrinkage * np.mean(np.diag(covariance)) * np.eye(dim)

    def update(self, features, labels):
        for label in np.unique(labels):
            class_features = features[labels == label]
            self.means[int(label)] = class_features.mean(axis=0)
            self.covs[int(label)] = self._estimate_covariance(class_features)

    def predict(self, features):
        classes = sorted(self.means)
        distances = []
        for label in classes:
            difference = features - self.means[label]
            precision = np.linalg.pinv(self.covs[label])
            distances.append(np.sum((difference @ precision) * difference, axis=1))
        return np.array(classes)[np.argmin(np.stack(distances), axis=0)]


@torch.no_grad()
def extract_features(encoder, dataset, device):
    features, labels = [], []
    for images, targets in DataLoader(dataset, batch_size=256):
        features.append(encoder(images.to(device)).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def run_fecam(encoder, train_tasks, test_tasks, device, cov_type="full", verbose=True):
    model = FeCAM(cov_type=cov_type)
    accuracies, matrices = [], []
    for task_id, task in enumerate(train_tasks):
        features, labels = extract_features(encoder, task, device)
        model.update(features, labels)
        predictions, all_labels = [], []
        for seen_task in range(task_id + 1):
            features, labels = extract_features(encoder, test_tasks[seen_task], device)
            predictions.extend(model.predict(features))
            all_labels.extend(labels)
        predictions, all_labels = np.array(predictions), np.array(all_labels)
        accuracy = (predictions == all_labels).mean()
        accuracies.append(accuracy)
        matrices.append(confusion_matrix(all_labels, predictions, labels=np.arange(NUM_CLASSES)))
        if verbose:
            print(f"FeCAM experience {task_id + 1}: cumulative accuracy={accuracy:.4f}")
    return accuracies, matrices


class ReservoirBuffer:
    """Fixed-size replay memory maintained with reservoir sampling."""

    def __init__(self, capacity: int):
        self.capacity, self.images, self.labels, self.n_seen = capacity, [], [], 0

    def add_batch(self, images, labels):
        for image, label in zip(images, labels):
            self.n_seen += 1
            if len(self.images) < self.capacity:
                self.images.append(image.cpu())
                self.labels.append(label.cpu())
            else:
                index = random.randint(0, self.n_seen - 1)
                if index < self.capacity:
                    self.images[index], self.labels[index] = image.cpu(), label.cpu()

    def sample(self, batch_size: int):
        if not self.images:
            return None, None
        indices = random.sample(range(len(self.images)), min(batch_size, len(self.images)))
        return torch.stack([self.images[i] for i in indices]), torch.stack([self.labels[i] for i in indices])


class ReplayCNN(nn.Module):
    """CNN classifier trained online with replayed examples."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, images):
        return self.network(images)


def run_experience_replay(buffer_size, train_tasks, test_tasks, device, epochs):
    set_seed()
    model, buffer = ReplayCNN().to(device), ReservoirBuffer(buffer_size)
    optimizer, loss_fn = torch.optim.Adam(model.parameters(), lr=1e-3), nn.CrossEntropyLoss()
    accuracies, matrices = [], []
    for task_id, task in enumerate(train_tasks):
        loader = DataLoader(task, batch_size=64, shuffle=True)
        model.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                replay_images, replay_labels = buffer.sample(64)
                if replay_images is not None:
                    images = torch.cat([images, replay_images.to(device)])
                    labels = torch.cat([labels, replay_labels.to(device)])
                loss = loss_fn(model(images), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"ER-{buffer_size} task {task_id + 1}, epoch {epoch}/{epochs}: loss={running_loss:.3f}")
        for images, labels in loader:
            buffer.add_batch(images, labels)
        model.eval()
        predictions, all_labels = [], []
        with torch.no_grad():
            for seen_task in range(task_id + 1):
                for images, labels in DataLoader(test_tasks[seen_task], batch_size=256):
                    predictions.extend(model(images.to(device)).argmax(1).cpu().numpy())
                    all_labels.extend(labels.numpy())
        predictions, all_labels = np.array(predictions), np.array(all_labels)
        accuracy = (predictions == all_labels).mean()
        accuracies.append(accuracy)
        matrices.append(confusion_matrix(all_labels, predictions, labels=np.arange(NUM_CLASSES)))
        print(f"ER-{buffer_size} experience {task_id + 1}: cumulative accuracy={accuracy:.4f}")
    return accuracies, matrices


def plot_confusion_grid(matrices, title, cmap, path):
    figure, axes = plt.subplots(1, len(matrices), figsize=(18, 4))
    for experience, (axis, matrix) in enumerate(zip(axes, matrices), start=1):
        sns.heatmap(matrix, ax=axis, cmap=cmap, cbar=experience == len(matrices), square=True)
        axis.set(title=f"After experience {experience}", xlabel="Predicted", ylabel="True")
    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_reports(results_dir, fecam, replay_500, replay_100, ablation):
    header = "Experience".ljust(14) + "".join(f"{label:>10}" for label in TASK_LABELS) + f"{'Average':>10}"
    lines = ["Continual Learning on Split-MNIST - Accuracy Report", "=" * 55, ""]
    for name, scores in {"FeCAM": fecam, "Experience Replay (buffer 500)": replay_500, "Experience Replay (buffer 100)": replay_100}.items():
        values = np.array(scores) * 100
        lines.extend([name, "-" * len(name), header, "accuracy %".ljust(14) + "".join(f"{value:>10.2f}" for value in values) + f"{values.mean():>10.2f}", ""])
    (results_dir / "continual_learning_metrics.txt").write_text("\n".join(lines), encoding="utf-8")

    header = "cov_type".ljust(12) + "".join(f"{label:>10}" for label in TASK_LABELS) + f"{'Mean':>10}"
    lines = ["FeCAM Covariance-Model Ablation on Split-MNIST", "=" * len(header), "", header, "-" * len(header)]
    for name, scores in ablation.items():
        values = np.array(scores) * 100
        lines.append(name.ljust(12) + "".join(f"{value:>10.2f}" for value in values) + f"{values.mean():>10.2f}")
    (results_dir / "fecam_ablation.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for data, models, figures, and results.")
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--replay-epochs", type=int, default=5)
    parser.add_argument("--skip-ablation", action="store_true", help="Skip the FeCAM covariance ablation.")
    args = parser.parse_args()

    set_seed()
    root = args.output_dir.resolve()
    data_dir, models_dir = root / "data", root / "models"
    figures_dir, results_dir = root / "figures", root / "results"
    for directory in (data_dir, models_dir, figures_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = pretrain_feature_extractor(data_dir, models_dir / "fmnist_pretrained.pth", device, args.pretrain_epochs)
    encoder.eval()
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=TRANSFORM)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=TRANSFORM)
    train_tasks, test_tasks = split_into_tasks(mnist_train), split_into_tasks(mnist_test)

    fecam, fecam_matrices = run_fecam(encoder, train_tasks, test_tasks, device)
    plot_confusion_grid(fecam_matrices, "FeCAM - Confusion Matrices", "Greens", figures_dir / "fecam_confusion_matrices.png")
    ablation = {"full": fecam}
    if not args.skip_ablation:
        for covariance in ("identity", "diagonal"):
            ablation[covariance], _ = run_fecam(encoder, train_tasks, test_tasks, device, covariance, verbose=False)

    replay_500, matrices_500 = run_experience_replay(500, train_tasks, test_tasks, device, args.replay_epochs)
    replay_100, matrices_100 = run_experience_replay(100, train_tasks, test_tasks, device, args.replay_epochs)
    plot_confusion_grid(matrices_500, "Experience Replay (buffer 500) - Confusion Matrices", "Blues", figures_dir / "experience_replay_500_confusion_matrices.png")
    plot_confusion_grid(matrices_100, "Experience Replay (buffer 100) - Confusion Matrices", "Blues", figures_dir / "experience_replay_100_confusion_matrices.png")
    write_reports(results_dir, fecam, replay_500, replay_100, ablation)
    print(f"Finished. Outputs are in: {root}")


if __name__ == "__main__":
    main()
