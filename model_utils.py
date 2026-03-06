from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import warnings
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


@dataclass
class TrainResult:
    model: nn.Module
    classes: List[str]
    val_accuracy: float



def build_transfer_model(num_classes: int) -> nn.Module:
    try:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as exc:
        warnings.warn(
            f"No se pudieron descargar/cargar pesos preentrenados ({exc}). Se usará ResNet18 sin preentrenar.",
            RuntimeWarning,
        )
        model = models.resnet18(weights=None)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model



def load_dataset(dataset_dir: Path) -> datasets.ImageFolder:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {dataset_dir}")
    ds = datasets.ImageFolder(root=str(dataset_dir), transform=IMAGENET_TRANSFORM)
    if len(ds.classes) < 2:
        raise ValueError("Se necesitan al menos 2 clases para entrenar.")
    if len(ds) < 10:
        raise ValueError("Dataset muy pequeño; agrega más imágenes.")
    return ds



def train_transfer(
    dataset_dir: Path,
    epochs: int = 5,
    lr: float = 5e-4,
    batch_size: int = 32,
    device: str | None = None,
) -> TrainResult:
    ds = load_dataset(dataset_dir)
    classes = ds.classes

    val_size = max(1, int(0.2 * len(ds)))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_transfer_model(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    val_acc = evaluate_accuracy(model, val_loader, device)
    return TrainResult(model=model, classes=classes, val_accuracy=val_acc)



def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    return correct / max(total, 1)



def predict_topk(model: nn.Module, image_path: Path, classes: List[str], topk: int = 3) -> List[Tuple[str, float]]:
    model.eval()
    device = next(model.parameters()).device

    pil = datasets.folder.default_loader(str(image_path))
    x = IMAGENET_TRANSFORM(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    k = min(topk, len(classes))
    values, indices = torch.topk(probs, k=k)
    return [(classes[i], float(v)) for v, i in zip(values.cpu().tolist(), indices.cpu().tolist())]
