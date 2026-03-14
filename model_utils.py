from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import torch
import torch.nn as nn
import torchvision
from PIL import Image
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


@dataclass
class LoadedCheckpoint:
    model: nn.Module
    classes: List[str]
    metadata: Dict[str, Any]


def resolve_device(device: str | None = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def checkpoint_exists(checkpoint_path: Path) -> bool:
    return Path(checkpoint_path).is_file()


def _checkpoint_metadata(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in checkpoint.items() if key != "model_state_dict"}


def _first_parameter(model: nn.Module) -> nn.Parameter:
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("El modelo no contiene parametros.")
    return parameter



def build_transfer_model(num_classes: int, use_pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    try:
        model = models.resnet18(weights=weights)
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
    seed: int = 42,
) -> TrainResult:
    ds = load_dataset(dataset_dir)
    classes = ds.classes

    val_size = max(1, int(0.2 * len(ds)))
    train_size = len(ds) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = resolve_device(device)

    model = cast(nn.Module, build_transfer_model(num_classes=len(classes)).to(device))
    optimizer = torch.optim.Adam((param for param in model.parameters() if param.requires_grad), lr=lr)

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


def save_checkpoint(
    model: nn.Module,
    classes: List[str],
    dataset_dir: Path,
    val_accuracy: float,
    epochs: int,
    output_path: Path,
    trained_at: str | None = None,
    architecture: str = "resnet18",
) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "arch": architecture,
        "classes": list(classes),
        "dataset_dir": str(dataset_dir),
        "epochs": int(epochs),
        "num_classes": len(classes),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "trained_at": trained_at or datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "training_device": _first_parameter(model).device.type,
        "val_accuracy": float(val_accuracy),
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, output_path)

    metadata = _checkpoint_metadata(checkpoint)
    metadata["checkpoint_path"] = str(output_path)
    return metadata


def load_checkpoint(checkpoint_path: Path, device: str | None = None) -> LoadedCheckpoint:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")

    device = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError("El checkpoint no tiene un formato valido.")

    classes = checkpoint.get("classes")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(classes, list) or not classes:
        raise ValueError("El checkpoint no incluye clases validas.")
    if state_dict is None:
        raise ValueError("El checkpoint no incluye los pesos del modelo.")

    model = cast(nn.Module, build_transfer_model(num_classes=len(classes), use_pretrained=False).to(device))
    model.load_state_dict(state_dict)

    metadata = _checkpoint_metadata(checkpoint)
    metadata["checkpoint_path"] = str(checkpoint_path)
    metadata["load_device"] = device
    return LoadedCheckpoint(model=model, classes=classes, metadata=metadata)



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


def _predict_topk_from_pil(
    model: nn.Module,
    pil_image: Image.Image,
    classes: List[str],
    topk: int = 3,
) -> List[Tuple[str, float]]:
    model.eval()
    device = _first_parameter(model).device

    tensor = IMAGENET_TRANSFORM(pil_image.convert("RGB"))
    assert isinstance(tensor, torch.Tensor)
    x = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    k = min(topk, len(classes))
    values, indices = torch.topk(probs, k=k)
    return [(classes[i], float(v)) for v, i in zip(values.cpu().tolist(), indices.cpu().tolist())]



def predict_topk(model: nn.Module, image_path: Path, classes: List[str], topk: int = 3) -> List[Tuple[str, float]]:
    pil = datasets.folder.default_loader(str(image_path))
    return _predict_topk_from_pil(model, pil, classes, topk=topk)


def predict_topk_from_bytes(
    model: nn.Module,
    image_bytes: bytes,
    classes: List[str],
    topk: int = 3,
) -> List[Tuple[str, float]]:
    pil = Image.open(io.BytesIO(image_bytes))
    return _predict_topk_from_pil(model, pil, classes, topk=topk)
