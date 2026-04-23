from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


@dataclass(frozen=True)
class TrainConfig:
    train_dir: Path
    val_dir: Path
    test_dir: Path
    image_size: int = 224
    batch_size: int = 32
    num_epochs: int = 5
    learning_rate: float = 1e-4
    num_workers: int = 2
    model_output_path: Path = Path("artifacts/roboflow_resnet50.pth")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    transform = build_transform(config.image_size)

    train_dataset = datasets.ImageFolder(config.train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(config.val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(config.test_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_model(model: nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def load_model(model: nn.Module, model_path: Path, device: torch.device) -> nn.Module:
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def run(config: TrainConfig) -> None:
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = create_dataloaders(config)
    print(f"Classes: {class_names}")

    model = build_model(num_classes=len(class_names), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    best_val_accuracy = 0.0

    for epoch in range(1, config.num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model, config.model_output_path)

    print(f"Best model saved to: {config.model_output_path}")

    best_model = build_model(num_classes=len(class_names), device=device)
    best_model = load_model(best_model, config.model_output_path, device)

    test_loss, test_accuracy = evaluate(
        model=best_model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    config = TrainConfig(
        train_dir=Path("data/roboflow_augmented/train"),
        val_dir=Path("data/processed/val"),
        test_dir=Path("data/processed/test"),
    )
    run(config)