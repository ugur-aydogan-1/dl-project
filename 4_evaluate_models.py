from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


@dataclass(frozen=True)
class EvalConfig:
    test_dir: Path
    baseline_model_path: Path
    roboflow_model_path: Path
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    output_dir: Path = Path("artifacts/evaluation")


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


def create_test_loader(config: EvalConfig) -> tuple[DataLoader, list[str]]:
    transform = build_transform(config.image_size)
    dataset = datasets.ImageFolder(config.test_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return loader, dataset.classes


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def load_model(model_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    model = build_model(num_classes=num_classes, device=device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()

        all_predictions.extend(predictions)
        all_labels.extend(labels.tolist())

    return all_labels, all_predictions


def get_label_ids(class_names: list[str]) -> list[int]:
    return list(range(len(class_names)))


def save_classification_report(
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    label_ids = get_label_ids(class_names)
    report = classification_report(
        labels,
        predictions,
        labels=label_ids,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")


def save_confusion_matrix_plot(
    labels: list[int],
    predictions: list[int],
    class_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    label_ids = get_label_ids(class_names)
    cm = confusion_matrix(labels, predictions, labels=label_ids)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=30, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    model_name: str,
    model_path: Path,
    dataloader: DataLoader,
    class_names: list[str],
    device: torch.device,
    output_dir: Path,
) -> None:
    model = load_model(model_path=model_path, num_classes=len(class_names), device=device)
    labels, predictions = collect_predictions(model=model, dataloader=dataloader, device=device)
    label_ids = get_label_ids(class_names)

    accuracy = accuracy_score(labels, predictions)
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {accuracy:.4f}\n")

    report = classification_report(
        labels,
        predictions,
        labels=label_ids,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(report)

    report_path = output_dir / f"{model_name.lower().replace(' ', '_')}_report.txt"
    matrix_path = output_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"

    save_classification_report(
        labels=labels,
        predictions=predictions,
        class_names=class_names,
        output_path=report_path,
    )

    save_confusion_matrix_plot(
        labels=labels,
        predictions=predictions,
        class_names=class_names,
        title=model_name,
        output_path=matrix_path,
    )

    print(f"Saved report to: {report_path}")
    print(f"Saved confusion matrix to: {matrix_path}")


def run(config: EvalConfig) -> None:
    device = get_device()
    print(f"Using device: {device}")

    test_loader, class_names = create_test_loader(config)
    print(f"Classes: {class_names}")

    evaluate_model(
        model_name="Baseline ResNet50",
        model_path=config.baseline_model_path,
        dataloader=test_loader,
        class_names=class_names,
        device=device,
        output_dir=config.output_dir,
    )

    evaluate_model(
        model_name="Roboflow ResNet50",
        model_path=config.roboflow_model_path,
        dataloader=test_loader,
        class_names=class_names,
        device=device,
        output_dir=config.output_dir,
    )


if __name__ == "__main__":
    config = EvalConfig(
        test_dir=Path("data/processed/test"),
        baseline_model_path=Path("artifacts/baseline_resnet50.pth"),
        roboflow_model_path=Path("artifacts/roboflow_resnet50.pth"),
    )
    run(config)
