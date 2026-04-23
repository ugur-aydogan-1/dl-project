from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


CLASS_COLUMNS = ["healthy", "multiple_diseases", "rust", "scab"]


@dataclass(frozen=True)
class SplitConfig:
    csv_path: Path
    images_dir: Path
    output_dir: Path
    image_extension: str = ".jpg"
    train_size: float = 0.70
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42


def load_labeled_samples(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing_columns = [col for col in ["image_id", *CLASS_COLUMNS] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {missing_columns}")

    df["label"] = df[CLASS_COLUMNS].idxmax(axis=1)

    invalid_rows = df[CLASS_COLUMNS].sum(axis=1) != 1
    if invalid_rows.any():
        raise ValueError("Some rows do not have exactly one active class.")

    return df[["image_id", "label"]].copy()


def split_dataframe(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError("train_size, val_size and test_size must sum to 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df["label"],
        random_state=random_state,
    )

    val_ratio_in_temp = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_ratio_in_temp),
        stratify=temp_df["label"],
        random_state=random_state,
    )

    return train_df, val_df, test_df


def copy_images(
    df: pd.DataFrame,
    images_dir: Path,
    output_dir: Path,
    split_name: str,
    image_extension: str,
) -> None:
    for _, row in df.iterrows():
        image_name = f"{row['image_id']}{image_extension}"
        source_path = images_dir / image_name
        target_dir = output_dir / split_name / row["label"]
        target_path = target_dir / image_name

        if not source_path.exists():
            raise FileNotFoundError(f"Image not found: {source_path}")

        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def print_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{split_name.upper()} ({len(split_df)} images)")
        print(split_df["label"].value_counts().sort_index())


def run(config: SplitConfig) -> None:
    df = load_labeled_samples(config.csv_path)

    train_df, val_df, test_df = split_dataframe(
        df=df,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    copy_images(train_df, config.images_dir, config.output_dir, "train", config.image_extension)
    copy_images(val_df, config.images_dir, config.output_dir, "val", config.image_extension)
    copy_images(test_df, config.images_dir, config.output_dir, "test", config.image_extension)

    print_summary(train_df, val_df, test_df)
    print(f"\nDone. Output directory: {config.output_dir}")


if __name__ == "__main__":
    config = SplitConfig(
        csv_path=Path("data/raw/train.csv"),
        images_dir=Path("data/raw/images"),
        output_dir=Path("data/processed"),
    )
    run(config)