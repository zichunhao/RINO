from pathlib import Path
from datasets import load_dataset
import argparse
import logging


def download(
    download_dir: str | Path, splits: list[str] = ["train", "validation", "test"]
):
    download_dir = Path(download_dir).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("dl4phys/top_tagging")
    for split_name in splits:
        split_data = dataset[split_name]
        download_path = download_dir / f"{split_name}.parquet"
        split_data.to_parquet(download_path)
        logging.info(f"Saved {split_name} split to {download_path}")
    logging.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download top tagging dataset")
    parser.add_argument(
        "--download-dir",
        type=str,
        default=".",
        help="Directory to download the dataset",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to download",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    download(download_dir=args.download_dir, splits=args.splits)
