"""
Split the large CSV into train/val/test using streaming.
This keeps memory low for 2GB dataset.
"""

import argparse
from os import makedirs

from config import DATASET_PATH, DATA_DIR, TRAIN_RATIO, VAL_RATIO, SEED
from data_utils import split_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DATASET_PATH)
    parser.add_argument("--out_dir", default=DATA_DIR)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit_rows", type=int, default=None)
    args = parser.parse_args()

    makedirs(args.out_dir, exist_ok=True)

    train_path, val_path, test_path = split_dataset(
        args.csv,
        args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_rows=args.limit_rows,
    )

    print("Done splitting dataset")
    print("Train:", train_path)
    print("Val:", val_path)
    print("Test:", test_path)


if __name__ == "__main__":
    main()
