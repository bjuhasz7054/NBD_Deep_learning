import pathlib
import argparse
from src.data_process.data_collector import DataCollector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect FairFace dataset")
    parser.add_argument(
        "-d",
        "--dataset_folder",
        type=pathlib.Path,
        default="dataset",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="overwrite existing dataset",
    )
    parser.add_argument(
        "-k",
        "--keep_zip",
        action="store_false",
        help="do not remove zip file after extracting",
    )

    args = parser.parse_args()

    data_collector = DataCollector(
        output_folder=args.dataset_folder,
        overwrite_folder=args.overwrite,
        remove_zip_file=args.keep_zip,
    )
    data_collector.collect_dataset()
