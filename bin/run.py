import argparse
import configparser
import pathlib

from src.data_process.data_encoder import DataEncoder
from src.data_process.data_generator import DataGenerator
from src.data_process.data_loader import DataLoader
from src.evaluator import Evaluator
from src.fair_face_model import FairFaceModel


def train(
    model_path: str = None, dataset_folder: str = None, results_dir: str = None
):
    config = configparser.ConfigParser()
    config.read("config.ini")

    data_loader = DataLoader(config=config, dataset_folder=dataset_folder)
    data_encoder = DataEncoder()
    data_generator = DataGenerator(config=config)

    data_loader.load_dataset()
    data_encoder.encode_dataset(data_loader=data_loader)
    data_generator.create_data_generators(
        data_encoder=data_encoder, dataset_folder=dataset_folder
    )

    model = FairFaceModel(config=config)

    if model_path:
        from keras.models import load_model

        model.model = load_model(
            model_path,
            custom_objects={"loss": model.weighted_categorical_crossentropy},
        )
    else:
        model.build()
        model.compile(data_loader=data_loader)

    model.train(
        data_generator=data_generator,
        data_loader=data_loader,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train or evaluate Face classification model on FairFace dataset"
    )
    parser.add_argument("run_type", choices=("train", "evaluate"))
    parser.add_argument(
        "-d",
        "--dataset_folder",
        type=pathlib.Path,
        default="dataset",
    )
    parser.add_argument(
        "-r",
        "--results_folder",
        type=pathlib.Path,
        default="results",
        help="Folder to save model checkpoints or save evaluations",
    )
    parser.add_argument(
        "-l",
        "--load_model",
        type=pathlib.Path,
        help="Load model to continue training or to evaluate it",
    )

    args = parser.parse_args()

    if args.run_type == "train":
        train(
            model_path=args.load_model,
            dataset_folder=args.dataset_folder,
            results_dir=args.results_folder,
        )
    elif args.run_type == "evaluate":
        pass
