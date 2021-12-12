"""Create Data Generators"""
import configparser
import random

from keras.preprocessing.image import ImageDataGenerator

from src.data_process.data_encoder import DataEncoder
from src.data_process.jpeg_compressor import jpeg_compress


class DataGenerator:
    def __init__(self, config: configparser.ConfigParser):
        self.random_seed = int(config.get("main", "random_seed"))
        self.train_batch_size = int(config.get("main", "train_batch_size"))
        self.validate_batch_size = int(
            config.get("main", "validate_batch_size")
        )

        self.train_generator: ImageDataGenerator
        self.test_generator: ImageDataGenerator
        self.validate_generator: ImageDataGenerator

    def create_data_generators(
        self, data_encoder: DataEncoder, dataset_folder: str = "dataset"
    ):
        train_datagen = ImageDataGenerator(
            rotation_range=random.randint(40, 90),
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
            preprocessing_function=jpeg_compress,
        )

        validate_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        common_generator_settings = {
            "x_col": "file",
            "y_col": ["age", "race", "gender"],
            "class_mode": "multi_output",
            "seed": self.random_seed,
            "target_size": (224, 224),
            "validate_filenames": True,
            "directory": dataset_folder,
        }

        self.train_generator = train_datagen.flow_from_dataframe(
            dataframe=data_encoder.oh_train_labels_df,
            batch_size=self.train_batch_size,
            **common_generator_settings
        )

        self.test_generator = test_datagen.flow_from_dataframe(
            dataframe=data_encoder.oh_test_labels_df,
            batch_size=128,
            **common_generator_settings,
            shuffle=False
        )

        self.validate_generator = validate_datagen.flow_from_dataframe(
            dataframe=data_encoder.oh_validate_labels_df,
            batch_size=self.validate_batch_size,
            **common_generator_settings
        )
