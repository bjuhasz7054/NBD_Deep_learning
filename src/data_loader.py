import configparser
import logging
from typing import Tuple

import pandas as pd


class DataLoader:
    def __init__(
        self, config: configparser.ConfigParser, dataset_folder: str = "dataset"
    ):
        self.train_test_split_ratio = config.get(
            "main", "train_test_split_ratio"
        )
        self.decrease_ratio = config.get("main", "decrease_ratio")
        self.random_seed = config.get("main", "random_seed")
        self.dataset_folder = dataset_folder
        self.logger = logging.getLogger(__name__)

        self.train_labels_df: pd.DataFrame
        self.validate_labels_df: pd.DataFrame
        self.test_labels_df: pd.DataFrame

        self.train_size = 0
        self.validate_size = 0
        self.test_size = 0

    def load_dataset(self):
        """
        Load labels from csv into dataframes
        """
        initial_train_dataset, _ = self._split_dataframe(
            base_dataframe=pd.read_csv("fairface_label_train.csv"),
            fraction=self.decrease_ratio,
        )
        self.validate_labels_df, _ = self._split_dataframe(
            base_dataframe=pd.read_csv("fairface_label_val.csv"),
            fraction=self.decrease_ratio,
        )
        self.train_labels_df, self.test_labels_df = self._split_dataframe(
            base_dataframe=initial_train_dataset,
            fraction=self.train_test_split_ratio,
        )

        self.train_size = len(self.train_labels_df)
        self.validate_size = len(self.validate_labels_df)
        self.test_size = len(self.test_labels_df)

        dataset_size = self.train_size + self.validate_size + self.test_size
        self.logger.info(
            f"train percantage = {self.train_size / dataset_size * 100}%"
        )
        self.logger.info(
            f"test percantage = {self.test_size / dataset_size * 100}%"
        )
        self.logger.info(
            f"validate percantage = {self.validate_size / dataset_size * 100}%"
        )

    def _split_dataframe(
        self, base_dataframe, fraction
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        first = base_dataframe.sample(
            frac=fraction, random_state=self.random_seed
        )
        second = base_dataframe.drop(first.index)
        return (first, second)
