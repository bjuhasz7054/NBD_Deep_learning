#  """Encode categorical data to one-hot representation"""
import pandas as pd

from src.data_process.data_loader import DataLoader


class DataEncoder:
    CLASSES = ["age", "race", "gender"]

    def encode_dataset(self, data_loader: DataLoader):
        self.oh_train_labels_df = self.one_hot_encode_data(
            data_loader.train_labels_df
        )
        self.oh_validate_labels_df = self.one_hot_encode_data(
            data_loader.validate_labels_df
        )
        self.oh_test_labels_df = self.one_hot_encode_data(
            data_loader.test_labels_df
        )

        self.int_train_labels_df = self.int_encode_data(
            data_loader.train_labels_df
        )
        self.int_validate_labels_df = self.int_encode_data(
            data_loader.validate_labels_df
        )
        self.int_test_labels_df = self.int_encode_data(
            data_loader.test_labels_df
        )

    @classmethod
    def one_hot_encode_data(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical data to onehot representation"""
        new_df = dataframe.copy()
        for coloumn in cls.CLASSES[:-1]:
            new_df[coloumn] = (
                dataframe[coloumn].str.get_dummies().values.tolist()
            )
            new_df["gender"] = (dataframe["gender"] == "Male").astype(int)

        return new_df

    @classmethod
    def int_encode_data(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical data to integer representation"""

        new_df = dataframe.copy()
        for coloumn in cls.CLASSES:
            new_df[coloumn] = pd.factorize(dataframe[coloumn], sort=True)[0]
        return new_df
