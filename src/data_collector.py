import gdown
import os
import logging
import zipfile

class DataCollector:
    """
    Download and extract dataset from google drive.
    """

    DATASET_ID = "1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86"
    LABELS_TRAIN_ID = "1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH"
    LABELS_VALIDATE_ID = "1wOdja-ezstMEp81tX1a-EYkFebev4h7D"

    def __init__(self, output_folder: str = "dataset", overwrite_folder: bool = False, remove_zip_file: bool = True):
        self.logger = logging.getLogger(__name__)

        if not os.path.exists(output_folder):
            self.logger.info(f"creating folder {output_folder}")
            os.makedirs(output_folder)

        self.output_folder = output_folder
        self.overwrite = overwrite_folder
        self.remove_zip_file = remove_zip_file


    def collect_dataset(self):
        self.download_dataset()
        self.extract_dataset()

    def download_dataset(self):
        if not self.overwrite and os.listdir(self.output_folder):
            self.logger.warning(f"dataset folder {self.output_folder} not empty, download failed")
            return

        gdown.download(id=self.DATASET_ID, output=os.path.join(self.output_folder, "dataset.zip"))
        gdown.download(id=self.LABELS_TRAIN_ID, output=os.path.join(self.output_folder, "labels_train.csv"))
        gdown.download(id=self.LABELS_VALIDATE_ID, output=os.path.join(self.output_folder, "labels_validate.csv"))

    def extract_dataset(self):
        dataset_file = os.path.join(self.output_folder, "dataset.zip")

        if not os.path.exists(dataset_file):
            self.logger.warning(f"dataset file {dataset_file} not found, extraction failed")
            return

        with zipfile.ZipFile(dataset_file, "r") as zip_ref:
            zip_ref.extractall(self.output_folder)

        if self.remove_zip_file:
            os.remove(dataset_file)
