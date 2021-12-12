import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import offsetbox
from sklearn import manifold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model


class Evaluator:
    """
    Measure performance of model
    """

    AGE_LABELS = [
        "0-2",
        "10-19",
        "20-29",
        "3-9",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "more than 70",
    ]
    RACE_LABELS = [
        "Black",
        "East Asian",
        "Indian",
        "Latino Hispanic",
        "Middle Eastern",
        "Southeast Asian",
        "White",
    ]
    GENDER_LABELS = ["Female", "Male"]

    def __init__(
        self,
        model: Model,
        test_generator: ImageDataGenerator,
        int_test_labels_df: pd.DataFrame,
        results_folder: str = "results",
    ):
        self.model = model
        self.test_generator = test_generator
        self.int_test_labels_df = int_test_labels_df
        self.results_folder = os.path.join(results_folder, str(time.time()))
        os.makedirs(self.results_folder)

        Age_pred, Race_pred, Gender_pred = model.predict(test_generator)
        # convert output vectors to single integers
        self.age_pred = np.argmax(Age_pred, axis=1)
        self.race_pred = np.argmax(Race_pred, axis=1)
        self.gender_pred = Gender_pred >= 0.5

    def confusion_matrix(self):
        conf = confusion_matrix(
            self.int_test_labels_df["age"].values, self.age_pred
        )
        plt.figure(figsize=(13, 10))
        ax = sns.heatmap(conf, annot=True, fmt="d", cmap="Blues")
        ax.set(xlabel="Age Predicted Label", ylabel="Age True label")
        ax.set_title("Age Confusion Matrix")
        ax.xaxis.set_ticklabels(self.AGE_LABELS)
        ax.yaxis.set_ticklabels(self.AGE_LABELS)
        plt.savefig(os.path.join(self.results_folder, "confusion_matrix_age"))

        conf = confusion_matrix(
            self.int_test_labels_df["race"].values, self.race_pred
        )
        plt.figure(figsize=(13, 10))
        ax = sns.heatmap(conf, annot=True, fmt="d", cmap="Blues")
        ax.set(xlabel="Race Predicted Label", ylabel="Race True label")
        ax.set_title("Race Confusion Matrix")
        ax.xaxis.set_ticklabels(self.RACE_LABELS)
        ax.yaxis.set_ticklabels(self.RACE_LABELS)
        plt.savefig(os.path.join(self.results_folder, "confusion_matrix_race"))

        conf = confusion_matrix(
            self.int_test_labels_df["gender"].values, self.gender_pred
        )
        plt.figure(figsize=(13, 10))
        ax = sns.heatmap(conf, annot=True, fmt="d", cmap="Blues")
        ax.set(xlabel="Gender Predicted Label", ylabel="Gender True label")
        ax.set_title("Race Confusion Matrix")
        ax.xaxis.set_ticklabels(self.GENDER_LABELS)
        ax.yaxis.set_ticklabels(self.GENDER_LABELS)
        plt.savefig(
            os.path.join(self.results_folder, "confusion_matrix_gender")
        )

    def classification_report(self):
        with open(os.path.join(self.results_folder, "classification_report.txt"),'w',encoding = 'utf-8') as f:
            f.write("Age Classification Report\n")
            f.write(
                classification_report(
                    self.int_test_labels_df["age"],
                    self.age_pred,
                    target_names=self.AGE_LABELS,
                )
            )

            f.write("Race Classification Report\n")
            f.write(
                classification_report(
                    self.int_test_labels_df["race"],
                    self.race_pred,
                    target_names=self.RACE_LABELS,
                )
            )

            f.write("Gender Classification Report\n")
            f.write(
                classification_report(
                    self.int_test_labels_df["gender"],
                    self.gender_pred,
                    target_names=self.GENDER_LABELS,
                )
            )

    def plt_tsne_for_all_classes(self):
        encoder = Model(self.model.input, self.model.layers[-4].output)

        self.plot_tsne(
            encoder.predict(self.test_generator),
            self.int_test_labels_df["age"],
            classes=self.AGE_LABELS,
            name="age_tsne.pdf",
        )
        self.plot_tsne(
            encoder.predict(self.test_generator),
            self.int_test_labels_df["race"],
            classes=self.RACE_LABELS,
            name="race_tsne.pdf",
        )
        self.plot_tsne(
            encoder.predict(self.test_generator),
            self.int_test_labels_df["gender"],
            classes=self.GENDER_LABELS,
            name="gender_tsne.pdf",
        )

    @staticmethod
    def plot_tsne(x, y, classes, with_pictures=False, name="figure.eps"):
        name = os.path.join(self.results_folder, name)
        colors = np.asarray(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        cmap, norm = matplotlib.colors.from_levels_and_colors(
            np.arange(0, 9 + 2), colors[: 9 + 1]
        )
        fig, ax = plt.subplots(figsize=(8, 8))
        # xr=x.reshape(-1, x.shape[1]**2)
        tsne = manifold.TSNE(n_components=2, init="pca", random_state=0)
        X_tsne = tsne.fit_transform(x)
        x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
        X_tsne = (X_tsne - x_min) / (x_max - x_min)
        scatter = ax.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=y,
            marker="o",
            linewidth=1,
            cmap=cmap,
            norm=norm,
        )

        legend1 = ax.legend(
            handles=scatter.legend_elements()[0],
            labels=classes,
            loc="upper left",
            title="Osztályok",
            bbox_to_anchor=(1.04, 1),
        )
        ax.add_artist(legend1)
        if hasattr(offsetbox, "AnnotationBbox") and with_pictures:
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1.0, 1.0]])  # just something big
            for i in range(X_tsne.shape[0]):
                dist = np.sum((X_tsne[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X_tsne[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(x[i], zoom=1.5, cmap=plt.cm.gray_r),
                    X_tsne[i],
                )
                ax.add_artist(imagebox)

        plt.savefig(name, format="pdf", bbox_inches="tight", dpi=300)
