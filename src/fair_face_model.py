import configparser
from src.data_process.data_loader import DataLoader
from src.data_process.data_generator import DataGenerator
from keras_vggface.vggface import VGGFace
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import (
    CategoricalAccuracy,
    BinaryAccuracy,
    Accuracy,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
from keras import backend as K
from sklearn.utils import class_weight
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class FairFaceModel:
    def __init__(self, config: configparser.ConfigParser):
        self.learning_rate = float(config.get("main", "learning_rate"))
        self.train_batch_size = int(config.get("main", "train_batch_size"))
        self.validate_batch_size = int(
            config.get("main", "validate_batch_size")
        )
        self.epochs = int(config.get("main", "epochs"))
        self.patience = int(config.get("main", "patience"))

    def build(self):
        vgg_features = VGGFace(
            include_top=False, input_shape=(224, 224, 3), pooling="avg"
        )

        for layer in vgg_features.layers:
            layer.trainable = False

        x = vgg_features.output

        x = Dense(640, activation="leaky_relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(896, activation="relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(2176, activation="tanh")(x)
        x = Dropout(0.4)(x)
        x = Dense(640, activation="relu")(x)
        x = Dropout(0.6)(x)
        x = Dense(1664, activation="relu")(x)
        x = Dropout(0.2)(x)

        # 3 outputs, one for each category
        age_output = Dense(9, activation="softmax", name="age")(x)
        race_output = Dense(7, activation="softmax", name="race")(x)
        gender_output = Dense(1, activation="sigmoid", name="gender")(x)

        self.model = Model(
            inputs=vgg_features.input,
            outputs=[age_output, race_output, gender_output],
        )

    def compile(self, data_loader: DataLoader):
        age_class_weights_np = self.weight_class(
            data_loader.train_labels_df.age
        )
        race_class_weights_np = self.weight_class(
            data_loader.train_labels_df.race
        )

        loss_age = self.weighted_categorical_crossentropy(age_class_weights_np)
        loss_race = self.weighted_categorical_crossentropy(
            race_class_weights_np
        )
        # Gender is not weighted because the relevant dataset
        # is very close to balanced
        loss_gender = BinaryCrossentropy()

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=[Accuracy(), BinaryAccuracy(), CategoricalAccuracy()],
            loss={"age": loss_age, "race": loss_race, "gender": loss_gender},
        )

    def train(self, data_generator: DataGenerator, data_loader: DataLoader):
        early_stopping = EarlyStopping(patience=self.patience, verbose=1)
        checkpointer = ModelCheckpoint(
            filepath="/content/drive/MyDrive/figures/final_model.hdf5",
            save_best_only=True,
            verbose=1,
        )

        self.model.fit(
            data_generator.train_generator,
            steps_per_epoch=data_loader.train_size / self.train_batch_size,
            validation_data=data_generator.validate_generator,
            validation_steps=data_loader.validate_size
            / self.validate_batch_size,
            epochs=self.epochs,
            callbacks=(early_stopping, checkpointer),
        )

    @staticmethod
    def weight_class(dataframe_column):
        return class_weight.compute_class_weight(
            "balanced", classes=pd.unique(dataframe_column), y=dataframe_column
        )

    @staticmethod
    def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy

        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        """

        weights = K.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = (
                tf.dtypes.cast(y_true, tf.float64)
                * tf.dtypes.cast(K.log(y_pred), tf.float64)
                * tf.dtypes.cast(weights, tf.float64)
            )
            loss = -K.sum(loss, -1)
            return loss

        return loss
