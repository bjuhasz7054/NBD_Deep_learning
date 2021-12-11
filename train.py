import configparser
from src.fair_face_model import FairFaceModel
from src.data_process.data_generator import DataGenerator
from src.data_process.data_encoder import DataEncoder
from src.data_process.data_loader import DataLoader


model_path = None
config = configparser.ConfigParser()
config.read("config.ini")

data_loader = DataLoader(config=config)
data_encoder = DataEncoder()
data_generator = DataGenerator(config=config)

data_loader.load_dataset()
data_encoder.encode_dataset(data_loader=data_loader)
data_generator.create_data_generators(data_encoder=data_encoder)

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

model.train(data_generator=data_generator, data_loader=data_loader)
