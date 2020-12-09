import os
import cv2
import glob
import numpy as np
from typing import List
import constants as constants
from keras import models, losses, optimizers
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from model_architecture import model_architecture

np.random.seed(constants.SEED_VALUE)
os.environ[constants.TF_CPP_MIN_LOG_LEVEL] = constants.TF_CPP_MIN_LOG_LEVEL_VALUE

MODEL_NAME = 'model.h5'
MODEL_PATH = './model/'
DATASET_PATH = './dataset/'
MODEL_JSON_NAME = 'model.json'
MODEL_WEIGHTS_NAME = 'model_weights.h5'
ACTIVITIES = ['sitting', 'standing', 'walking']

label_binarizer = LabelBinarizer()

class Data(object):

    def __init__(self) -> None:
        self.X = []
        self.Y = []

    def __init__(self, X: List, Y: List) -> None:
        self.X = X
        self.Y = Y

    def __del__(self) -> None:
        del self.X
        del self.Y


class TrainTestData(object):

    def __init__(self) -> None:
        self.trainX = []
        self.testX = []
        self.trainY = []
        self.testY = []

    def __init__(self, trainX: List, testX: List, trainY: List, testY: List) -> None:
        self.trainX = trainX
        self.testX = testX
        self.trainY = trainY
        self.testY = testY

    def __del__(self) -> None:
        del self.trainX
        del self.testX
        del self.trainY
        del self.testY


def load_data() -> TrainTestData:
    X = []
    Y = []
    for activity in ACTIVITIES:
        path = DATASET_PATH + activity
        label = path.split(os.path.sep)[-1]

        for image_path in glob.glob(path + '/*.png'):
            print(image_path)
            img = cv2.resize(cv2.imread(image_path), (100, 37))
            X.append(img)
            Y.append(label)

    X = np.array(X)
    Y = np.array(Y)
    Y = label_binarizer.fit_transform(Y)
    (trainX, testX, trainY, testY) = train_test_split(X, Y)

    return TrainTestData(trainX, testX, trainY, testY)


def train_model(TrainTestData: TrainTestData) -> models.Sequential():
    # Model architecture
    model = model_architecture().construct()
    model.summary()
    model.compile(
        loss=losses.categorical_crossentropy,
        metrics=['accuracy'],
        optimizer=optimizers.Adam(learning_rate=0.001))

    # Train model
    model.fit(
        TrainTestData.trainX,
        TrainTestData.trainY,
        validation_data=(TrainTestData.testX, TrainTestData.testY),
        epochs=2,
        batch_size=32)

    # Check accuracy
    _, accuracy = model.evaluate(TrainTestData.testX, TrainTestData.testY)
    print('Accuracy: %.2f' % (accuracy * 100))
    return model


def save_model(model: models.Sequential()) -> None:
    model_json = model.to_json()
    with open(MODEL_PATH + MODEL_JSON_NAME, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(MODEL_PATH + MODEL_WEIGHTS_NAME)
    model.save(MODEL_PATH + MODEL_NAME)


def main() -> None:
    train_test_data = load_data()
    model = train_model(train_test_data)
    save_model(model)


if __name__ == '__main__':
    main()
