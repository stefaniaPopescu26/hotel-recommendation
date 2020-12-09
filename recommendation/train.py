import os
import cv2 
import numpy as np
from os import listdir
from typing import List
from keras.models import load_model
import constants as constants
from keras import models, losses, optimizers
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from model_architecture import model_architecture

np.random.seed(constants.SEED_VALUE)
os.environ[constants.TF_CPP_MIN_LOG_LEVEL] = constants.TF_CPP_MIN_LOG_LEVEL_VALUE

label_binarizer = LabelBinarizer()

ACTIVITIES = ['sitting', 'standing', 'walking']

def prepare_data(data: data) -> train_test_data:
    X = np.array(data.X)
    Y = np.array(data.Y)
    Y = label_binarizer.fit_transform(Y)
    (trainX, testX, trainY, testY) = train_test_split(X, Y)
    print(len(trainX))
    print(len(testX))
    print(len(trainY))
    print(len(testY))

    return train_test_data(trainX, testX, trainY, testY)

def train_model(train_test_data: train_test_data) -> models.Sequential():
    # Model architecture
    model = model_architecture().construct()
    model.summary()
    model.compile(
        loss=losses.categorical_crossentropy,
        metrics=['accuracy'],
        optimizer=optimizers.Adam(learning_rate=0.001))

    # Train model
    model.fit(
        train_test_data.trainX,
        train_test_data.trainY,
        validation_data=(train_test_data.testX, train_test_data.testY),
        epochs = 200,
        batch_size = 32)

    # Check accuracy
    _, accuracy = model.evaluate(train_test_data.testX, train_test_data.testY)
    print('Accuracy: %.2f' % (accuracy * 100))
    return model

def save_model(model: models.Sequential()) -> None:
    model_json = model.to_json()
    with open(constants.MODEL_PATH + constants.MODEL_JSON_NAME, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(constants.MODEL_PATH + constants.MODEL_WEIGHTS_NAME)
    model.save(constants.MODEL_PATH + constants.MODEL_NAME)

folder='dataset/'

def load_data():
    images = []
    for activity in ACTIVITIES:
        path = folder + activity
        print(path)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return images

def main() -> None:
    print('starting...')
    data = load_data()

    data = generate_data()
    train_test_data = prepare_data(data)
    model = train_model(train_test_data)
    save_model(model)

if __name__ == '__main__':
    main()
