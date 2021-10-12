import os
import numpy as np
import datetime
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
import tensorflow as tf

from src.config import Config
from src.dataset import TrainingSet


class NNModel:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        if os.path.isfile(Config.KERAS_MODEL_PATH):
            model = keras.models.load_model(Config.KERAS_MODEL_PATH)
        else:
            print("Model file does not exist! Creating new model...")
            model = self.create_model()
        return model

    def save_model(self):
        self.model.save(Config.KERAS_MODEL_PATH)

    @property
    def check_model(self) -> bool:
        if self.model is None:
            print("Model is not loaded!")
            return True
        return False

    @staticmethod
    def create_model():
        model = Sequential()

        model.add(Dense(input_dim=(Config.TIMESTEPS * Config.NUMBER_OF_SAMPLE_COLUMNS), units=Config.DENSE_NEURONS[0], activation="relu"))
        for number_of_neurons in Config.DENSE_NEURONS[1:]:
            model.add(Dense(units=number_of_neurons, activation="relu"))

        model.add(Dense(units=2, activation='softmax'))
        opt = Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=["sparse_categorical_accuracy"])
        print("Model created.")
        model.save(Config.KERAS_MODEL_PATH)
        print("Model saved.")
        return model

    def predict(self, input_sample):
        return self.model.predict(input_sample, batch_size=np.shape(input_sample)[0])


class NNHandler(NNModel):
    def __init__(self, dataset: TrainingSet):
        super().__init__()
        self.dataset = dataset

    def train_model(self):
        if self.check_model:
            return
        print(f"Training on data from {self.dataset.number_of_files_loaded} mat files.")
        log_dir = f"{Config.MODEL_FOLDER}logs/fit/file_batch_number_{Config.FILE_BATCH_COUNTER}"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model.fit(self.dataset.x_train, self.dataset.y_train, batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, callbacks=[tensorboard_callback])
        self.save_model()

    def speed_test(self):
        if self.check_model:
            return
        cas = datetime.datetime.now()
        for i in range(Config.SPEED_TEST_ITERATIONS):
            prediction = self.predict(input_sample=self.dataset.x_test[[i], :])
        cas2 = datetime.datetime.now()
        print(f"us/sample: {((cas2 - cas) / Config.SPEED_TEST_ITERATIONS).microseconds}")

    def evaluate_model(self):
        pocet_spravnych = 0
        predikce = self.predict(input_sample=self.dataset.x_test[:np.shape(self.dataset.x_test)[0], :])
        for i in range(len(self.dataset.x_test)):
            if self.dataset.y_test[[i]] == np.argmax(predikce[i, :]):
                pocet_spravnych += 1

        total_predictions = len(self.dataset.x_test)
        acc: float = round((pocet_spravnych / total_predictions), 8)
        print(f"Model accuracy on test data: {acc}")
        return acc, pocet_spravnych, total_predictions

    def evaluate_train_and_test(self):
        pocet_spravnych = 0
        predikce = self.predict(input_sample=self.dataset.x_test[:np.shape(self.dataset.x_test)[0], :])
        for i in range(len(self.dataset.x_test)):
            if self.dataset.y_test[[i]] == np.argmax(predikce[i, :]):
                pocet_spravnych += 1

        predikce = self.predict(input_sample=self.dataset.x_train[:np.shape(self.dataset.x_train)[0], :])
        for i in range(len(self.dataset.x_train)):
            if self.dataset.y_train[[i]] == np.argmax(predikce[i, :]):
                pocet_spravnych += 1

        total_predictions = len(self.dataset.x_train)+len(self.dataset.x_test)
        acc: float = round((pocet_spravnych / total_predictions), 8)
        print(f"Model accuracy on test data: {acc}")
        return acc, pocet_spravnych, total_predictions

    def print_predicted_values(self):
        for i in range(20):
            input_sample = self.dataset.x_test[i, :].reshape(1, Config.TIMESTEPS,
                                                                Config.NUMBER_OF_SAMPLE_COLUMNS).astype(np.float32)
            nn_output = self.predict(input_sample=input_sample)
            print(f"prediction: {np.argmax(nn_output)} - right class: {int(self.dataset.y_test[i, :])} ... softmax output: {nn_output}")
