import os
import pandas as pd
import numpy as np
import datetime
import tensorflow as tf
import tensorflow.lite as lite
import keras
import random

from src.config import Config
from src.dataset import TrainingSet
from src.scaler import Scaler


class TFLiteModel:
    def __init__(self, dataset: TrainingSet):
        self.dataset = dataset
        self.scaler = Scaler()
        self._input_details = None
        self._input_shape = None
        self._output_details = None
        self.tfmodel = None
        self._interpreter = None

        self.set_up_interpreter()
        self.export_layers_to_excel()

    def set_up_interpreter(self):
        if os.path.isfile(Config.TFLITE_MODEL_PATH):
            self._interpreter = tf.lite.Interpreter(model_path=Config.TFLITE_MODEL_PATH)
        else:
            self.to_tf_lite()
            self._interpreter = tf.lite.Interpreter(model_path=Config.TFLITE_MODEL_PATH)

        self.prepare_model()

    def representative_data_gen(self):
        for input_value in tf.data.Dataset.from_tensor_slices(self.dataset.x_train).batch(1).take(10000):
            input_value = tf.cast(input_value, tf.float32)
            yield [input_value]

    def to_tf_lite(self):
        keras_model = keras.models.load_model(Config.KERAS_MODEL_PATH)
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

        converter.post_training_quantize = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        dataset = self.representative_data_gen
        converter.representative_dataset = dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

        tflite_model = converter.convert()
        open(Config.TFLITE_MODEL_PATH, "wb").write(tflite_model)

    def tflite_model_layers_info(self):
        all_layers_details = self._interpreter.get_tensor_details()
        net_layers = pd.DataFrame(all_layers_details)
        return net_layers

    def export_layers_to_excel(self):
        layers_df = self.tflite_model_layers_info()
        writer = pd.ExcelWriter(Config.TFLITE_MODEL_LAYERS_EXCEL_FILE)
        layers_df.to_excel(writer, 'TFlite_model')
        writer.save()

    def prepare_model(self):
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        self._input_shape = self._input_details[0]['shape']

    def predict(self, input_sample):
        self._interpreter.set_tensor(self._input_details[0]['index'], input_sample)
        self._interpreter.invoke()
        output_data = self._interpreter.get_tensor(self._output_details[0]['index'])
        return output_data

    def speed_test(self):
        cas = datetime.datetime.now()

        for i in range(Config.SPEED_TEST_ITERATIONS):
            input_sample = self.dataset.x_test[i, :].reshape(1, Config.TIMESTEPS * Config.NUMBER_OF_SAMPLE_COLUMNS).astype(np.float32)
            predikce = np.argmax(self.predict(input_sample=input_sample))
        cas2 = datetime.datetime.now()
        print(f"us/sample: {((cas2 - cas) / Config.SPEED_TEST_ITERATIONS).microseconds}")

    def evaluate_model(self):
        pocet_spravnych = 0
        total_predictions = np.shape(self.dataset.x_test)[0]
        for i in range(total_predictions):
            predikce = self.predict(input_sample=self.dataset.x_test[i, :].reshape((1, Config.TIMESTEPS * Config.NUMBER_OF_SAMPLE_COLUMNS)).astype(np.float32))
            if self.dataset.y_test[[i]] == np.argmax(predikce):
                pocet_spravnych += 1

        acc: float = round((pocet_spravnych / total_predictions), 8)
        print(f"Model accuracy on test data: {acc}")
        return acc, pocet_spravnych, total_predictions

    def evaluate_train_and_test(self):
        pocet_spravnych = 0

        for i in range(np.shape(self.dataset.x_test)[0]):
            predikce = self.predict(input_sample=self.dataset.x_test[i, :].reshape((1, Config.TIMESTEPS * Config.NUMBER_OF_SAMPLE_COLUMNS)).astype(np.float32))
            if self.dataset.y_test[[i]] == np.argmax(predikce):
                pocet_spravnych += 1

        for i in range(np.shape(self.dataset.x_train)[0]):
            predikce = self.predict(input_sample=self.dataset.x_train[i, :].reshape((1, Config.TIMESTEPS * Config.NUMBER_OF_SAMPLE_COLUMNS)).astype(np.float32))
            if self.dataset.y_train[[i]] == np.argmax(predikce):
                pocet_spravnych += 1

        total_predictions = np.shape(self.dataset.x_train)[0] + np.shape(self.dataset.x_test)[0]
        acc: float = round((pocet_spravnych / total_predictions), 8)
        print(f"Model accuracy on test data: {acc}")
        return acc, pocet_spravnych, total_predictions

    def print_predicted_values(self):
        for i in range(100):
            sample_index = random.randint(0, np.shape(self.dataset.x_test)[0])

            input_sample = self.dataset.x_test[sample_index, :].reshape((1, Config.TIMESTEPS * Config.NUMBER_OF_SAMPLE_COLUMNS)).astype(np.float32)
            nn_output = self.predict(input_sample=input_sample)
            print(f"prediction: {np.argmax(nn_output)} - right class: {int(self.dataset.y_test[sample_index, :])} ... softmax output: {nn_output}")
