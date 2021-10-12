import numpy as np
import math
import random

from src.config import Config
from src.matlab_handler import MatlabData


class Dataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.x_train, self.y_train, self.x_test, self.y_test = self.create_3d_samples()

    def load_data(self):
        data = MatlabData(file_path=self.filepath)
        return data

    def concat_features(self):
        pass

    def create_3d_samples(self):
        data = self.load_data()

        x_train_list_of_samples, y_train_list_of_samples = self.preprocess_samples_to_list(data=data)
        self.shuffle_lists_in_same_order(x_train_list_of_samples, y_train_list_of_samples)
        x_train_list, y_train_list, x_test_list, y_test_list = self.train_test_split(x_samples=x_train_list_of_samples, y_samples=y_train_list_of_samples)

        return x_train_list, y_train_list, x_test_list, y_test_list

    @staticmethod
    def list_of_samples_to_array(list_of_samples: list):
        samples = np.vstack([sample for sample in np.array(list_of_samples)])
        return samples

    @staticmethod
    def shuffle_lists_in_same_order(*args):    # pozor! metoda promicha reference. Takze uz zavolanim se zmeni puvodni listy
        for seznam in args:
            random.seed(Config.SEED)
            random.shuffle(seznam)

    def preprocess_samples_to_list(self, data: MatlabData):
        x_train_list = []
        y_train_list = []

        for sample in range(np.shape(data.dq_oscilace_filtrovane_16)[0] - Config.TIMESTEPS):
            x_train_list.append(np.array(data.dq_oscilace_filtrovane_16[sample:sample + Config.TIMESTEPS, :].reshape((1, Config.TIMESTEPS * Config.NUMBER_OF_SAMPLE_COLUMNS))))
            y_train_list.append(int(data.FAULT))

        return x_train_list, y_train_list

    @staticmethod
    def train_test_split(x_samples: list, y_samples: list):
        train_length = math.floor(len(x_samples) * (1 - Config.TEST_SAMPLES_RATIO))

        x_train = x_samples[:train_length]
        x_test = x_samples[train_length:]

        y_train = y_samples[:train_length]
        y_test = y_samples[train_length:]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def normalize(array_2d):
        minumum = array_2d.min()
        maximum = array_2d.max()
        normalized_array = (array_2d - minumum) / (maximum - minumum)
        return normalized_array

    @staticmethod
    def normalize_by_columns(array):
        scaled_array = array[:]
        for i in range(np.shape(array)[1]):
            scaled_array[:, i] = Dataset.normalize(array[:, i])
        return scaled_array


class TrainingSet:
    def __init__(self, *args):
        self.args = args
        self.number_of_files_loaded = 0

        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.x_train_list, self.y_train_list, self.x_test_list, self.y_test_list = [], [], [], []

        self.load_all_datasets()

    def load_all_datasets(self):
        self.load_from_mat()

    def load_from_mat(self):
        for file_path in self.args[0]:
            print(f"Creating samples from {self.number_of_files_loaded+1}/{len(self.args[0])} of .mat files...")
            dataset = Dataset(filepath=file_path)

            self.x_train_list.extend(dataset.x_train)
            self.y_train_list.extend(dataset.y_train)
            self.x_test_list.extend(dataset.x_test)
            self.y_test_list.extend(dataset.y_test)

            self.number_of_files_loaded += 1
        Dataset.shuffle_lists_in_same_order(self.x_train_list, self.y_train_list, self.x_test_list, self.y_test_list)
        self.stack_all()

    def stack_all(self):
        self.x_train = Dataset.list_of_samples_to_array(self.x_train_list)
        del self.x_train_list

        self.y_train = Dataset.list_of_samples_to_array(self.y_train_list)
        del self.y_train_list

        self.x_test = Dataset.list_of_samples_to_array(self.x_test_list)
        del self.x_test_list

        self.y_test = Dataset.list_of_samples_to_array(self.y_test_list)
        del self.y_test_list
