import glob
import random
import copy

from src.config import Config
from src.nn_model import NNHandler
from src.tfmodel import TFLiteModel
from src.dataset import TrainingSet
from src.logger import TrainLogger, TestLogger, TestLoggerTFLite, ConfigLogger


class InterfaceTrainTest:
    @staticmethod
    def get_all_matfiles_names():
        list_of_files = []
        for folder in Config.DATA_FOLDERS:
            list_from_folder = glob.glob(f"{folder}*.mat")

            list_of_files.extend(list_from_folder)
        return list_of_files


class Trainer(InterfaceTrainTest):
    LOGGER = TrainLogger()

    @staticmethod
    def log_config():
        ConfigLogger.save_config_attributes_to_file()

    @staticmethod
    def load_training_set(list_of_files: list):
        return TrainingSet(list_of_files)

    def batch_train(self):
        Config.FILE_BATCH_COUNTER = 0
        all_files = self.get_all_matfiles_names()

        while True:
            already_trained_on: list = self.LOGGER.get_all_filenames()
            remaining_to_train: list = [file_name for file_name in all_files if file_name not in already_trained_on]   # vytvori list, kde jsou vsechny filenames z all_files, ktere jeste nejsou v already_trained_on

            if len(remaining_to_train) == 0:
                print("The model has already been trained on all available data.")
                self.log_config()
                break

            random.seed(Config.SEED)
            random.shuffle(remaining_to_train)

            train_batch = remaining_to_train[:Config.FILE_BATCH]

            print(f"Files remaining to train: {len(remaining_to_train)}")
            print(f"Already trained on files: {len(already_trained_on)}")
            print(f"Files loaded to train: {len(train_batch)}")

            training_set = self.load_training_set(train_batch)
            self._train(training_set=training_set)

            for file in train_batch:
                if file not in already_trained_on:
                    self.LOGGER.write_log(str(file))

    @staticmethod
    def _train(training_set):
        Config.FILE_BATCH_COUNTER += 1
        nn_model = NNHandler(dataset=training_set)
        nn_model.train_model()


class Tester(InterfaceTrainTest):
    LOGGER = TestLogger()
    LOGGER_TFLITE = TestLoggerTFLite()

    def __init__(self):
        self.total_right_predictions = 0
        self.total_number_of_predictions = 0
        self.trainer = Trainer()

    def evaluate_per_file(self):
        all_files = self.get_all_matfiles_names()
        for path in all_files:
            training_files = self.trainer.LOGGER.get_all_filenames()
            already_tested_files = self.LOGGER.get_all_filenames()

            if path in already_tested_files:
                continue
            if path in training_files:
                self.evaluate_model_on_file(file_path=path, training_file=True)
            else:
                self.evaluate_model_on_file(file_path=path, training_file=False)

    def evaluate_model_on_file(self, file_path, training_file: bool):
        dataset = TrainingSet([file_path])
        tf_dataset = copy.deepcopy(dataset)

        keras_model = NNHandler(dataset)
        tfmodel = TFLiteModel(tf_dataset)

        print(f"\n\nevaluating keras model")
        self.evaluate_file(model=keras_model, logger=self.LOGGER, training_file=training_file, file_path=file_path)
        print(f"\n\nevaluating tfmodel")
        self.evaluate_file(model=tfmodel, logger=self.LOGGER_TFLITE, training_file=training_file, file_path=file_path)

    @staticmethod
    def evaluate_file(model, logger, training_file, file_path):
        if training_file:
            print("Using only test data from this file for testing.")
            _, total_right, total = model.evaluate_model()
        else:
            print("Using all data from this file as testing data.")
            _, total_right, total = model.evaluate_train_and_test()
        # self.total_right_predictions += total_right
        # self.total_number_of_predictions += total
        acc_str = str(format(round((total_right/total), 8), '.8f'))
        print(f"File acc: {acc_str} , path: {file_path}")

        logger.write_log(acc_str, str(file_path), f"{training_file}")


class TestIntegrity(InterfaceTrainTest):
    def __init__(self):
        self.all_file_paths = self.get_all_matfiles_names()
        self.trainer = Trainer()
        self.tester = Tester()

        self.train_files = self.trainer.LOGGER.get_all_filenames()
        self.test_files = self.tester.LOGGER.get_all_filenames()
        self.test_files_tflite = self.tester.LOGGER_TFLITE.get_all_filenames()

    def check_tested_all_files(self, test_files) -> bool:
        for file_path in self.all_file_paths:
            if file_path not in test_files:
                return False
        return True

    def check_trained_all_files(self) -> bool:
        for file_path in self.all_file_paths:
            if file_path not in self.train_files:
                return False
        return True

    def check_test_files_in_train_files(self, test_files):
        for file_name in self.train_files:
            if file_name in test_files:
                return True
        return False

    def test_result(self):
        print("CHECK INTEGRITY RESULT #############################################################")
        print(f"Duplicates in Keras tested files: {self.tester.LOGGER.contain_duplicates()}")
        print(f"Duplicates in TF Lite tested files: {self.tester.LOGGER_TFLITE.contain_duplicates()}")
        print(f"Duplicates in trained files: {self.trainer.LOGGER.contain_duplicates()}")
        print(f"Trained on all files: {self.check_trained_all_files()}")
        print(f"Tested Keras on all files: {self.check_tested_all_files(self.test_files)}")
        print(f"Tested TF Lite on all files: {self.check_tested_all_files(self.test_files_tflite)}")
        # print(f"Keras testing ran on some training files too: {self.check_test_files_in_train_files(self.test_files)}")
        # print(f"TF Lite testing ran on some training files too: {self.check_test_files_in_train_files(self.test_files_tflite)}")
        print("####################################################################################")
