import os
import pandas as pd

from src.config import Config


class TrainLogger:
    LOG_FILE = Config.TRAIN_LOG_CSV

    def __init__(self):
        self.write_column_names()

    def write_column_names(self):
        if os.path.isfile(self.LOG_FILE):
            if os.stat(self.LOG_FILE).st_size == 0: # empty file
                csv_file = open(self.LOG_FILE, "a")
                csv_file.writelines("file path;" + "\n")
                csv_file.close()
        else:
            csv_file = open(self.LOG_FILE, "a")
            csv_file.writelines("file path;" + "\n")
            csv_file.close()

    def write_log(self, *args):
        text_to_write = ";".join(args) + ";"
        csv_file = open(self.LOG_FILE, "a")
        csv_file.writelines(text_to_write+"\n")
        csv_file.close()

    def get_all_filenames(self):
        files_list = []
        try:
            file_log = open(self.LOG_FILE, "r")
            files_list = file_log.readlines()
            files_list = list(map(lambda x: x.split(";")[0], files_list))
        except:
            raise Exception("Unable to open train log file.")
        return files_list[1:]   # not including column names

    def contain_duplicates(self) -> bool:
        all_files = self.get_all_filenames()
        for file_name in all_files:
            if all_files.count(file_name) > 1:
                return True
        return False


class TestLogger(TrainLogger):
    LOG_FILE = Config.TEST_LOG_CSV

    def __init__(self):
        self.write_column_names()

    def write_column_names(self):
        if os.path.isfile(self.LOG_FILE):
            if os.stat(self.LOG_FILE).st_size == 0:  # empty file
                csv_file = open(self.LOG_FILE, "a")
                csv_file.writelines("accuracy;file path;file is also training file;" + "\n")
                csv_file.close()
        else:
            csv_file = open(self.LOG_FILE, "a")
            csv_file.writelines("accuracy;file path;file is also training file;" + "\n")
            csv_file.close()

    def get_all_filenames(self):
        files_list = []

        file_log = open(Config.TEST_LOG_CSV)
        files = file_log.readlines()
        files_list = list(map(lambda x: x.split(";")[-3], files))

        return files_list[1:]   # not including column names


class TestLoggerTFLite(TestLogger):
    LOG_FILE = Config.TEST_TFLITE_LOG_CSV


class TestAnalysis:
    LOG_FILE = Config.TEST_LOG_CSV

    def __init__(self):
        self.test_data = self.load_data()

    def load_data(self):
        return pd.read_csv(self.LOG_FILE, sep=";").iloc[:, :-1]

    def get_files_with_acc_between(self, accuracy_lower: float, accuracy_higher: float):
        result = self.test_data.loc[self.test_data['accuracy'] <= accuracy_higher]
        return result.loc[self.test_data['accuracy'] > accuracy_lower]

    @property
    def overall_test_accuracy(self) -> float:
        return self.test_data["accuracy"].mean()

    def get_nb_of_files_by_acc(self):
        percentage: list = []
        number_of_files: list = []

        for i in range(1, 101):
            df = self.get_files_with_acc_between(accuracy_lower=(i-1)/100, accuracy_higher=i/100)
            if df.shape[0] > 0:
                percentage.append(f"{i-1}% - {i}%")
                number_of_files.append(df.shape[0])

        return dict(zip(percentage, number_of_files))


class TestAnalysisTFLite(TestAnalysis):
    LOG_FILE = Config.TEST_TFLITE_LOG_CSV


class ConfigLogger:
    @staticmethod
    def save_config_attributes_to_file():
        dictionary = Config.attributes_as_dictionary()
        df = pd.DataFrame(dictionary.items(), columns=["attributes", "values"])
        df.to_csv(Config.CONFIG_LOG_CSV, sep=";", index=False)
