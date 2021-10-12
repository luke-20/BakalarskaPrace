import pandas as pd
import numpy as np
import glob

from src.config import Config
from src.matlab_handler import MatlabData


class DataLimits:
    def __init__(self):
        self.df_column_names = self.generate_column_names()
        self.dataframe = pd.DataFrame(columns=self.df_column_names)

    @staticmethod
    def get_all_matfiles_names():
        list_of_files = []
        for folder in Config.DATA_FOLDERS:
            list_from_folder = glob.glob(f"{folder}*.mat")

            list_of_files.extend(list_from_folder)
        return list_of_files

    @staticmethod
    def generate_column_names() -> list:
        column_names = []
        for i in range(4):
            column_names.extend([f"proudy-{i+1}: min", f"proudy-{i+1}: max"])
        for i in range(4):
            column_names.extend([f"napeti-{i+1}: min", f"napeti-{i+1}: max"])
        for i in range(16):
            column_names.extend([f"oscilace-{i+1}: min", f"oscilace-{i+1}: max"])
        column_names.append("filepath")
        return column_names

    def find_limits(self):
        all_files = self.get_all_matfiles_names()
        for i, file_path in enumerate(all_files):
            print(f"Searching for max and min values in file {i+1}/{len(all_files)}")
            matlab_handler = MatlabData(file_path=file_path)
            list_of_limits = []
            for column in range(np.shape(matlab_handler.dq_proudy_filtrovane_4)[1]):
                list_of_limits.extend([np.min(matlab_handler.dq_proudy_filtrovane_4[:, column]), np.max(matlab_handler.dq_proudy_filtrovane_4[:, column])])
            for column in range(np.shape(matlab_handler.dq_napeti_filtrovane_4)[1]):
                list_of_limits.extend([np.min(matlab_handler.dq_napeti_filtrovane_4[:, column]), np.max(matlab_handler.dq_napeti_filtrovane_4[:, column])])
            for column in range(np.shape(matlab_handler.dq_oscilace_filtrovane_16)[1]):
                list_of_limits.extend([np.min(matlab_handler.dq_oscilace_filtrovane_16[:, column]), np.max(matlab_handler.dq_oscilace_filtrovane_16[:, column])])

            row = list_of_limits + [file_path]
            row_df = pd.DataFrame(dict(zip(self.df_column_names, row)), index=[0])

            if i == 0:
                self.dataframe = row_df
            else:
                self.dataframe = pd.concat([self.dataframe, row_df], ignore_index=True)

    def limits_to_csv(self):
        self.dataframe.to_csv(Config.DATA_LIMITS_CSV, index=False, sep=";")

    def get_results(self):
        self.find_limits()
        self.limits_to_csv()

    @staticmethod
    def get_min_max_values():
        df = pd.read_csv(Config.DATA_LIMITS_CSV, sep=";")
        array = df.to_numpy()
        proudy_array = array[:, 0:8]
        napeti_array = array[:, 8:16]
        oscilace_array = array[:, 16:-1]

        min_proudy = np.min(proudy_array[::2])
        max_proudy = np.max(proudy_array[1::2])

        min_napeti = np.min(napeti_array[::2])
        max_napeti = np.max(napeti_array[1::2])

        min_oscilace = np.min(oscilace_array[::2])
        max_oscilace = np.max(oscilace_array[1::2])
        return min_proudy, max_proudy, min_napeti, max_napeti, min_oscilace, max_oscilace

    @staticmethod
    def get_min_max_for_oscilations():
        df = pd.read_csv(Config.DATA_LIMITS_CSV, sep=";")
        array = df.to_numpy()
        oscilace_array = array[:, 16:-1]

        list_of_mins_maxs = []
        for i in range(0, np.shape(oscilace_array)[1]-1, 2):
            list_of_mins_maxs.append(np.min(oscilace_array[:, i]))
            list_of_mins_maxs.append(np.max(oscilace_array[:, i+1]))

        min_max_array = np.array([np.array(list_of_mins_maxs[::2]), np.array(list_of_mins_maxs[1::2])])
        return min_max_array
