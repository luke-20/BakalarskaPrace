import numpy as np

from src.config import Config
from src.data_limits import DataLimits


class Scaler:
    def scale_2d_array_per_column(self, dataset):
        min_max_array = DataLimits.get_min_max_for_oscilations()
        for column in range(np.shape(dataset)[1]):
            dataset[:, column] = self._scale_array(array=dataset[:, column], min_value=min_max_array[0, column], max_value=min_max_array[1, column])
        return dataset.astype(np.int8)

    def scale_dataset_per_column(self, dataset):
        if len(np.shape(dataset)) < 2:
            raise Exception("Invalid array shape. Use 2D or 3D array.")

        min_max_array = DataLimits.get_min_max_for_oscilations()
        for timestep in range(Config.TIMESTEPS):
            for column in range(Config.NUMBER_OF_SAMPLE_COLUMNS):
                column_index = column + timestep*Config.NUMBER_OF_SAMPLE_COLUMNS
                dataset[:, column_index] = self._scale_array(array=dataset[:, column_index], min_value=min_max_array[0, column], max_value=min_max_array[1, column])
        return dataset.astype(np.int8)

    def scale_dataset(self, dataset):
        _, _, _, _, min_oscilace, max_oscilace = DataLimits.get_min_max_values()
        return self._scale_array(dataset, min_value=min_oscilace, max_value=max_oscilace).astype(np.int8)

    @staticmethod
    def _scale_array(array, min_value, max_value):
        normalized_array = (((array - min_value) / (max_value - min_value))*255)-128
        return normalized_array
