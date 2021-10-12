from src.tfmodel import TFLiteModel
from src.nn_model import NNHandler
from src.train_test import Trainer, Tester, TestIntegrity
from src.model_conversion import ModelConversion
from src.logger import TestAnalysis, TestAnalysisTFLite
from src.dataset import Dataset, TrainingSet
from src.nn_model import NNModel
from src.data_visualizer import DataVisualizer

from src.data_limits import DataLimits
from src.config import Config


def mat_data_visualization(file_path):
    data = DataVisualizer(file_path)
    data.plot_napeti()
    data.plot_proudy()
    data.plot_proudy_240()
    data.plot_oscilace()


def visualize_dataset_quantization(file_path):
    data = DataVisualizer(file_path)
    data.plot_oscilace()
    data.plot_oscilace_scaled()
    data.plot_oscilace_scaled_per_column()


def data_limits_to_csv():
    limits = DataLimits()
    limits.get_min_max_values()


def test_tflite():
    data = TrainingSet([".\data\AI_InputData\Measurement_HS_Procesed\FaultSubSys1\GrabbedData_Fault_3z_U_RampUp_0_5Nm_12-14-2020_13-02_Em_0.mat", #vybrat dostupný soubor
                        ".\data\AI_InputData\Measurement_HS_Procesed\\Normal\GrabbedData_Normal_RampUp_0_5Nm_12-14-2020_12-26_Em_0.mat"])
    tflite = TFLiteModel(data)
    tflite.print_predicted_values()
    tflite.evaluate_model()
    tflite.speed_test()


def test_keras():
    data = TrainingSet([".\data\AI_InputData\Measurement_HS_Procesed\FaultSubSys1\GrabbedData_Fault_3z_U_RampUp_0_5Nm_12-14-2020_13-02_Em_0.mat", #vybrat dostupný soubor
                        ".\data\AI_InputData\Measurement_HS_Procesed\\Normal\GrabbedData_Normal_RampUp_0_5Nm_12-14-2020_12-26_Em_0.mat"]) #TODO: prepsat na soubory, ktere budou na cd
    keras = NNHandler(data)
    keras.print_predicted_values()
    keras.evaluate_model()
    keras.speed_test()


def integrity_test_and_analysis():
    integrity_tester = TestIntegrity()
    integrity_tester.test_result()

    print(f"model: {Config.MODEL_FOLDER}")
    print(f"analyza Keras modelu:")
    analyza_testu = TestAnalysis()
    print(analyza_testu.get_nb_of_files_by_acc())
    print(f"total accuracy: {analyza_testu.overall_test_accuracy}")

    print(f"\n\nanalyza TFLite modelu:")
    analyza_testu_tf = TestAnalysisTFLite()
    print(analyza_testu_tf.get_nb_of_files_by_acc())
    print(f"total accuracy: {analyza_testu_tf.overall_test_accuracy}")


if __name__ == "__main__":

    """
    # rychlý test modelů
    test_keras()"""
    test_tflite()

    """"""
    # SPUSTENI TRENINKU 
    trener = Trainer()
    trener.batch_train()


    """"""
    # SPUSTENI TESTU
    tester = Tester()
    tester.evaluate_per_file()

    integrity_test_and_analysis()

    # visualize_dataset_quantization(".\data\AI_InputData\Measurement_HS_Procesed\FaultSubSys1\GrabbedData_Fault_3z_U_RampUp_0_5Nm_12-14-2020_13-02_Em_0.mat")







