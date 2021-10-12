class Config:
    """ NN model & training """
    DENSE_NEURONS = [16, 8]
    TIMESTEPS = 1
    BATCH_SIZE = 256
    EPOCHS = 4

    NUMBER_OF_SAMPLE_COLUMNS = 16
    TEST_SAMPLES_RATIO = 0.3
    FILE_BATCH: int = 35

    MODEL_FOLDER = "./src/models/1timestep-16-8-2/"
    # MODEL_FOLDER = "./src/models/1timestep-16-8-4-2/"
    # MODEL_FOLDER = "./src/models/1timestep-32-16-2/"
    # MODEL_FOLDER = "./src/models/1timestep-64-32-16-2/"
    # MODEL_FOLDER = "./src/models/5timesteps-80-40-2/"
    # MODEL_FOLDER = "./src/models/5timesteps-160-80-2/"

    KERAS_MODEL_PATH = f"{MODEL_FOLDER}keras_model.h5"
    KERAS_FROM_ONNX_MODEL_PATH = f"{MODEL_FOLDER}keras_model_from_onnx.h5"

    """ Tests """

    TEST_LOG_CSV = f"{MODEL_FOLDER}files_test_log.csv"
    TEST_TFLITE_LOG_CSV = f"{MODEL_FOLDER}files_test_tflite_log.csv"
    TRAIN_LOG_CSV = f"{MODEL_FOLDER}files_train_log.csv"
    CONFIG_LOG_CSV = f"{MODEL_FOLDER}config_log.csv"

    SPEED_TEST_ITERATIONS = 5000

    """ TFLite model """
    TFLITE_MODEL_PATH = f"{MODEL_FOLDER}tflite_model.tflite"
    TFLITE_MODEL_LAYERS_EXCEL_FILE = f"{MODEL_FOLDER}models_layers.xlsx"

    """ ONNX model """
    ONNX_MODEL_PATH = f"{MODEL_FOLDER}onnx_model_from_keras.onnx"

    """ Datasets """
    NUMBER_OF_MAT_FILES = 384
    DATASET_SHAPE = (1, TIMESTEPS * NUMBER_OF_SAMPLE_COLUMNS)

    DATA_LIMITS_CSV = f"./data/data_limits.csv"

    SEED = 4
    DATA_FOLDER = ".\data\\"
    DATA_FOLDERS = [".\data\AI_InputData\Measurement_HS_Procesed\FaultSubSys1\\",
                    ".\data\AI_InputData\Measurement_HS_Procesed\FaultSubSys2\\",
                    ".\data\AI_InputData\Measurement_HS_Procesed\\Normal\\"]

    FILE_BATCH_COUNTER = 0

    @staticmethod
    def attributes_as_dictionary():
        attrs_dict = vars(Config)
        dictionary = ('; '.join("%s:%s" % item for item in attrs_dict.items()))
        list_of_attributes = dictionary.split(";")
        output_dictionary = dict(zip([attr.split(":")[0] for attr in list_of_attributes if attr.split(":")[0][1] != '_'],
                                     [attr.split(":")[1] for attr in list_of_attributes if attr.split(":")[0][1] != '_']))
        return output_dictionary
