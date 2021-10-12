import keras
import onnx
import keras2onnx


class ModelConversion:
    @staticmethod
    def keras_model_to_onnx(keras_model_path, onnx_model_path):
        keras_model = keras.models.load_model(keras_model_path)
        onnx_model = keras2onnx.convert_keras(keras_model, keras_model_path)
        onnx.save_model(onnx_model, onnx_model_path)
