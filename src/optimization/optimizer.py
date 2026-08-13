import os
import shutil
from pathlib import Path

from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType
)


class ModelOptimizer:
    """
    Optimizes AI models for ARM CPUs.

    Current Support
    ----------------
    • FP32 (Original)
    • INT8 Dynamic Quantization
    """

    def __init__(self, model_path):
        self.model_path = str(Path(model_path).resolve())
        model_dir = os.path.dirname(self.model_path)
        self.output_dir = os.path.join(model_dir, "optimized_models")
        os.makedirs(self.output_dir, exist_ok=True)

    def optimize(self, quantization="FP32"):

        quantization = quantization.upper()

        if quantization == "FP32":

            return self.copy_original()

        elif quantization == "INT8":

            return self.quantize_int8()

        raise ValueError(f"Unsupported quantization: {quantization}")

    def copy_original(self):

        filename = os.path.basename(self.model_path)

        destination = os.path.join(
            self.output_dir,
            filename
        )

        shutil.copy2(
            self.model_path,
            destination
        )

        return {

            "optimized_model": os.path.abspath(destination),

            "quantization": "FP32",

            "optimized": False

        }

    def quantize_int8(self):

        filename = os.path.basename(self.model_path)

        name = os.path.splitext(filename)[0]

        output_model = os.path.join(

            self.output_dir,

            name + "_int8.onnx"

        )

        quantize_dynamic(

            model_input=self.model_path,

            model_output=output_model,

            weight_type=QuantType.QInt8

        )

        return {

            "optimized_model": os.path.abspath(output_model),

            "quantization": "INT8",

            "optimized": True

        }