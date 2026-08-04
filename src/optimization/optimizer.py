from pathlib import Path

from onnxruntime.quantization import quantize_dynamic, QuantType


class ModelOptimizer:

    def __init__(self, input_model):

        self.input_model = input_model

    def quantize(self):

        output_dir = Path("optimized_models")

        output_dir.mkdir(exist_ok=True)

        output_model = output_dir / (
        Path(self.input_model).stem + "_int8.onnx"
   )

        quantize_dynamic (
            model_input=self.input_model,
            model_output=str(output_model),
            weight_type=QuantType.QInt8,
        )

        return str(output_model)