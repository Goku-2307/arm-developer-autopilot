from pathlib import Path
from typing import List, Dict, Any
import os


class AIDetector:
    """
    Detects AI models inside a project.
    Supports ONNX, TFLite, PyTorch, and TorchScript formats.
    """

    def __init__(self, project_path):

        self.project_path = Path(project_path)

        self.supported_models = {

            ".onnx": "ONNX Runtime",

            ".tflite": "TensorFlow Lite",

            ".pt": "PyTorch",

            ".pth": "PyTorch",

            ".torchscript": "TorchScript"

        }

    def _get_model_size_mb(self, model_path: str) -> float:
        """Get model file size in MB."""
        try:
            return round(os.path.getsize(model_path) / (1024 * 1024), 2)
        except Exception:
            return 0.0

    def _detect_onnx_metadata(self, model_path: str) -> Dict[str, Any]:
        """Extract metadata from ONNX model if available."""
        try:
            import onnx

            model = onnx.load(model_path)

            info = {
                "input_count": len(model.graph.input),
                "output_count": len(model.graph.output),
            }

            # Extract input names and shapes
            input_details = []
            for inp in model.graph.input:
                name = inp.name
                # Extract shape
                shape = []
                if hasattr(inp, 'type') and hasattr(inp.type, 'tensor_type'):
                    shape_dim = inp.type.tensor_type.shape.dim
                    for dim in shape_dim:
                        if dim.dim_param:
                            shape.append(f"?{dim.dim_param}")
                        elif dim.dim_value:
                            shape.append(str(dim.dim_value))
                        else:
                            shape.append("?")
                else:
                    shape = ["?"]

                input_details.append({
                    "name": name,
                    "shape": shape,
                })

            info["input_names"] = [inp.name for inp in model.graph.input]
            info["output_names"] = [out.name for out in model.graph.output]

            # Extract opset version
            if model.opset_import:
                info["opset_version"] = model.opset_import[0].version
            else:
                info["opset_version"] = None

            return info

        except Exception:
            return {
                "input_count": 0,
                "output_count": 0,
                "input_names": [],
                "output_names": [],
                "opset_version": None,

            }

    def detect_models(self):
        """
        Detect AI models inside a project.

        Returns a list of dicts with model information including:
        - model_name, model_path, framework, model_type
        - size_mb, input_count, output_count
        - input_names, output_names (when available)
        - input_shapes, output_shapes (when available)
        - opset_version (when available)
        """

        models = []

        for file in self.project_path.rglob("*"):

            if not file.is_file():
                continue

            suffix = file.suffix.lower()

            if suffix in self.supported_models:

                model_type = suffix.replace(".", "").upper()
                framework = self.supported_models[suffix]
                model_path = str(file)
                model_size = self._get_model_size_mb(model_path)

                model_info = {
                    "model_name": file.name,
                    "model_path": model_path,
                    "format": framework,
                    "model_type": model_type,
                    "size_mb": model_size,
                    "input_count": 0,
                    "output_count": 0,
                    "input_names": [],
                    "output_names": [],
                    "input_shapes": [],
                    "output_shapes": [],
                    "opset_version": None,

                }

                # Extract format-specific metadata
                if framework == "ONNX Runtime" and model_size > 0:
                    onnx_meta = self._detect_onnx_metadata(model_path)
                    model_info["input_count"] = onnx_meta.get("input_count", 0)
                    model_info["output_count"] = onnx_meta.get("output_count", 0)
                    model_info["input_names"] = onnx_meta.get("input_names", [])
                    model_info["output_names"] = onnx_meta.get("output_names", [])
                    model_info["opset_version"] = onnx_meta.get("opset_version")

                    # Build shape lists
                    for inp in model_info["input_names"]:
                        # Try to get shape from the model file
                        shape_info = self._infer_input_shape(model_path, inp)
                        model_info["input_shapes"].append(shape_info)

                    for out in model_info["output_names"]:
                        shape_info = self._infer_output_shape(model_path, out)
                        model_info["output_shapes"].append(shape_info)
                else:
                    # For non-ONNX formats, set basic counts
                    model_info["input_count"] = 1
                    model_info["output_count"] = 1

                models.append(model_info)

        return models

    def _infer_input_shape(self, model_path: str, input_name: str) -> List[int]:
        """Attempt to infer input shape for a model."""
        try:
            import onnx
            model = onnx.load(model_path)
            for inp in model.graph.input:
                if inp.name == input_name:
                    shape = []
                    for dim in inp.type.tensor_type.shape.dim:
                        if dim.dim_value:
                            shape.append(dim.dim_value)
                        else:
                            shape.append(1)
                    return shape
        except Exception:
            pass
        return [1, 3, 224, 224]

    def _infer_output_shape(self, model_path: str, output_name: str) -> List[int]:
        """Attempt to infer output shape for a model."""
        try:
            import onnx
            model = onnx.load(model_path)
            for out in model.graph.output:
                if out.name == output_name:
                    shape = []
                    for dim in out.type.tensor_type.shape.dim:
                        if dim.dim_value:
                            shape.append(dim.dim_value)
                        else:
                            shape.append(1)
                    return shape
        except Exception:
            pass
        return [1, 1000]