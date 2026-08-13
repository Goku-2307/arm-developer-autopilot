import os
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import onnxruntime as ort


class RuntimeValidator:
    """
    Validates that ONNX Runtime is properly installed and functional
    on the current system, with a focus on CPU execution provider.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._providers: Optional[List[str]] = None
        self._session: Optional[ort.InferenceSession] = None

    def validate_installation(self) -> Dict[str, Any]:
        """
        Validate that ONNX Runtime is installed and loaded correctly.

        Returns:
            Dict with validation status for each check.
        """
        results = {
            "onnx_runtime_installed": False,
            "cpu_execution_provider": False,
            "model_loaded": False,
            "inference_verified": False,
            "input_output_resolved": False,
        }

        # Check ONNX Runtime installation
        try:
            import onnxruntime
            results["onnx_runtime_installed"] = True
            results["onnxruntime_version"] = onnxruntime.__version__
        except ImportError:
            return results

        # Check CPU execution provider
        try:
            providers = ort.get_available_providers()
            results["cpu_execution_provider"] = "CPUExecutionProvider" in providers
            self._providers = providers
        except Exception:
            pass

        # If model path provided, validate model
        if self.model_path and Path(self.model_path).exists():
            try:
                session = ort.InferenceSession(
                    str(self.model_path),
                    providers=["CPUExecutionProvider"]
                )
                results["model_loaded"] = True
                self._session = session

                # Try inference
                input_info = session.get_inputs()[0]
                shape = []
                for dim in input_info.shape:
                    if isinstance(dim, int):
                        shape.append(dim)
                    else:
                        shape.append(1)

                input_data = np.random.rand(*shape).astype(np.float32)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    session.run(None, {input_info.name: input_data})

                results["inference_verified"] = True

                # Resolve input/output info
                try:
                    input_names = [inp.name for inp in session.get_inputs()]
                    output_names = [out.name for out in session.get_outputs()]
                    results["input_output_resolved"] = True
                    results["input_names"] = input_names
                    results["output_names"] = output_names
                except Exception:
                    pass

            except Exception:
                pass

        return results

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about available execution providers.

        Returns:
            Dict with provider information.
        """
        if self._providers is not None:
            return {"available_providers": self._providers}

        try:
            providers = ort.get_available_providers()
            return {"available_providers": providers}
        except Exception:
            return {"available_providers": []}

    def verify_model_compatibility(
        self, model_path: str
    ) -> Dict[str, Any]:
        """
        Verify that a specific ONNX model is compatible with CPU execution.

        Args:
            model_path: Path to the ONNX model file.

        Returns:
            Dict with compatibility verification results.
        """
        result = {
            "model_path": model_path,
            "loadable": False,
            "inference_works": False,
            "input_count": 0,
            "output_count": 0,
            "input_names": [],
            "output_names": [],
            "opset_version": None,
            "error": None,
        }

        try:
            session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            result["loadable"] = True

            inputs = session.get_inputs()
            outputs = session.get_outputs()

            result["input_count"] = len(inputs)
            result["output_count"] = len(outputs)
            result["input_names"] = [inp.name for inp in inputs]
            result["output_names"] = [out.name for out in outputs]

            # Get opset version if available
            if inputs:
                try:
                    result["opset_version"] = inputs[0].opset_version
                except Exception:
                    pass

            # Try inference
            try:
                for inp in inputs:
                    shape = []
                    for dim in inp.shape:
                        if isinstance(dim, int):
                            shape.append(dim)
                        else:
                            shape.append(1)

                input_data = np.random.rand(*shape).astype(np.float32)
                session.run(None, {inputs[0].name: input_data})
                result["inference_works"] = True
            except Exception as e:
                result["error"] = str(e)

        except Exception as e:
            result["error"] = str(e)

        return result


class ONNXModelInfo:
    """
    Extract metadata from an ONNX model for display and reporting.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.info: Optional[Dict[str, Any]] = None
        self._extract()

    def _extract(self):
        """Extract model metadata."""
        try:
            import onnx
            model = onnx.load(self.model_path)

            self.info = {
                "model_name": os.path.basename(self.model_path),
                "model_size_mb": round(os.path.getsize(self.model_path) / (1024 * 1024), 2),
                "input_count": len(model.graph.input),
                "output_count": len(model.graph.output),
                "input_names": [
                    inp.name for inp in model.graph.input
                ],
                "output_names": [
                    out.name for out in model.graph.output
                ],
                "opset_version": model.opset_import[0].version
                if model.opset_import
                else None,
                "framework": "ONNX",
            }

            # Extract input shapes and dtypes
            input_details = []
            for inp in model.graph.input:
                dims = []
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        dims.append(f"?{dim.dim_param}")
                    elif dim.dim_value:
                        dims.append(str(dim.dim_value))
                    else:
                        dims.append("?")
                input_details.append(
                    {
                        "name": inp.name,
                        "shape": dims,
                        "dtype": inp.type.tensor_type.elem_type,
                    }
                )
            self.info["input_details"] = input_details

        except Exception as e:
            self.info = {"error": str(e)}

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the model information."""
        if self.info:
            return self.info
        return {"error": "Could not extract model information"}