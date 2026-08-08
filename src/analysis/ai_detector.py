from pathlib import Path


class AIDetector:
    """
    Detects AI models inside a project.
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

    def detect_models(self):

        models = []

        for file in self.project_path.rglob("*"):

            if not file.is_file():
                continue

            suffix = file.suffix.lower()

            if suffix in self.supported_models:

                models.append({

                    "model_name": file.name,

                    "model_path": str(file),

                    "framework": self.supported_models[suffix],

                    "model_type": suffix.replace(".", "").upper()

                })

        return models