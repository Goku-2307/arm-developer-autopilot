from pathlib import Path


class AIDetector:

    def __init__(self, project_path):

        self.project_path = Path(project_path)

        self.supported_models = {
            ".onnx": "ONNX",
            ".tflite": "TensorFlow Lite",
            ".pt": "PyTorch",
            ".pth": "PyTorch"
        }

    def detect_models(self):

        models = []

        for file in self.project_path.rglob("*"):

            suffix = file.suffix.lower()

            if suffix in self.supported_models:

                models.append({

                    "model_name": file.name,

                    "model_path": str(file),

                    "model_type": self.supported_models[suffix]

                })

        return models