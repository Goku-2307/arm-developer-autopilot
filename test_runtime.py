from src.optimization.runtime_tuner import RuntimeTuner

tuner = RuntimeTuner(
    "models/model.onnx"
)

result = tuner.tune()

print()

print(result)