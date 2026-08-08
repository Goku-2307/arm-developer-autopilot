from src.optimization.optimizer import ModelOptimizer

optimizer = ModelOptimizer(
    "models/model.onnx"
)

print("=" * 50)
print("FP32")
print("=" * 50)

print(
    optimizer.optimize("FP32")
)

print()

print("=" * 50)
print("INT8")
print("=" * 50)

print(
    optimizer.optimize("INT8")
)