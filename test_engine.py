from src.optimization.optimization_engine import OptimizationEngine

engine = OptimizationEngine(
    "models/model.onnx"
)

result = engine.optimize()

print()

print(result["best_configuration"])