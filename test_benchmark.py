from src.optimization.benchmark import Benchmark

benchmark = Benchmark(
    "models/model.onnx",
    threads=4
)

result = benchmark.benchmark()

print(result)