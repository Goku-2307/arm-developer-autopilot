from src.optimization.optimizer import ModelOptimizer
from src.optimization.benchmark import Benchmark


class OptimizationSearch:

    def __init__(self, model_path):

        self.model_path = model_path

    def execute(self, candidates):

        print()
        print("=" * 60)
        print("Optimization Search Started")
        print("=" * 60)

        results = []

        optimizer = ModelOptimizer(self.model_path)

        # --------------------------------------------------
        # Generate models ONLY ONCE
        # --------------------------------------------------

        fp32_model = optimizer.optimize("FP32")["optimized_model"]

        int8_model = optimizer.optimize("INT8")["optimized_model"]

        # --------------------------------------------------

        for candidate in candidates:

            print(f"\nRunning Candidate {candidate['candidate_id']}")

            if candidate["quantization"] == "FP32":

                model = fp32_model

            else:

                model = int8_model

            benchmark = Benchmark(

                model,

                threads=candidate["threads"]

            )

            benchmark_result = benchmark.benchmark()

            benchmark_result["candidate_id"] = candidate["candidate_id"]

            benchmark_result["quantization"] = candidate["quantization"]

            benchmark_result["graph_optimization"] = candidate["graph_optimization"]

            benchmark_result["execution_mode"] = candidate["execution_mode"]

            results.append(benchmark_result)

        print()
        print("=" * 60)
        print("Optimization Search Complete")
        print("=" * 60)

        return results