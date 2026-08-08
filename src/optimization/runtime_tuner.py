import multiprocessing

from src.optimization.benchmark import Benchmark


class RuntimeTuner:
    """
    Finds the optimal CPU thread configuration
    for ONNX Runtime inference.
    """

    def __init__(self, model_path):

        self.model_path = model_path

        self.cpu_count = multiprocessing.cpu_count()

    def candidate_threads(self):

        candidates = []

        if self.cpu_count >= 1:
            candidates.append(1)

        if self.cpu_count >= 2:
            candidates.append(2)

        if self.cpu_count >= 4:
            candidates.append(4)

        if self.cpu_count >= 6:
            candidates.append(6)

        if self.cpu_count >= 8:
            candidates.append(8)

        if self.cpu_count >= 12:
            candidates.append(12)

        if self.cpu_count >= 16:
            candidates.append(16)

        return candidates

    def tune(self):

        print("\n" + "=" * 60)
        print("ARM Runtime Thread Tuning")
        print("=" * 60)

        results = []

        for threads in self.candidate_threads():

            print(f"\nTesting {threads} Thread(s)...")

            benchmark = Benchmark(

                self.model_path,

                threads=threads

            )

            metrics = benchmark.benchmark()

            metrics["threads"] = threads

            results.append(metrics)

            print(f"Latency : {metrics['latency']} ms")
            print(f"Memory  : {metrics['memory']} MB")

        best = min(

            results,

            key=lambda x: x["latency"]

        )

        print("\n" + "=" * 60)
        print("BEST THREAD CONFIGURATION")
        print("=" * 60)

        print(f"Threads : {best['threads']}")
        print(f"Latency : {best['latency']} ms")

        return {

            "best_threads": best["threads"],

            "best_latency": best["latency"],

            "results": results

        }