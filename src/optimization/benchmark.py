import os
import time
import numpy as np
import onnxruntime as ort
import psutil
from typing import Dict, Any, List


class Benchmark:
    """
    Benchmarks an ONNX model on the ARM CPU.

    Collects comprehensive inference metrics including latency,
    throughput, memory usage, and model size.

    Attributes:
        model_path: Path to the ONNX model file.
        threads: Number of CPU threads to use for inference.
        warmup_iterations: Number of warmup iterations before benchmarking.
        benchmark_iterations: Number of iterations for latency measurement.
    """

    DEFAULT_WARMUP = 10
    DEFAULT_BENCHMARK = 50

    def __init__(self, model_path: str, threads: int = 1):
        self.model_path = model_path
        self.threads = threads
        self.warmup_iterations = self.DEFAULT_WARMUP
        self.benchmark_iterations = self.DEFAULT_BENCHMARK

    def set_warmup(self, count: int):
        """Set the number of warmup iterations."""
        self.warmup_iterations = max(0, count)

    def set_benchmark(self, count: int):
        """Set the number of benchmark iterations."""
        self.benchmark_iterations = max(1, count)

    def benchmark(self) -> Dict[str, Any]:
        """
        Execute the benchmark and return structured results.

        Returns:
            Dict with the following keys:
                - latency_mean_ms: Mean latency in milliseconds
                - latency_median_ms: Median latency in milliseconds
                - latency_p95_ms: P95 latency in milliseconds
                - latency_min_ms: Min latency in milliseconds
                - latency_max_ms: Max latency in milliseconds
                - latency_std_ms: Standard deviation of latency
                - throughput_fps: Throughput in frames per second
                - memory_mb: Memory usage in MB (RSS delta)
                - model_size_mb: Model file size in MB
                - threads: Number of threads used
        """
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1

        session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

        input_info = session.get_inputs()[0]

        # Calculate input shape
        shape = []
        for dim in input_info.shape:
            if isinstance(dim, int):
                shape.append(dim)
            else:
                shape.append(1)

        # Prepare input data
        input_data = np.random.rand(*shape).astype(np.float32)

        process = psutil.Process()

        # --- Warmup ---
        for _ in range(self.warmup_iterations):
            session.run(None, {input_info.name: input_data})

        # --- Measure memory before ---
        memory_before = process.memory_info().rss / (1024 * 1024)

        # --- Benchmark iterations ---
        runs: List[float] = []
        for _ in range(self.benchmark_iterations):
            start = time.perf_counter()
            session.run(None, {input_info.name: input_data})
            end = time.perf_counter()
            runs.append((end - start) * 1000)  # Convert to milliseconds

        # --- Measure memory after ---
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_delta = round(memory_after - memory_before, 2)

        # --- Calculate latency statistics ---
        latency_mean = round(float(np.mean(runs)), 4)
        latency_median = round(float(np.median(runs)), 4)
        latency_p95 = round(float(np.percentile(runs, 95)), 4)
        latency_min = round(float(np.min(runs)), 4)
        latency_max = round(float(np.max(runs)), 4)
        latency_std = round(float(np.std(runs)), 4)

        # --- Throughput ---
        throughput = round(
            self.benchmark_iterations / sum(runs) * 1000,
            2
        )

        # --- Model size ---
        model_size = round(
            os.path.getsize(self.model_path) / (1024 * 1024),
            2
        )

        return {
            "latency_mean_ms": latency_mean,
            "latency_median_ms": latency_median,
            "latency_p95_ms": latency_p95,
            "latency_min_ms": latency_min,
            "latency_max_ms": latency_max,
            "latency_std_ms": latency_std,
            "throughput_fps": throughput,
            "memory_mb": memory_delta,
            "model_size_mb": model_size,
            "threads": self.threads,
            "warmup_iterations": self.warmup_iterations,
            "benchmark_iterations": self.benchmark_iterations
        }