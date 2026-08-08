import os
import time

import psutil
import numpy as np
import onnxruntime as ort


class Benchmark:
    """
    Benchmarks an ONNX model.

    Returns:
        - Latency
        - Memory Usage
        - Model Size
    """

    def __init__(self, model_path, threads=1):

        self.model_path = model_path
        self.threads = threads

    def benchmark(self):

        options = ort.SessionOptions()

        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1

        session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"]
        )

        input_info = session.get_inputs()[0]

        shape = []

        for dim in input_info.shape:

            if isinstance(dim, int):
                shape.append(dim)
            else:
                shape.append(1)

        input_data = np.random.rand(*shape).astype(np.float32)

        process = psutil.Process()

        memory_before = process.memory_info().rss / (1024 * 1024)

        # Warmup
        for _ in range(2):
            session.run(
                None,
                {input_info.name: input_data}
            )

        runs = []

        for _ in range(5):

            start = time.perf_counter()

            session.run(
                None,
                {input_info.name: input_data}
            )

            end = time.perf_counter()

            runs.append((end - start) * 1000)

        memory_after = process.memory_info().rss / (1024 * 1024)

        latency = round(float(np.mean(runs)), 2)

        memory = round(memory_after - memory_before, 2)

        model_size = round(
            os.path.getsize(self.model_path) / (1024 * 1024),
            2
        )

        return {

            "latency": latency,

            "memory": memory,

            "model_size": model_size,

            "threads": self.threads

        }