import time
import psutil
import numpy as np
import onnxruntime as ort


class Benchmark:

    def __init__(self, model_path):

        self.model_path = model_path

    def benchmark(self):

        session = ort.InferenceSession(self.model_path)

        input_name = session.get_inputs()[0].name

        input_shape = session.get_inputs()[0].shape

        shape = []

        for dim in input_shape:

            if isinstance(dim, int):

                shape.append(dim)

            else:

                shape.append(1)

        dummy_input = np.random.rand(*shape).astype(np.float32)

        process = psutil.Process()

        memory_before = process.memory_info().rss

        cpu_before = psutil.cpu_percent()

        start = time.perf_counter()

        session.run(None, {input_name: dummy_input})

        end = time.perf_counter()

        cpu_after = psutil.cpu_percent()

        memory_after = process.memory_info().rss

        latency = (end - start) * 1000

        memory_used = (memory_after - memory_before) / 1024 / 1024

        return {

            "latency": round(latency, 2),

            "memory": round(memory_used, 2),

            "cpu": cpu_after

        }