import time
import numpy as np
import onnxruntime as ort


class RuntimeTuner:

    def __init__(self, model_path):

        self.model_path = model_path

    def benchmark(self, threads):

        options = ort.SessionOptions()

        options.intra_op_num_threads = threads

        session = ort.InferenceSession(
            self.model_path,
            sess_options=options
        )

        input_name = session.get_inputs()[0].name

        input_shape = session.get_inputs()[0].shape

        shape = []

        for dim in input_shape:

            if isinstance(dim, int):

                shape.append(dim)

            else:

                shape.append(1)

        dummy = np.random.rand(*shape).astype(np.float32)

        # Warm-up
        for _ in range(10):
            session.run(None, {input_name: dummy})

        times = []

        for _ in range(50):

            start = time.perf_counter()

            session.run(None, {input_name: dummy})

            end = time.perf_counter()

            times.append((end - start) * 1000)

        return np.mean(times)

    def find_best_threads(self):

        candidates = [1, 2, 4, 8]

        results = {}

        for t in candidates:

            try:

                latency = self.benchmark(t)

                results[t] = latency

            except Exception:

                pass

        best = min(results, key=results.get)

        return best, results