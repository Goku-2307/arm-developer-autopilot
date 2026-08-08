class BenchmarkComparator:
    """
    Compares FP32 and optimized model benchmarks.
    """

    def compare(self, before, after):

        latency_before = before["latency"]
        latency_after = after["latency"]

        memory_before = before["memory"]
        memory_after = after["memory"]

        latency_improvement = round(
            ((latency_before - latency_after) / latency_before) * 100,
            2
        )

        memory_improvement = round(
            ((memory_before - memory_after) / memory_before) * 100,
            2
        ) if memory_before != 0 else 0

        return {

            "Latency Before": latency_before,
            "Latency After": latency_after,
            "Latency Improvement (%)": latency_improvement,

            "Memory Before": memory_before,
            "Memory After": memory_after,
            "Memory Improvement (%)": memory_improvement

        }