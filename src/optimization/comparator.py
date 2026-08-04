class BenchmarkComparator:

    def compare(self, before, after):

        latency_before = before["latency"]
        latency_after = after["latency"]

        memory_before = before["memory"]
        memory_after = after["memory"]

        latency_gain = (
            (latency_before - latency_after)
            / latency_before
        ) * 100

        memory_gain = (
            (memory_before - memory_after)
            / memory_before
        ) * 100

        return {

            "Latency Before": latency_before,

            "Latency After": latency_after,

            "Latency Improvement (%)": round(latency_gain,2),

            "Memory Before": memory_before,

            "Memory After": memory_after,

            "Memory Improvement (%)": round(memory_gain,2)

        }