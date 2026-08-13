from typing import Dict, Any


class BenchmarkComparator:
    """
    Compares baseline and optimized model benchmarks.

    Calculates improvement percentages for latency, memory, model size,
    and throughput. Handles both "lower is better" and "higher is better"
    metrics correctly.
    """

    @staticmethod
    def compare(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare baseline and optimized benchmark results.

        Args:
            before: Baseline benchmark result dict.
            after: Optimized benchmark result dict.

        Returns:
            Dict with comparison results including improvement percentages.
            All percentage values represent:
                - Positive values: improvement (metric got better)
                - Negative values: degradation (metric got worse)
        """

        latency_before = before.get("latency_mean_ms", 0)
        latency_after = after.get("latency_mean_ms", 0)

        memory_before = before.get("memory", 0)
        memory_after = after.get("memory", 0)

        model_size_before = before.get("model_size_mb", 0)
        model_size_after = after.get("model_size_mb", 0)

        throughput_before = before.get("throughput_fps", 0)
        throughput_after = after.get("throughput_fps", 0)

        # Latency improvement: positive = improvement (lower is better)
        if latency_before > 0:
            latency_improvement = round(
                ((latency_before - latency_after) / latency_before) * 100,
                2,
            )
        else:
            latency_improvement = 0

        # Memory improvement: positive = improvement (lower is better)
        if memory_before > 0:
            memory_improvement = round(
                ((memory_before - memory_after) / memory_before) * 100,
                2,
            )
        else:
            memory_improvement = 0

        # Model size reduction: positive = improvement (smaller is better)
        if model_size_before > 0:
            model_size_reduction = round(
                ((model_size_before - model_size_after) / model_size_before) * 100,
                2,
            )
        else:
            model_size_reduction = 0

        # Throughput improvement: positive = improvement (higher is better)
        if throughput_before > 0:
            throughput_improvement = round(
                ((throughput_after - throughput_before) / throughput_before) * 100,
                2,
            )
        else:
            throughput_improvement = 0

        return {
            "latency_before_ms": latency_before,
            "latency_after_ms": latency_after,
            "latency_improvement_percent": latency_improvement,

            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_improvement_percent": memory_improvement,

            "model_size_before_mb": model_size_before,
            "model_size_after_mb": model_size_after,
            "model_size_reduction_percent": model_size_reduction,

            "throughput_before_fps": throughput_before,
            "throughput_after_fps": throughput_after,
            "throughput_improvement_percent": throughput_improvement,
        }