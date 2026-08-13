from typing import Any, List, Dict

from src.optimization.optimizer import ModelOptimizer
from src.optimization.benchmark import Benchmark
from pathlib import Path


class OptimizationSearch:
    """
    Executes optimization search over candidates on the ARM CPU.

    Responsibilities:
    - Validate candidate configurations
    - Prepare optimized models (FP32 and INT8)
    - Benchmark every candidate on the actual CPU
    - Collect metrics for every candidate
    - Gracefully continue if a candidate fails
    """

    def __init__(self, model_path: str):
        self.model_path = self._normalize_path(model_path)

    def _normalize_path(self, model_path: str) -> str:
        """Normalize model path; return the path string."""
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return str(p)

    def execute(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Execute benchmark for every candidate.

        Args:
            candidates: List of candidate configuration dicts.

        Returns:
            List of benchmark result dicts. Failed candidates include
            a "status": "FAILED" key with error information, and the
            run continues with remaining candidates.
        """

        print()
        print("=" * 60)
        print("Optimization Search Started")
        print("=" * 60)

        results = []

        # --------------------------------------------------
        # Generate optimized models ONCE (FP32 and INT8)
        # --------------------------------------------------

        from src.optimization.optimizer import ModelOptimizer

        optimizer = ModelOptimizer(self.model_path)

        # Generate FP32 and INT8 optimized models once
        fp32_result = optimizer.optimize("FP32")
        int8_result = optimizer.optimize("INT8")

        fp32_model = fp32_result["optimized_model"] if fp32_result else None
        int8_model = int8_result["optimized_model"] if int8_result else None

        for candidate in candidates:

            # Support both Candidate dataclass and dict formats
            if hasattr(candidate, 'get'):
                c_id = candidate.get("candidate_id", "?")
            else:
                c_id = candidate.candidate_id

            print(f"\nRunning Candidate {c_id}")

            # Support both Candidate dataclass and dict formats
            def _get(cand, key, default="?"):
                return cand.get(key, default) if hasattr(cand, 'get') else getattr(cand, key, default)

            result = {
                "candidate_id": _get(candidate, "candidate_id", "?"),
                "quantization": _get(candidate, "quantization", "?"),
                "graph_optimization": _get(candidate, "graph_optimization", "?"),
                "execution_mode": _get(candidate, "execution_mode", "?"),
                "threads": _get(candidate, "threads", 1),
                "status": "SUCCESS",
                "error": None,
            }

            # Select the appropriate optimized model based on quantization
            if _get(candidate, "quantization", "FP32") == "FP32":
                model = fp32_model
            else:
                model = int8_model

            if model is None:
                # Model could not be generated; mark as failed
                result["status"] = "FAILED"
                result["error"] = "Optimized model could not be generated"
                results.append(result)
                print(f"  ✗ Candidate {c_id}: FAILED - optimized model unavailable")
                continue

            model_path_str = (
                str(model)
                if isinstance(model, Path)
                else (
                    str(model["optimized_model"])
                    if isinstance(model, dict)
                    else str(model)
                )
            )

            try:
                benchmark = Benchmark(model_path_str,
                                      threads=_get(candidate, "threads", 1))

                benchmark_result = benchmark.benchmark()

                # Attach candidate metadata
                benchmark_result["candidate_id"] = _get(candidate, "candidate_id", "?")
                benchmark_result["quantization"] = _get(candidate, "quantization", "?")
                benchmark_result["graph_optimization"] = _get(candidate, "graph_optimization", "?")
                benchmark_result["execution_mode"] = _get(candidate, "execution_mode", "?")

                result.update(benchmark_result)
                results.append(result)

                print(
                    f"  ✓ Completed: "
                    f"latency={benchmark_result.get('latency_mean_ms', '?')}ms, "
                    f"throughput={benchmark_result.get('throughput_fps', '?')}fps"
                )

            except Exception as e:
                result["status"] = "FAILED"
                result["error"] = str(e)
                result["latency_mean_ms"] = None
                result["throughput_fps"] = None
                result["latency_mean_ms"] = None
                result["memory_mb"] = None
                result["model_size_mb"] = None
                result["throughput_fps"] = None
                results.append(result)

                print(f"  ✗ Candidate {c_id}: FAILED - {e}")

        print()
        print("=" * 60)
        print("Optimization Search Complete")
        print("=" * 60)

        return results
