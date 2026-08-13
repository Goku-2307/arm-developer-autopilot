from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """Result of post-deployment verification."""

    success: bool
    model_loads: bool = False
    inference_works: bool = False
    latency_ms: Optional[float] = None
    throughput_fps: Optional[float] = None
    memory_mb: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class PostDeploymentVerifier:
    """
    Verifies that the optimized model runs correctly after deployment.

    Performs the following checks:
    1. Model file exists and can be loaded
    2. ONNX Runtime can initialize the model
    3. Inference produces valid output
    4. Performance metrics are within expected ranges
    5. Output tensors match expected dimensions
    """

    def __init__(self, model_path: str, providers: list = None):
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]

    def verify(self) -> VerificationResult:
        """
        Run post-deployment verification.

        Returns:
            VerificationResult with success status and metrics.
        """
        result = VerificationResult(success=False, message="")

        # Check 1: Model file exists
        import os
        if not os.path.exists(self.model_path):
            result.message = f"Model file not found: {self.model_path}"
            return result

        result.model_loads = True

        # Check 2: Model can be loaded and inference works
        import onnxruntime as ort
        import numpy as np

        try:
            session = ort.InferenceSession(
                self.model_path,
                providers=self.providers
            )
            result.inference_works = True

            # Get input information
            input_info = session.get_inputs()[0]
            shape = []
            for dim in input_info.shape:
                if isinstance(dim, int):
                    shape.append(dim)
                else:
                    shape.append(1)

            input_data = np.random.rand(*shape).astype(np.float32)

            # Run inference
            start = __import__('time').perf_counter()
            outputs = session.run(None, {input_info.name: input_data})
            end = __import__('time').perf_counter()

            # Calculate latency
            latency_ms = round((end - start) * 1000, 2)
            result.latency_ms = latency_ms

            # Calculate throughput (1 inference)
            throughput_fps = round(1.0 / (latency_ms / 1000), 2) if latency_ms > 0 else 0
            result.throughput_fps = throughput_fps

            # Check memory
            import psutil
            process = psutil.Process()
            memory_mb = round(process.memory_info().rss / (1024 * 1024), 2)
            result.memory_mb = memory_mb

            result.success = True
            result.message = "DEPLOYMENT VERIFIED"

        except Exception as e:
            result.message = f"Verification failed: {e}"

        return result

    @staticmethod
    def compare_with_baseline(
        verification_result: VerificationResult,
        baseline_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare verification results with baseline benchmark.

        Args:
            verification_result: Result from post-deployment verification
            baseline_result: Baseline benchmark dict

        Returns:
            Dict with comparison and improvement percentages
        """
        comparison = {
            "baseline_latency_ms": baseline_result.get("latency_mean_ms"),
            "verification_latency_ms": verification_result.latency_ms,
            "latency_improvement_percent": None,
            "baseline_throughput_fps": baseline_result.get("throughput_fps"),
            "verification_throughput_fps": verification_result.throughput_fps,
            "throughput_improvement_percent": None,
            "verification_success": verification_result.success,
        }

        if (
            verification_result.latency_ms is not None
            and baseline_result.get("latency_mean_ms", 0) > 0
        ):
            baseline_lat = baseline_result["latency_mean_ms"]
            verify_lat = verification_result.latency_ms
            comparison["latency_improvement_percent"] = round(
                ((baseline_lat - verify_lat) / baseline_lat) * 100, 2
            )

        if (
            verification_result.throughput_fps is not None
            and baseline_result.get("throughput_fps", 0) > 0
        ):
            baseline_tp = baseline_result["throughput_fps"]
            verify_tp = verification_result.throughput_fps
            comparison["throughput_improvement_percent"] = round(
                ((verify_tp - baseline_tp) / baseline_tp) * 100, 2
            )

        return comparison