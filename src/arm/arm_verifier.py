from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class VerificationResult:
    is_arm: bool = False
    architecture: str = ""
    supported: bool = False
    reason: str = ""
    hardware_profile: Dict[str, Any] = None


class ARMVerifier:
    """
    Central hardware verification layer for ARM environments.

    Verifies that the current system is ARM64/aarch64 and meets
    the minimum requirements for real ARM CPU benchmarking.
    """

    def __init__(self, require_arm: bool = True, min_cores: int = 1, min_memory_mb: int = 256):

        self.require_arm = require_arm
        self.min_cores = min_cores
        self.min_memory_mb = min_memory_mb
        self._profile: Optional[Any] = None
        self._result: Optional[VerificationResult] = None

    def _get_hardware_profile(self) -> Any:
        """Lazy-load the hardware profile."""
        if self._profile is None:
            from src.arm.hardware_profiler import profile
            self._profile = profile()
        return self._profile

    def verify(self) -> VerificationResult:
        """
        Run ARM environment verification.

        Returns:
            VerificationResult with status and reasoning.
        """
        profile_data = self._get_hardware_profile()

        if not self.require_arm:
            return VerificationResult(
                is_arm=False,
                architecture=profile_data.architecture,
                supported=True,
                reason="ARM verification skipped (development mode)",
                hardware_profile=asdict(profile_data)
            )

        # Check if ARM64/aarch64
        if not profile_data.is_arm:
            return VerificationResult(
                is_arm=False,
                architecture=profile_data.architecture,
                supported=False,
                reason="ARM64/aarch64 hardware is required for real ARM benchmarking",
                hardware_profile=asdict(profile_data)
            )

        # Check minimum cores
        if profile_data.physical_cores < self.min_cores:
            return VerificationResult(
                is_arm=True,
                architecture=profile_data.architecture,
                supported=False,
                reason=f"Detected {profile_data.physical_cores} core(s), minimum {self.min_cores} required for benchmarking",
                hardware_profile=asdict(profile_data)
            )

        # Check minimum memory
        if profile_data.memory_total_mb < self.min_memory_mb:
            return VerificationResult(
                is_arm=True,
                architecture=profile_data.architecture,
                supported=False,
                reason=f"Detected {profile_data.memory_total_mb} MB RAM, minimum {self.min_memory_mb} MB required for benchmarking",
                hardware_profile=asdict(profile_data)
            )

        return VerificationResult(
            is_arm=True,
            architecture=profile_data.architecture,
            supported=True,
            reason="ARM64 Linux environment detected and verified",
            hardware_profile=asdict(profile_data)
        )

    def reset_profile_cache(self):
        """Clear the cached hardware profile."""
        self._profile = None
        self._result = None

    def get_status_badge(self) -> str:
        """
        Get the display badge for the dashboard.

        Returns:
            String like '🟢 ARM VERIFIED' or '🔴 ARM NOT DETECTED'
        """
        result = self.verify()
        if result.is_arm and result.supported:
            return "🟢 ARM VERIFIED"
        elif not result.is_arm:
            return "🔴 ARM NOT DETECTED"
        else:
            return "🟡 ARM ENVIRONMENT INCOMPLETE"


# Convenience function for quick verification
def quick_verify(
    require_arm: bool = True,
    min_cores: int = 1,
    min_memory_mb: int = 256,
) -> VerificationResult:
    """
    Quick verification function.

    Example:
        result = quick_verify()
        if result.is_arm and result.supported:
            print("ARM environment ready for benchmarking")
    """
    verifier = ARMVerifier(
        require_arm=require_arm,
        min_cores=min_cores,
        min_memory_mb=min_memory_mb
    )
    return verifier.verify()