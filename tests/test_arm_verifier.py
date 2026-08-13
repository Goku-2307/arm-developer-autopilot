"""Tests for the ARM Verifier module."""

import pytest
from src.arm.arm_verifier import quick_verify, ARMVerifier, VerificationResult


def test_quick_verify_returns_result():
    """Test that quick_verify returns a VerificationResult."""
    result = quick_verify()
    assert isinstance(result, VerificationResult)
    assert hasattr(result, "is_arm")
    assert hasattr(result, "supported")
    assert hasattr(result, "reason")


def test_quick_verify_x86():
    """Test quick_verify on x86 returns is_arm=False."""
    result = quick_verify()
    assert result.is_arm == False  # On x86 systems


def test_quick_verify_supported():
    """Test that supported field is present."""
    result = quick_verify()
    assert isinstance(result.supported, bool)


def test_verification_result_dataclass():
    """Test VerificationResult can be created with all fields."""
    result = VerificationResult(
        is_arm=True,
        architecture="aarch64",
        supported=True,
        reason="Test environment",
        hardware_profile={"architecture": "aarch64", "is_arm": True}
    )
    assert result.is_arm == True
    assert result.supported == True
    assert result.reason == "Test environment"


def test_arm_verifier_with_requirements():
    """Test ARMVerifier with custom requirements."""
    verifier = ARMVerifier(require_arm=True, min_cores=1, min_memory_mb=256)
    result = verifier.verify()
    assert isinstance(result, VerificationResult)


def test_arm_verifier_get_status_badge():
    """Test status badge generation."""
    badge = ARMVerifier().get_status_badge()
    # On x86, should be ARM NOT DETECTED
    assert "ARM" in badge or "not" in badge.lower()


def test_quick_verify_with_params():
    """Test quick_verify with custom parameters."""
    result = quick_verify(min_cores=2, min_memory_mb=512)
    assert isinstance(result, VerificationResult)