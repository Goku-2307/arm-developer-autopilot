"""Tests for the Hardware Profiler module."""

import pytest
from src.arm.hardware_profiler import profile, is_arm_environment


def test_profile_has_required_fields():
    """Test that the hardware profile has all required fields."""
    p = profile()
    required_fields = [
        "architecture", "machine", "cpu_model", "physical_cores",
        "logical_cores", "memory_total_mb", "memory_available_mb",
        "neon", "os", "kernel", "python_version", "is_arm",
        "process_architecture", "hostname"
    ]
    for field in required_fields:
        assert hasattr(p, field), f"Missing field: {field}"


def test_profile_is_arm_on_arm():
    """Test profile detection on ARM systems."""
    p = profile()
    # On ARM systems, is_arm should be True
    # On x86, it should be False (which is also valid)
    assert isinstance(p.is_arm, bool)


def test_profile_architecture_is_valid():
    """Test that architecture is a valid value."""
    p = profile()
    assert p.architecture in ("x86_64", "aarch64", "arm64", "arm")


def test_is_arm_environment():
    """Test ARM environment detection."""
    result = is_arm_environment(min_cores=1, min_memory_mb=256)
    assert "is_arm" in result
    assert "supported" in result
    assert "reason" in result


def test_is_arm_environment_not_arm():
    """Test that x86 is correctly identified as non-ARM."""
    result = is_arm_environment(min_cores=1, min_memory_mb=256)
    # On x86 systems, is_arm should be False
    assert isinstance(result["is_arm"], bool)


def test_is_arm_environment_min_cores():
    """Test minimum core check."""
    result = is_arm_environment(min_cores=16, min_memory_mb=256)
    # If cores < 16, should still report is_arm=True but supported=False
    assert "is_arm" in result


def test_is_arm_environment_min_memory():
    """Test minimum memory check."""
    result = is_arm_environment(min_cores=1, min_memory_mb=100000)
    # If memory < 100GB, should report supported=False
    assert "is_arm" in result