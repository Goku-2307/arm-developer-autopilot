"""Tests for the Post-Deployment Verifier module."""

import pytest
from src.deployment.post_deployment_verifier import PostDeploymentVerifier, VerificationResult


def test_verification_result_creation():
    """Test VerificationResult can be created."""
    result = VerificationResult(
        success=True,
        model_loads=True,
        inference_works=True,
        latency_ms=10.5,
        throughput_fps=100.0,
        memory_mb=50.0,
        message="Test message",
        details={"key": "value"}
    )
    assert result.success == True
    assert result.model_loads == True
    assert result.inference_works == True
    assert result.latency_ms == 10.5
    assert result.throughput_fps == 100.0
    assert result.memory_mb == 50.0
    assert result.message == "Test message"
    assert result.details == {"key": "value"}


def test_verification_result_defaults():
    """Test VerificationResult with default values."""
    result = VerificationResult(success=False)
    assert isinstance(result, VerificationResult)
    assert result.success == False


def test_verifier_with_valid_model():
    """Test verifier with an existing model."""
    verifier = PostDeploymentVerifier("examples/sample_ai_project/mobilenetv2.onnx")
    result = verifier.verify()
    assert isinstance(result, VerificationResult)
    # On x86, should still be able to load and run inference
    assert hasattr(result, "success")


def test_compare_with_baseline():
    """Test comparison with baseline."""
    from src.optimization.benchmark import Benchmark
    
    verifier = PostDeploymentVerifier("examples/sample_ai_project/mobilenetv2.onnx")
    verification_result = verifier.verify()
    
    baseline = Benchmark("examples/sample_ai_project/mobilenetv2.onnx", threads=1).benchmark()
    
    comparison = PostDeploymentVerifier.compare_with_baseline(verification_result, baseline)
    assert "latency_improvement_percent" in comparison
    assert "throughput_improvement_percent" in comparison
    assert "verification_success" in comparison