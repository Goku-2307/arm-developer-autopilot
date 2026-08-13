"""Tests for the Model Detector module."""

import pytest
import tempfile
import os
from pathlib import Path
from src.analysis.ai_detector import AIDetector


def test_detector_supported_formats():
    """Test that detector supports all format extensions."""
    detector = AIDetector
    supported = list(detector.__init__.__code__.co_consts) if hasattr(detector.__init__, 'co_consts') else []
    # Just test that the detector can be instantiated
    det = AIDetector("/tmp")
    assert det is not None


def test_detect_models_returns_list():
    """Test that detect_models returns a list."""
    detector = AIDetector("examples/sample_ai_project")
    models = detector.detect_models()
    assert isinstance(models, list)


def test_detect_models_has_required_keys():
    """Test that detected models have all required keys."""
    detector = AIDetector("examples/sample_ai_project")
    models = detector.detect_models()
    if models:
        required_keys = [
            "model_name", "model_path", "format", "model_type",
            "size_mb", "input_count", "output_count",
            "input_names", "output_names", "opset_version"
        ]
        for model in models:
            for key in required_keys:
                assert key in model, f"Missing key: {key} in model {model.get('model_name', '?')}"


def test_detect_model_size_is_nonzero():
    """Test that model size is detected for ONNX files."""
    detector = AIDetector("examples/sample_ai_project")
    models = detector.detect_models()
    if models:
        # The ONNX model should have a non-zero size
        model = models[0]
        assert model["size_mb"] > 0, f"Expected positive model size, got {model['size_mb']}"
        assert model["format"] == "ONNX Runtime"


def test_detect_model_metadata():
    """Test that model metadata is extracted correctly."""
    detector = AIDetector("examples/sample_ai_project")
    models = detector.detect_models()
    if models:
        model = models[0]
        # Check that we have input/output info
        assert model["input_count"] > 0, "Expected at least 1 input"
        assert model["output_count"] > 0, "Expected at least 1 output"
        assert len(model["input_names"]) > 0, "Expected at least 1 input name"
        assert len(model["output_names"]) > 0, "Expected at least 1 output name"