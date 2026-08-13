#!/usr/bin/env python3
"""Deployment script for ARM-optimized AI model.

This script loads and runs the optimized model on the ARM CPU.
It validates the deployment and runs a verification benchmark.

Usage:
    python deployment.py

Generated automatically by ARM Developer AutoPilot.
"""

import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import psutil

# Model path - set this to the optimized model path
MODEL_PATH = "deployment/model/optimized_model.onnx"

def main():
    """Run deployment verification."""
    print("ARM Developer AutoPilot")
    print("-----------------------")

    # Load model
    print("Loading model...")
    try:
        session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        print("OK Model loaded")
    except Exception as e:
        print(f"X Failed to load model: {e}")
        sys.exit(1)

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
    print("Running inference...")
    try:
        results = session.run(None, {input_info.name: input_data})
        print("OK Inference successful")
        print(f"Output shape: {results[0].shape}")
    except Exception as e:
        print(f"X Inference failed: {e}")
        sys.exit(1)

    # Verify memory
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Memory usage: {memory_mb:.2f} MB")

    print()
    print("DEPLOYMENT VERIFIED")

if __name__ == "__main__":
    main()
