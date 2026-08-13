import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def generate_deployment_package(
    session,
    output_dir: str = "deployment",
    hardware_profile: Dict[str, Any] = None,
) -> Dict[str, str]:
    """
    Generate a complete deployment package after optimization.

    Creates the following structure under output_dir:
    ├── config.json          # Device and optimization configuration
    ├── benchmark.json       # Baseline and optimized benchmark results
    ├── deployment.py      # Runtime deployment script
    └── README.md           # Deployment documentation

    Args:
        session: The optimization session object containing results.
        output_dir: Directory name for the deployment package.
        hardware_profile: Dict with hardware profile from HardwareProfiler.

    Returns:
        Dict with paths to generated files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get best result from session
    best_result = getattr(session, "best_result", {}) or {}

    # Get session data
    project_name = getattr(session, "project_name", "ARM Optimization Project") or "ARM Optimization Project"
    language = getattr(session, "language", "-") or "-"
    models = getattr(session, "models", []) or []

    # Hardware profile
    if hardware_profile is None:
        from src.arm.hardware_profiler import profile
        from dataclasses import asdict
        hardware_profile = asdict(profile())

    # Generate config.json
    config = {
        "device": {
            "architecture": hardware_profile.get("architecture", "unknown"),
            "cpu": hardware_profile.get("cpu_model", "Unknown"),
            "cores": hardware_profile.get("physical_cores", 0),
            "threads": best_result.get("threads", 1),
            "ram_total_mb": hardware_profile.get("memory_total_mb", 0),
            "is_arm": hardware_profile.get("is_arm", False),
        },
        "optimization": {
            "quantization": best_result.get("quantization", "FP32"),
            "graph_optimization": best_result.get("graph_optimization", "BASIC"),
            "execution_mode": best_result.get("execution_mode", "PARALLEL"),
        },
        "benchmark": {
            "warmup_iterations": 10,
            "benchmark_iterations": 50,
        },
    }

    config_path = output_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Generate benchmark.json with baseline and optimized results
    baseline = getattr(session, "benchmark_results", [])[0] if getattr(session, "benchmark_results", []) else {}
    optimized_result = {
        "latency_ms": best_result.get("latency"),
        "memory_mb": best_result.get("memory"),
        "model_size_mb": best_result.get("model_size"),
        "throughput_fps": best_result.get("score"),
    }

    # Try to get baseline from the first benchmark result
    if baseline and isinstance(baseline, dict):
        benchmark_data = {
            "baseline": {
                "latency_ms": baseline.get("latency_mean_ms"),
                "memory_mb": baseline.get("memory"),
                "model_size_mb": baseline.get("model_size_mb"),
                "throughput_fps": baseline.get("throughput_fps"),
                "threads": baseline.get("threads"),
                "warmup_iterations": baseline.get("warmup_iterations"),
                "benchmark_iterations": baseline.get("benchmark_iterations"),
            },
            "optimized": optimized_result,
            "improvements": {},
        }

        # Calculate improvements
        if "latency_improvement_percent" in baseline:
            benchmark_data["improvements"]["latency_percent"] = baseline[
                "latency_improvement_percent"
            ]
        if "memory_improvement_percent" in baseline:
            benchmark_data["improvements"]["memory_percent"] = baseline[
                "memory_improvement_percent"
            ]
        if "model_size_reduction_percent" in baseline:
            benchmark_data["improvements"]["model_size_percent"] = baseline[
                "model_size_reduction_percent"
            ]
        if "throughput_improvement_percent" in baseline:
            benchmark_data["improvements"]["throughput_percent"] = baseline[
                "throughput_improvement_percent"
            ]
    else:
        benchmark_data = {
            "baseline": None,
            "optimized": optimized_result,
            "improvements": {},
        }

    benchmark_path = output_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_data, indent=2), encoding="utf-8")

    # Generate deployment.py
    # Create the model directory
    model_dir = output_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Write a placeholder optimized model note (in real usage, this would be the actual ONNX model)
    # For now, just create the directory structure

    # Generate deployment.py using string replacement to avoid format issues
    deployment_code = """#!/usr/bin/env python3
\"\"\"Deployment script for ARM-optimized AI model.

This script loads and runs the optimized model on the ARM CPU.
It validates the deployment and runs a verification benchmark.

Usage:
    python deployment.py

Generated automatically by ARM Developer AutoPilot.
\"\"\"

import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import psutil

# Model path - set this to the optimized model path
MODEL_PATH = "{model_path}"

def main():
    \"\"\"Run deployment verification.\"\"\"
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
"""

    # Write deployment.py with model path replacement
    model_path_str = str(model_dir / "optimized_model.onnx")
    deployment_code = deployment_code.replace("{model_path}", model_path_str)

    deploy_path = output_path / "deployment.py"
    deploy_path.write_text(deployment_code, encoding="utf-8")

    # Generate README.md
    hardware = hardware_profile
    readme_lines = [
        "# Deployment Package",
        "",
        "Generated by ARM Developer AutoPilot",
        f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        "",
        "## Project",
        f"- Project: {project_name}",
        f"- Language: {language}",
        f"- Models detected: {len(models)}",
        "",
        "## Hardware",
        f"- Architecture: {hardware.get('architecture', 'unknown')}",
        f"- CPU: {hardware.get('cpu_model', 'Unknown')}",
        f"- Cores: {hardware.get('physical_cores', 0)}",
        f"- RAM: {hardware.get('memory_total_mb', 0)} MB",
        f"- NEON: {hardware.get('neon', False)}",
        "",
        "## Optimization",
        f"- Quantization: {best_result.get('quantization', 'FP32')}",
        f"- Threads: {best_result.get('threads', 1)}",
        f"- Graph Optimization: {best_result.get('graph_optimization', 'BASIC')}",
        f"- Execution Mode: {best_result.get('execution_mode', 'PARALLEL')}",
        "",
        "## Results",
        f"- Latency: {best_result.get('latency', '-')} ms",
        f"- Memory: {best_result.get('memory', '-')} MB",
        f"- Model Size: {best_result.get('model_size', '-')} MB",
        f"- Score: {best_result.get('score', '-')}",
        "",
        "## Files",
        f"- {output_dir}/config.json - Device and optimization configuration",
        f"- {output_dir}/benchmark.json - Baseline and optimized results",
        f"- {output_dir}/deployment.py - Deployment runner script",
        f"- {output_dir}/model/optimized_model.onnx - Optimized model",
        "",
        "---",
        "Generated by **ARM Developer AutoPilot**.",
    ]

    readme_path = output_path / "README.md"
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    return {
        "config": str(config_path),
        "benchmark": str(benchmark_path),
        "deployment": str(deploy_path),
        "readme": str(readme_path),
    }
