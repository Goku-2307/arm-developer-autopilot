"""CLI for ARM Developer AutoPilot.

Provides command-line interface for the optimization agent.
Usage:
    python -m src.cli profile
    python -m src.cli verify
    python -m src.cli optimize <project_path>
    python -m src.cli deploy <session_id>
    python -m src.cli report <session_id>
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import onnxruntime as ort

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.arm.hardware_profiler import profile, is_arm_environment
from src.arm.arm_verifier import quick_verify, ARMVerifier, VerificationResult
from src.onnx.runtime_validator import RuntimeValidator, ONNXModelInfo
from src.analysis.ai_detector import AIDetector
from src.optimization.candidate_generator import CandidateGenerator
from src.optimization.optimizer import ModelOptimizer
from src.optimization.optimization_search import OptimizationSearch
from src.optimization.ranking_engine import ScoringEngine
from src.core.session_manager import SessionManager
from src.services.optimization_service import OptimizationService


def cmd_profile(args):
    """Show hardware profile."""
    p = profile()
    print("Hardware Profile:")
    print(f"  Architecture: {p.architecture}")
    print(f"  Machine: {p.machine}")
    print(f"  CPU Model: {p.cpu_model}")
    print(f"  Physical Cores: {p.physical_cores}")
    print(f"  Logical Cores: {p.logical_cores}")
    print(f"  Memory Total: {p.memory_total_mb} MB")
    print(f"  Memory Available: {p.memory_available_mb} MB")
    print(f"  NEON: {p.neon}")
    print(f"  OS: {p.os}")
    print(f"  Kernel: {p.kernel}")
    print(f"  Python Version: {p.python_version}")
    print(f"  ONNX Runtime: {p.onnxruntime_version}")
    print(f"  NumPy: {p.numpy_version}")
    print(f"  Is ARM: {p.is_arm}")
    print(f"  Process Architecture: {p.process_architecture}")


def cmd_verify(args):
    """Verify ARM environment."""
    result = quick_verify()
    print("ARM Verification:")
    print(f"  is_arm: {result.is_arm}")
    print(f"  supported: {result.supported}")
    print(f"  reason: {result.reason}")
    print(f"  architecture: {result.architecture}")

    # Detailed profile
    p = profile()
    print()
    print("Hardware Details:")
    print(f"  architecture: {p.architecture}")
    print(f"  is_arm: {p.is_arm}")
    print(f"  physical_cores: {p.physical_cores}")
    print(f"  memory_total_mb: {p.memory_total_mb}")


def cmd_optimize(args):
    """Run optimization on a project."""
    if not args.project_path:
        print("Error: project_path is required")
        print("Usage: python -m src.cli optimize <project_path>")
        sys.exit(1)

    project_path = args.project_path

    print(f"🔍 Starting optimization for: {project_path}")
    print()

    # Hardware check
    print("Hardware Analysis:")
    p = profile()
    print(f"    Architecture: {p.architecture}")
    print(f"  Is ARM: {p.is_arm}")
    print()

    # Run optimization
    service = OptimizationService()
    session = service.optimize_project(project_path)

    if session is None or not session.models:
        print("❌ No AI models found in the project.")
        return

    print(f"✓ Detected {len(session.models)} model(s)")
    print(f"  Project: {session.project_name}")
    print(f"  Language: {session.language}")
    print()

    # Baseline benchmark
    print("Baseline Benchmark:")
    from src.optimization.benchmark import Benchmark

    for model in session.models[:1]:  # Just first model
        model_path = model.get("model_path", "")
        if model_path and __import__('pathlib').Path(model_path).exists():
            benchmark = Benchmark(model_path, threads=1)
            baseline_result = benchmark.benchmark()
            print(f"  Model: {model.get('model_name', '?')}")
            print(f"  Latency (mean): {baseline_result.get('latency_mean_ms', '-')} ms")
            print(f"  Throughput: {baseline_result.get('throughput_fps', '-')} FPS")
            print(f"  Memory: {baseline_result.get('memory', '-')} MB")
            print(f"  Model Size: {baseline_result.get('model_size_mb', '-')} MB")
            print()

    # Candidate generation
    print("Candidate Generation:")
    gen = CandidateGenerator()
    candidates = gen.generate()  # Already returns dicts
    candidates_dicts = candidates  # No conversion needed
    print(f"  Generated {len(candidates_dicts)} candidates")
    for c in candidates_dicts[:3]:
        print(f"    {c.get('candidate_id', '?')}: {c.get('quantization', '?')}, {c.get('threads', '?')} threads, {c.get('graph_optimization', '?')} graph opt")
    print()

    # Optimization search
    print("Optimization Search:")
    if session.models:
        model_path = session.models[0].get("model_path", "")
        if model_path and __import__('pathlib').Path(model_path).exists():
            search = OptimizationSearch(model_path)
            search_results = search.execute(candidates_dicts)

            # Display results
            print(f"  Completed {len(search_results)} candidate benchmarks")
            for r in search_results:
                status = r.get("status", "?")
                latency = r.get("latency_mean_ms", "N/A")
                quant = r.get("quantization", "?")
                print(f"    {r.get('candidate_id', '?')}: status={status}, latency={latency}ms, quant={quant}")
            print()

            # Ranking
            print("Candidate Ranking:")
            objective = "balanced"
            ranked = ScoringEngine.rank(search_results, objective=objective)
            for i, r in enumerate(ranked[:3]):
                score = r.get("score", "-")
                latency = r.get("latency_mean_ms", "-")
                quant = r.get("quantization", "?")
                print(f"  {i + 1}. {r.get('candidate_id', '?')}: score={score}, latency={latency}ms, quant={quant}")
            print()

            # Best candidate
            if ranked:
                best = ranked[0]
                print(f"Best Configuration:")
                print(f"  Quantization: {best.get('quantization', '?')}")
                print(f"  Threads: {best.get('threads', '?')}")
                print(f"  Graph Optimization: {best.get('graph_optimization', '?')}")
                print(f"  Execution Mode: {best.get('execution_mode', '?')}")
                print(f"  Latency: {best.get('latency_mean_ms', '-')} ms")
                print(f"  Throughput: {best.get('throughput_fps', '-')} FPS")
                print(f"  Score: {best.get('score', '-')}")
        else:
            print("  ❌ Model not found for optimization search")
    else:
        print("  ❌ No models available for optimization")


def cmd_deploy(args):
    """Deploy optimized model."""
    session_id = args.session_id

    print(f"Deploying model from session: {session_id}")
    print()

    # In a real implementation, this would load a session from storage
    # For now, we'll run verification on the example model
    model_path = "examples/sample_ai_project/mobilenetv2.onnx"

    if not __import__('pathlib').Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Run verification
    from src.deployment.post_deployment_verifier import PostDeploymentVerifier
    verifier = PostDeploymentVerifier(model_path)
    result = verifier.verify()

    print("Post-Deployment Verification:")
    print(f"  Success: {result.success}")
    print(f"  Model Loads: {result.model_loads}")
    print(f"  Inference Works: {result.inference_works}")
    print(f"  Latency: {result.latency_ms} ms")
    print(f"  Throughput: {result.throughput_fps} FPS")
    print(f"  Memory: {result.memory_mb} MB")
    print(f"  Message: {result.message}")

    if result.details:
        print(f"  Details: {json.dumps(result.details, indent=2)}")


def cmd_report(args):
    """Generate report for a session."""
    session_id = args.session_id

    print(f"Generating report for session: {session_id}")
    print()

    from src.services.report_service import ReportService

    # In a real implementation, this would load a session from storage
    # For now, check if we have a session in the dashboard state
    from src.services.optimization_service import OptimizationService

    service = OptimizationService()
    # We can't easily load a session by ID in this CLI format,
    # so let's just show what the report would contain
    print("Report Types Available:")
    print("  - HTML report with full optimization results")
    print("  - PDF report with summary tables")
    print("  - Deployment package (config.json, benchmark.py, deployment.py, README.md)")


def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ARM Developer AutoPilot - CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # profile command
    profile_parser = subparsers.add_parser("profile", help="Show hardware profile")
    profile_parser.set_defaults(func=cmd_profile)

    # verify command
    verify_parser = subparsers.add_parser("verify", help="Verify ARM environment")
    verify_parser.set_defaults(func=cmd_verify)

    # optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Run optimization on project")
    optimize_parser.add_argument("project_path", help="Path to project directory")
    optimize_parser.set_defaults(func=cmd_optimize)

    # deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy optimized model")
    deploy_parser.add_argument("session_id", help="Session ID")
    deploy_parser.set_defaults(func=cmd_deploy)

    # report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    report_parser.add_argument("session_id", help="Session ID")
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()