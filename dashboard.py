import os
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.arm.hardware_profiler import profile, is_arm_environment
from src.arm.arm_verifier import ARMVerifier, quick_verify, VerificationResult
from src.onnx.runtime_validator import RuntimeValidator, ONNXModelInfo
from src.analysis.ai_detector import AIDetector
from src.optimization.candidate_generator import CandidateGenerator, Candidate
from src.optimization.optimizer import ModelOptimizer
from src.optimization.benchmark import Benchmark
from src.optimization.ranking_engine import ScoringEngine
from src.optimization.comparator import BenchmarkComparator
from src.optimization.explanation_engine import ExplanationEngine
from src.deployment.deployment_package import generate_deployment_package
from src.deployment.post_deployment_verifier import PostDeploymentVerifier, VerificationResult as VResult
from src.services.optimization_service import OptimizationService
from src.services.report_service import ReportService
from src.services.github_service import GitHubService


st.set_page_config(
    page_title="ARM Developer AutoPilot",
    page_icon="🚀",
    layout="wide",
)


# --------------------------------------------------
# Session state: survives every Streamlit rerun
# --------------------------------------------------
if "session" not in st.session_state:
    st.session_state.session = None
if "reports" not in st.session_state:
    st.session_state.reports = None
if "publish_result" not in st.session_state:
    st.session_state.publish_result = None
if "optimization_objective" not in st.session_state:
    st.session_state.optimization_objective = "balanced"


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🚀 ARM Developer AutoPilot")
st.sidebar.caption("ARM CPU AI Optimization Platform")
st.sidebar.markdown("---")

project_path = st.sidebar.text_input(
    "Project to Analyze",
    value="examples/sample_ai_project",
)

if st.sidebar.button("🚀 Start Optimization", use_container_width=True, type="primary"):
    st.session_state.reports = None
    st.session_state.publish_result = None
    st.session_state.optimization_objective = "balanced"
    try:
        with st.spinner("Analyzing, benchmarking and optimizing..."):
            st.session_state.session = OptimizationService().optimize_project(project_path)
        st.sidebar.success("Optimization complete")
    except Exception as exc:
        st.session_state.session = None
        st.sidebar.error(f"Optimization failed: {exc}")

st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline")
for item in [
    "Hardware Analysis",
    "ARM Verification",
    "Project Analysis",
    "AI Model Detection",
    "Baseline Benchmark",
    "Candidate Generation",
    "Optimization",
    "Candidate Ranking",
    "Report Generation",
    "GitHub Publishing",
]:
    st.sidebar.write(f"✅ {item}")


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🚀 ARM Developer AutoPilot")
st.caption("Intelligent AI Model Optimization for ARM CPUs")
st.divider()

session = st.session_state.session

if session is None:
    st.info("Choose a project in the sidebar and click **Start Optimization**.")
    st.stop()


# --------------------------------------------------
# Section 1 — Hardware
# --------------------------------------------------
st.subheader("🖥️ Hardware")

# Hardware profiling
hw_profile = profile()
hw_verification = is_arm_environment(min_cores=1, min_memory_mb=256)

# Display ARM status badge
if not hw_verification["is_arm"]:
    st.error("🔴 ARM NOT DETECTED")
    st.caption("ARM64/aarch64 hardware is required for real ARM benchmarking")
elif hw_verification["supported"]:
    st.success("🟢 ARM VERIFIED")
else:
    st.warning("🟡 ARM ENVIRONMENT INCOMPLETE")

# Hardware details
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Architecture", hw_profile.architecture)
    st.metric("CPU Model", hw_profile.cpu_model)
with col2:
    st.metric("Cores", f"{hw_profile.physical_cores} physical / {hw_profile.logical_cores} logical")
    st.metric("RAM Total", f"{hw_profile.memory_total_mb} MB")
with col3:
    st.metric("RAM Available", f"{hw_profile.memory_available_mb} MB")
    st.metric("NEON", "Yes" if hw_profile.neon else "No")

# ONNX Runtime info
st.caption(f"ONNX Runtime: {hw_profile.onnxruntime_version}")
st.caption(f"NumPy: {hw_profile.numpy_version}")

# --------------------------------------------------
# Section 2 — Project
# --------------------------------------------------
st.subheader("📁 Project")

best = session.best_result or {}
col1, col2 = st.columns(2)
with col1:
    st.info(f"**Project:** {session.project_name or '-'}")
    st.info(f"**Language:** {session.language or '-'}")
with col2:
    st.info(f"**Models detected:** {len(session.models or [])}")
    if session.models:
        for model in session.models[:3]:
            st.caption(f"• {model.get('model_name', '-')} ({model.get('format', '-')})")

if session.models and len(session.models) > 3:
    st.caption(f"... and {len(session.models) - 3} more models")


# --------------------------------------------------
# Section 3 — Baseline
# --------------------------------------------------
st.subheader("📊 Baseline Benchmark")

if session.benchmark_results and len(session.benchmark_results) > 0:
    baseline = session.benchmark_results[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Latency (mean)", f"{baseline.get('latency_mean_ms', '-')} ms")
        st.metric("P95 Latency", f"{baseline.get('latency_p95_ms', '-')} ms")
    with col2:
        st.metric("Throughput", f"{baseline.get('throughput_fps', '-')} FPS")
        st.metric("Memory", f"{baseline.get('memory', '-')} MB")
    with col3:
        st.metric("Model Size", f"{baseline.get('model_size_mb', '-')} MB")
        st.metric("Threads", f"{baseline.get('threads', '-')}")
    st.caption(f"Warmup: {baseline.get('warmup_iterations', '-')} iters | Benchmark: {baseline.get('benchmark_iterations', '-')} iters")
else:
    st.info("Run baseline benchmark to display metrics here.")


# --------------------------------------------------
# Section 4 — Optimization Objective
# --------------------------------------------------
st.subheader("🎯 Optimization Objective")

objective = st.radio(
    "Select optimization objective:",
    options=["balanced", "lowest_latency", "lowest_memory", "smallest_model", "highest_throughput"],
    format_func=lambda x: x.replace("_", " ").title(),
    index=["balanced", "lowest_latency", "lowest_memory", "smallest_model", "highest_throughput"].index(
        st.session_state.optimization_objective
    ) if st.session_state.optimization_objective in ["balanced", "lowest_latency", "lowest_memory", "smallest_model", "highest_throughput"] else 0,
    key="objective_selector",
)

st.session_state.optimization_objective = objective

# Display objective info
objective_info = ScoringEngine.get_objective_info(objective)
st.caption(objective_info["description"])


# --------------------------------------------------
# Section 5 — Agent Progress
# --------------------------------------------------
st.subheader("📜 Optimization Timeline")

# Build timeline from session events
events = getattr(session, "events", []) or []

# Add default stages if no events
if not events:
    stages = [
        ("Hardware Analysis", "SUCCESS" if hw_verification["is_arm"] else "FAILED"),
        ("ARM Verification", "SUCCESS" if hw_verification["supported"] else "FAILED"),
        ("Project Analysis", "SUCCESS" if session.project_name else "FAILED"),
        ("AI Model Detection", "SUCCESS" if session.models else "FAILED"),
        ("Baseline Benchmark", "pending"),
        ("Candidate Generation", "pending"),
        ("Optimization", "pending"),
        ("Candidate Benchmarking", "pending"),
        ("Ranking", "pending"),
        ("Best Configuration", "pending"),
        ("Deployment", "pending"),
        ("Verification", "pending"),
    ]
    for i, (stage, status) in enumerate(stages):
        time_str = f"{(i * 10):02d}:00:01"
        events.append({
            "time": time_str,
            "status": status,
            "stage": stage,
            "message": f"{stage} {'completed' if status == 'SUCCESS' else 'in progress'}",
        })
else:
    # Update progress for completed events
    for event in events:
        if event.get("status") in ["pending", "running"]:
            event["status"] = "running"

progress_items = events[:12]  # Show max 12 stages
for event in progress_items:
    status_emoji = {
        "SUCCESS": "✅",
        "FAILED": "❌",
        "running": "⏳",
        "pending": "○",
    }.get(event.get("status", "pending"), "○")
    st.caption(f"{event.get('time', '')} {event.get('status', '?')} | {event.get('stage', '?')}")


# --------------------------------------------------
# Section 6 — Candidate Table
# --------------------------------------------------
st.subheader("📋 Optimization Candidates")

if session.benchmark_results and len(session.benchmark_results) > 0:
    # Build candidate data table
    results = session.benchmark_results
    candidates_data = []

    for r in results:
        cand = {
            "Candidate": r.get("candidate_id", "?"),
            "Quantization": r.get("quantization", "?"),
            "Threads": r.get("threads", "?"),
            "Graph Optimization": r.get("graph_optimization", "?"),
            "Execution Mode": r.get("execution_mode", "?"),
            "Latency (ms)": r.get("latency_mean_ms", "-"),
            "P95": r.get("latency_p95_ms", "-"),
            "Memory (MB)": r.get("memory", "-"),
            "Model Size (MB)": r.get("model_size_mb", "-"),
            "Throughput (FPS)": r.get("throughput_fps", "-"),
            "Score": r.get("score", "-"),
            "Status": r.get("status", "?"),
        }
        candidates_data.append(cand)

    df = pd.DataFrame(candidates_data)

    # Highlight best candidate
    if "Score" in df.columns:
        # Convert score to numeric for sorting, handling "-" values
        df_numeric = df.copy()
        df_numeric["Score"] = pd.to_numeric(df["Score"], errors="coerce")

        # Find best index based on current objective
        objective = st.session_state.optimization_objective
        best_idx = df_numeric["Score"].idxmax()

        st.dataframe(
            df.style.apply(
                lambda x: [
                    "#b7f7c2" if i == best_idx else "" for i in range(len(df))
                ],
                subset=["Candidate"],
            ),
            use_container_width=True,
        )
    else:
        st.dataframe(df, use_container_width=True)

    # Summary stats
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Total candidates: {len(df)}")
    with col2:
        st.caption(f"Best score: {df['Score'].max() if not df['Score'].isna().all() else 'N/A'}")
else:
    st.info("Run optimization to display candidate results here.")


# --------------------------------------------------
# Section 7 — Best Configuration
# --------------------------------------------------
st.subheader("🏆 Best Configuration")

if best:
    objective = st.session_state.optimization_objective
    quant = best.get("quantization", "?")
    threads = best.get("threads", "?")
    graph = best.get("graph_optimization", "?")
    execution = best.get("execution_mode", "?")

    st.success("Recommended ARM Configuration")
    st.write(f"**Quantization:** {quant}")
    st.write(f"**Threads:** {threads}")
    st.write(f"**Graph Optimization:** {graph}")
    st.write(f"**Execution Mode:** {execution}")

    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        latency = best.get("latency")
        st.metric("Latency", f"{latency} ms" if latency else "-")
    with col2:
        memory = best.get("memory")
        st.metric("Memory", f"{memory} MB" if memory else "-")
    with col3:
        model_size = best.get("model_size")
        st.metric("Model Size", f"{model_size} MB" if model_size else "-")

    score = best.get("score", "-")
    st.metric("Score", score)

    # Baseline vs optimized comparison
    st.caption("**Baseline vs Optimized:**")
    if session.benchmark_results and len(session.benchmark_results) > 0:
        baseline = session.benchmark_results[0]
        comp = BenchmarkComparator.compare(
            baseline,
            {
                "latency_mean_ms": best.get("latency"),
                "memory": best.get("memory"),
                "model_size_mb": best.get("model_size"),
                "throughput_fps": best.get("score"),
            }
        )

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"Latency: {comp.get('latency_before_ms', '-')} → {comp.get('latency_after_ms', '-')} ms")
            st.caption(f"Improvement: {comp.get('latency_improvement_percent', '-')}%")
        with c2:
            st.write(f"Model Size: {comp.get('model_size_before_mb', '-')} → {comp.get('model_size_after_mb', '-')} MB")
            st.caption(f"Reduction: {comp.get('model_size_reduction_percent', '-')}%")
else:
    st.info("No optimization results yet.")


# --------------------------------------------------
# Section 8 — Improvement Panel
# --------------------------------------------------
st.subheader("📈 Improvement Panel")

if session.benchmark_results and len(session.benchmark_results) > 0 and best:
    baseline = session.benchmark_results[0]

    # Calculate improvements
    baseline_lat = baseline.get("latency_mean_ms", 0)
    opt_lat = best.get("latency")

    c1, c2, c3 = st.columns(3)

    with c1:
        if baseline_lat > 0 and opt_lat and opt_lat > 0:
            imp = round(((baseline_lat - opt_lat) / baseline_lat) * 100, 2)
            st.metric("Latency", f"↓ {imp}%", f"{opt_lat} ms vs {baseline_lat} ms")
        else:
            st.metric("Latency", "-", "-")

    with c2:
        baseline_mem = baseline.get("memory", 0)
        opt_mem = best.get("memory")
        if baseline_mem > 0 and opt_mem and opt_mem > 0:
            imp = round(((baseline_mem - opt_mem) / baseline_mem) * 100, 2)
            st.metric("Memory", f"↓ {imp}%", f"{opt_mem} MB vs {baseline_mem} MB")
        else:
            st.metric("Memory", "-", "-")

    with c3:
        baseline_size = baseline.get("model_size_mb", 0)
        opt_size = best.get("model_size_mb")
        if baseline_size > 0 and opt_size and opt_size > 0:
            imp = round(((baseline_size - opt_size) / baseline_size) * 100, 2)
            st.metric("Model Size", f"↓ {imp}%", f"{opt_size} MB vs {baseline_size} MB")
        else:
            st.metric("Model Size", "-", "-")

    # Throughput
    baseline_tp = baseline.get("throughput_fps", 0)
    opt_tp = best.get("score")  # score can represent throughput
    if baseline_tp > 0 and opt_tp and opt_tp > 0:
        imp = round(((opt_tp - baseline_tp) / baseline_tp) * 100, 2)
        st.metric("Throughput", f"↑ {imp}%", f"{opt_tp} FPS vs {baseline_tp} FPS")
    elif opt_tp:
        st.metric("Throughput", f"{opt_tp} FPS")
    else:
        st.metric("Throughput", "-", "-")

# --------------------------------------------------
# Section 9 — Why This Configuration
# --------------------------------------------------
st.subheader("💡 Why This Configuration?")

if best and session.benchmark_results and len(session.benchmark_results) > 0:
    objective = st.session_state.optimization_objective

    # Prepare data for explanation
    baseline_data = session.benchmark_results[0]
    best_candidate_id = best.get("candidate_id", "?")

    # Find the best candidate details
    best_candidate_data = None
    for r in session.benchmark_results:
        if r.get("candidate_id") == best_candidate_id:
            best_candidate_data = r
            break

    if best_candidate_data:
        explanation = ExplanationEngine.generate(
            best_candidate=best_candidate_data,
            baseline=baseline_data,
            optimized=best,
            candidates=session.benchmark_results,
            objective=objective,
        )
        st.markdown(explanation)
    else:
        st.caption(f"Configuration: {best.get('quantization', '?')} quantization, "
                   f"{best.get('threads', '?')} threads, "
                   f"{best.get('graph_optimization', '?')} graph optimization")
        st.caption(f"This configuration achieved the highest score under "
                   f"the {objective} optimization objective")
else:
    st.info("Run optimization to display the explanation here.")


# --------------------------------------------------
# Section 10 — Reports and Deployment
# --------------------------------------------------
st.divider()
st.subheader("📄 Reports & Deployment")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Generate HTML + PDF", use_container_width=True):
        try:
            with st.spinner("Generating reports..."):
                st.session_state.reports = ReportService().generate(session)
            st.success("Reports generated.")
        except Exception as exc:
            st.error(f"Report generation failed: {exc}")

reports = st.session_state.reports
if reports:
    with open(reports["html"], "rb") as f:
        st.download_button(
            "⬇️ Download HTML",
            data=f.read(),
            file_name="optimization_report.html",
            mime="text/html",
            use_container_width=True,
        )
    with open(reports["pdf"], "rb") as f:
        st.download_button(
            "⬇️ Download PDF",
            data=f.read(),
            file_name="optimization_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

with col2:
    if st.button("Generate Deployment Package", use_container_width=True):
        try:
            with st.spinner("Generating deployment package..."):
                paths = generate_deployment_package(session)
            st.success("Deployment package generated.")
            st.caption(f"Files: {', '.join(paths.values())}")
        except Exception as exc:
            st.error(f"Deployment package generation failed: {exc}")

with col3:
    if st.button("Run Verification", use_container_width=True):
        try:
            with st.spinner("Running verification..."):
                # Get the optimized model path from best result
                best_result = best or {}
                opt_model_path = best_result.get("optimized_model", "")

                if opt_model_path and __import__('pathlib').Path(opt_model_path).exists():
                    verifier = PostDeploymentVerifier(opt_model_path)
                    verification_result = verifier.verify()

                    # Compare with baseline
                    if session.benchmark_results and len(session.benchmark_results) > 0:
                        baseline = session.benchmark_results[0]
                        comparison = PostDeploymentVerifier.compare_with_baseline(
                            verification_result, baseline
                        )

                        st.success(verification_result.message)
                        st.caption(f"Latency: {comparison.get('latency_improvement_percent', '-')}% improvement")
                        st.caption(f"Throughput: {comparison.get('throughput_improvement_percent', '-')}% improvement")
                        st.caption(f"Verification latency: {verification_result.latency_ms} ms")
                        st.caption(f"Verification throughput: {verification_result.throughput_fps} FPS")
                        st.caption(f"Memory: {verification_result.memory_mb} MB")
                else:
                    # Try to optimize the model first
                    with st.spinner("Optimizing model for verification..."):
                        opt = ModelOptimizer(
                            str(
                                (
                                    session.models[0].get("model_path")
                                    if session.models
                                    else "examples/sample_ai_project/mobilenetv2.onnx"
                            )
                        )
                        )
                        int8_result = opt.optimize("INT8")
                        opt_model_path = int8_result["optimized_model"] if int8_result else ""

                    if opt_model_path and __import__('pathlib').Path(opt_model_path).exists():
                        verifier = PostDeploymentVerifier(opt_model_path)
                        verification_result = verifier.verify()

                        if session.benchmark_results and len(session.benchmark_results) > 0:
                            baseline = session.benchmark_results[0]
                            comparison = PostDeploymentVerifier.compare_with_baseline(
                                verification_result, baseline
                            )

                            st.success(verification_result.message)
                            st.caption(f"Latency: {comparison.get('latency_improvement_percent', '-')}% improvement")
                            st.caption(f"Throughput: {comparison.get('throughput_improvement_percent', '-')}% improvement")
                    else:
                        st.error("Could not generate optimized model for verification")

        except Exception as exc:
            st.error(f"Verification failed: {exc}")


# GitHub publishing (existing functionality)
st.divider()
st.subheader("🐙 GitHub Deployment")

g1, g2 = st.columns(2)
with g1:
    github_owner = st.text_input("GitHub Username", value=os.getenv("GITHUB_USERNAME", ""))
    github_repo = st.text_input(
        "Repository Name",
        value="arm-developer-autopilot",
    )
with g2:
    github_token = st.text_input(
        "GitHub Personal Access Token",
        type="password",
        value=os.getenv("GITHUB_TOKEN", ""),
        help="Use a token with permission to create/push to repositories.",
    )
    github_private = st.checkbox("Create repository as private", value=False)

if st.button("🚀 Publish Project to GitHub", use_container_width=True, type="primary"):
    if not github_owner.strip():
        st.error("Enter your GitHub username.")
    elif not github_repo.strip():
        st.error("Enter a repository name.")
    elif not github_token.strip():
        st.error("Enter a GitHub Personal Access Token.")
    else:
        try:
            with st.spinner("Creating/configuring repository and pushing project..."):
                result = GitHubService().publish(
                    session=session,
                    token=github_token,
                    owner=github_owner.strip(),
                    repo_name=github_repo.strip(),
                    private=github_private,
                    project_path=str(Path.cwd()),
                )
            st.session_state.publish_result = result
        except Exception as exc:
            st.session_state.publish_result = {
                "success": False,
                "message": f"Publishing failed: {exc}",
            }

result = st.session_state.publish_result
if result:
    if result.get("success"):
        st.success(result.get("message", "Published successfully."))
        if result.get("url"):
            st.link_button("Open GitHub Repository", result["url"])
    else:
        st.error(result.get("message", "Publishing failed."))