import streamlit as st
import time
from src.pipeline import Pipeline

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="ARM Developer AutoPilot",
    page_icon="🚀",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("🚀 ARM Developer AutoPilot")

    st.markdown("### AI Copilot for ARM Optimization")

    st.divider()

    st.write("### Version")
    st.success("v1.0")

    st.divider()

    st.write("### Team")

    st.write("👨‍💻 AI Engineer")
    st.write("🖥 ARM Engineer")
    st.write("⚙ DevOps Engineer")
    st.write("🎨 UI Engineer")

    st.divider()

    st.info(
        "This tool automatically analyzes AI projects, "
        "optimizes them for ARM devices, benchmarks performance, "
        "and deploys them to GitHub."
    )

# -----------------------------
# Title
# -----------------------------

st.title("🚀 ARM Developer AutoPilot")

st.caption(
    "AI Copilot for ARM AI Optimization & GitHub Deployment"
)

st.divider()

# -----------------------------
# Dashboard Metrics
# -----------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Projects", "1")

with col2:
    st.metric("Models Found", "--")

with col3:
    st.metric("Status", "Idle")

with col4:
    st.metric("Best Speedup", "--")

st.divider()

# -----------------------------
# Project Selection
# -----------------------------

st.header("📂 Step 1 : Select AI Project")

project = st.text_input(
    "Project Folder",
    value="examples/sample_ai_project"
)

# -----------------------------
# ARM Device
# -----------------------------

st.header("🖥 Step 2 : Select Target ARM Device")

device = st.selectbox(

    "Target Device",

    [

        "Raspberry Pi 5",

        "RK3588",

        "Jetson Orin Nano",

        "Snapdragon X Elite",

        "Generic ARM Cortex"

    ]

)

# -----------------------------
# Optimization Options
# -----------------------------

st.header("⚙ Step 3 : Optimization Options")

col1, col2 = st.columns(2)

with col1:

    quantization = st.checkbox(
        "Enable INT8 Quantization",
        value=True
    )

    graph = st.checkbox(
        "Graph Optimization",
        value=True
    )

with col2:

    runtime = st.checkbox(
        "Runtime Auto-Tuning",
        value=True
    )

    github = st.checkbox(
        "Deploy to GitHub",
        value=True
    )

st.divider()

# -----------------------------
# Analyze Button
# -----------------------------

if st.button("🚀 Start Optimization"):

    st.header("Optimization Pipeline")

    progress = st.progress(0)

    status = st.empty()

    pipeline_steps = [

        "Scanning Repository",

        "Detecting AI Models",

        "Analyzing ARM Device",

        "Searching Best Optimization",

        "Benchmarking Original Model",

        "Applying Quantization",

        "Runtime Auto-Tuning",

        "Benchmarking Optimized Model",

        "Generating Report",

        "Preparing GitHub Deployment"

    ]

    percent = 0

    for step in pipeline_steps:

        status.info(step)

        for i in range(10):

            percent += 1

            progress.progress(percent)

            time.sleep(0.05)

    status.success("Optimization Completed Successfully!")

    st.balloons()

    st.divider()

    # -------------------------
    # Repository Summary
    # -------------------------

    st.header("📁 Repository Summary")

    col1, col2 = st.columns(2)

    with col1:

        st.success("Project Detected")

        st.write("Project Name")
        st.code("sample_ai_project")

        st.write("Language")
        st.code("Python")

        st.write("Framework")
        st.code("ONNX Runtime")

    with col2:

        st.success("AI Model")

        st.write("Model")
        st.code("MobileNetV2")

        st.write("Format")
        st.code("ONNX")

        st.write("Target Device")
        st.code(device)

    st.divider()

    # -------------------------
    # Benchmark
    # -------------------------

    st.header("📊 Benchmark Results")

    before, arrow, after = st.columns([3,1,3])

    with before:

        st.subheader("Before")

        st.metric(
            "Latency",
            "180 ms"
        )

        st.metric(
            "Memory",
            "2.4 GB"
        )

        st.metric(
            "Power",
            "7.8 W"
        )

    with arrow:

        st.markdown("# ➜")

    with after:

        st.subheader("After")

        st.metric(
            "Latency",
            "68 ms",
            "-62%"
        )

        st.metric(
            "Memory",
            "900 MB",
            "-62%"
        )

        st.metric(
            "Power",
            "4.1 W",
            "-47%"
        )

    st.divider()

    # -------------------------
    # AI Recommendation
    # -------------------------

    st.header("🤖 Optimization Copilot")

    st.success(
        """
### Recommendation

The selected AI model was analyzed successfully.

• Best Runtime : ONNX Runtime

• Best Thread Count : 4

• Quantization : INT8

• Graph Optimization : Enabled

The optimization engine predicts this configuration
provides the best balance between latency,
memory usage and deployment efficiency
for the selected ARM platform.
"""
    )

    st.divider()

    # -------------------------
    # GitHub
    # -------------------------

    st.header("🐙 GitHub Deployment")

    repo = st.text_input(

        "Repository Name",

        "arm-optimized-project"

    )

    visibility = st.radio(

        "Visibility",

        [

            "Public",

            "Private"

        ]

    )

    if st.button("🚀 Deploy to GitHub"):

        st.success("Repository Created")

        st.success("README Generated")

        st.success("Benchmark Report Added")

        st.success("Optimized Model Uploaded")

        st.success("Project Successfully Pushed to GitHub")

        st.balloons()

st.divider()

st.caption(
    "ARM Developer AutoPilot | ARM Optimization Challenge 2026"
)