import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

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
    "Project Analysis", "AI Model Detection", "Benchmarking",
    "FP32 / INT8 Optimization", "CPU Thread Tuning", "Candidate Ranking",
    "Report Generation", "GitHub Publishing",
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
# KPI cards
# --------------------------------------------------
best = session.best_result or {}
cols = st.columns(5)
cols[0].metric("Project", session.project_name or "-")
cols[1].metric("Language", session.language or "-")
cols[2].metric("Models", len(session.models or []))
cols[3].metric("Best Threads", best.get("threads", "-"))
cols[4].metric("Quantization", best.get("quantization", "-"))

st.divider()

# --------------------------------------------------
# Best result
# --------------------------------------------------
st.subheader("🏆 Best Configuration")
left, right = st.columns(2)
with left:
    st.success("Recommended ARM Configuration")
    st.write(f"**Quantization:** {best.get('quantization', '-')}")
    st.write(f"**Threads:** {best.get('threads', '-')}")
    st.write(f"**Graph Optimization:** {best.get('graph_optimization', '-')}")
    st.write(f"**Execution Mode:** {best.get('execution_mode', '-')}")
with right:
    st.info("Performance")
    st.write(f"**Latency:** {best.get('latency', '-')} ms")
    st.write(f"**Memory:** {best.get('memory', '-')} MB")
    st.write(f"**Model Size:** {best.get('model_size', '-')} MB")
    st.write(f"**Score:** {best.get('score', '-')}")

st.divider()

# --------------------------------------------------
# Models
# --------------------------------------------------
st.subheader("🤖 Detected AI Models")
for model in session.models or []:
    st.info(
        f"**{model.get('model_name', '-')}**  |  "
        f"Framework: {model.get('framework', '-')}  |  "
        f"Type: {model.get('model_type', '-')}"
    )

# --------------------------------------------------
# Candidate table and charts
# --------------------------------------------------
df = pd.DataFrame(session.benchmark_results or [])
if not df.empty:
    st.subheader("📋 Optimization Candidates")
    if "score" in df.columns:
        st.dataframe(
            df.style.highlight_max(subset=["score"], color="#b7f7c2"),
            use_container_width=True,
        )
    else:
        st.dataframe(df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📈 Latency")
        fig = px.bar(
            df, x="candidate_id", y="latency", color="quantization",
            hover_data=[c for c in ["threads", "score"] if c in df.columns],
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("💾 Memory")
        fig2 = px.bar(
            df, x="candidate_id", y="memory", color="quantization",
        )
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# Timeline
# --------------------------------------------------
st.subheader("📜 Optimization Timeline")
for event in session.events or []:
    st.write(
        f"🕒 **{event.get('time', '')}** | "
        f"**{event.get('status', '')}** | "
        f"**{event.get('stage', '')}** | "
        f"{event.get('message', '')}"
    )

st.divider()

# --------------------------------------------------
# Reports
# --------------------------------------------------
st.subheader("📄 Reports")
rc1, rc2, rc3 = st.columns(3)

with rc1:
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
        rc2.download_button(
            "⬇️ Download HTML",
            data=f.read(),
            file_name="optimization_report.html",
            mime="text/html",
            use_container_width=True,
        )
    with open(reports["pdf"], "rb") as f:
        rc3.download_button(
            "⬇️ Download PDF",
            data=f.read(),
            file_name="optimization_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# --------------------------------------------------
# GitHub publishing
# --------------------------------------------------
st.divider()
st.subheader("🐙 GitHub Deployment")
st.caption("The token is used only for this operation and is not saved to the repository or git config.")

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
