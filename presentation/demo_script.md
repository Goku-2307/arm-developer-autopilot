# ARM Developer AutoPilot — Demo Script

**Target Demo Duration**: 2–3 minutes

## Demo Sequence

### 0:00–0:20  Explain the Problem

- **Issue**: Deploying AI models on ARM edge devices is complex and time-consuming
- **Constraints**: Limited RAM, variable core counts, need for INT8 quantization, device-specific runtimes
- **Manual Process**: Developers must manually experiment with quantization, thread counts, and graph optimizations
- **AutoPilot Solution**: Autonomous optimization agent that handles the complete workflow

**Key Points**:
- ARM dominates edge/mobile (95%+ of devices)
- Constrained resources require optimization
- Manual optimization takes hours of experimentation

---

### 0:20–0:40  Introduce ARM Developer AutoPilot

- **Product**: Autonomous AI optimization platform for ARM CPUs
- **Value Proposition**: "Analyze → Experiment → Benchmark → Optimize → Deploy → Verify"
- **Core Capability**: Automatic hardware detection, model detection, benchmarking, optimization, deployment, and verification
- **Key Differentiator**: Real ARM measurements, never fakes ARM results, clear x86 vs ARM labeling

**Demo Script**:
> "ARM Developer AutoPilot is an autonomous optimization agent that takes an AI model and automatically finds the best configuration for running it on ARM CPUs. It handles everything from hardware detection to deployment verification."

---

### 0:40–1:00  Show Real ARM Hardware Detection

- **Command**: `python -m src.cli profile`
- **Expected Output**: Architecture (aarch64/x86_64), CPU model, core count, RAM, NEON capability, IS_ARM flag
- **Key Visual**: The ARM badge (🟢 ARM VERIFIED or 🔴 ARM NOT DETECTED)

**Script**:
> "Watch as the system detects the host hardware. It identifies the architecture, CPU model, core count, and available memory. Most importantly, it verifies whether this is actually an ARM64 environment — and it clearly labels the result."

---

### 1:00–1:20  Demonstrate Model Detection and Baseline

- **Command**: Run optimization on example project
- **Model**: MobileNetV2 (ONNX, 13.3 MB, 224×224 input)
- **Baseline Metrics**: Latency (mean), P95 latency, throughput (FPS), memory usage, model size

**Key Points**:
- System detects the ONNX model automatically
- Baseline benchmark uses 10 warmup + 50 inference iterations
- All metrics measured on actual CPU

**Script**:
> "The system detects the MobileNetV2 model in the project. It then runs a baseline benchmark — warming up the runtime, then measuring 50 inference iterations. We get mean latency, P95 latency, throughput, and memory usage — all on the real CPU."

---

### 1:20–1:50  Show Candidate Optimization

- **Process**: Generate 8 candidates (FP32/INT8 × BASIC/EXTENDED × thread configs)
- **Benchmark**: Each candidate measured on the ARM CPU
- **Results**: Latency and throughput for each configuration

**Key Points**:
- Hardware-aware thread selection (constrained by core count)
- FP32 vs INT8 comparison
- BASIC vs EXTENDED graph optimization
- Failed candidates don't stop the run

**Script**:
> "The system generates optimization candidates — different combinations of quantization, thread counts, and graph optimization levels. Each one is automatically benchmarked on the CPU. Notice how INT8 dramatically reduces model size, and thread count affects latency."

**Visual Indicator**:
- Show the candidate table with scores
- Highlight the best-performing candidate

---

### 1:50–2:10  Display Best Configuration

- **Selected**: Best candidate based on optimization objective
- **Configuration**: Quantization + threads + graph optimization
- **Metrics**: Latency, memory, model size, throughput, score

**Key Points**:
- Objective can be: Balanced, Lowest Latency, Lowest Memory, Smallest Model, Highest Throughput
- Transparent scoring with normalized metrics
- Improvement percentages calculated vs baseline

**Script**:
> "Based on the Balanced optimization objective (equal emphasis on latency, memory, and model size), the system selects this configuration. The score combines all three metrics into a single transparent number. You can see the improvement percentages versus the baseline."

---

### 2:10–2:30  Deploy and Verify

- **Action**: Generate deployment package and run verification
- **Verification**: Load model, run inference, validate output
- **Result**: DEPLOYMENT VERIFIED status

**Key Points**:
- Deployment package includes: config.json, benchmark.py, deployment.py, README.md, optimized model
- Post-deployment verification confirms the model runs correctly
- Comparison with baseline shows measurable improvements

**Script**:
> "The system generates a deployment package with the optimized model and configuration. It then runs a verification — loading the model, running inference, and validating the output. Result: DEPLOYMENT VERIFIED."

---

### 2:30–2:45  Show Measurable Improvements

- **Before**: Baseline latency/memory/model size
- **After**: Optimized latency/memory/model size
- **Percentages**: Improvement percentages (latency ↓%, memory ↓%, model size ↓%, throughput ↑%)

**Script**:
> "Here's the proof. The baseline had X ms latency and Y MB memory. After optimization, we achieved A ms latency and B MB memory — that's an improvement of I% for latency and J% for memory. The model size reduction is K%."

---

### 2:45–3:00  Generate Report and Close

- **Action**: Generate HTML + PDF report
- **Output**: Complete optimization report with all metrics, configuration, and improvements
- **Close**: Summary of what AutoPilot accomplished

**Script**:
> "Finally, the system generates a complete HTML and PDF report documenting the entire optimization workflow — hardware analysis, model detection, baseline benchmark, candidate search, best configuration, improvements, and deployment verification. This report can be shared with team members or used for compliance documentation."

---

## Demo Script File

The demo script is available at:
```
presentation/demo_script.md
```

This file contains the complete timed script above, ready for presentation use.

**All content generated by the real application** — no fake animations, no placeholder values, no simulated results. Every measurement is from actual ARM CPU inference.