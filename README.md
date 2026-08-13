# ARM Developer AutoPilot

## Problem

Deploying AI models efficiently on ARM-based edge devices requires careful configuration and optimization. Unlike x86 servers, ARM CPUs have constrained resources:

- **Latency**: Real-time inference requirements are stricter on resource-constrained devices
- **Memory**: Limited RAM (typically 256MB-4GB on edge devices) requires INT8 quantization
- **Compute**: ARM CPUs lack the raw frequency of server-grade x86 processors
- **Model Size**: Mobile and embedded models need to be compact (often <10MB)
- **Device-Specific Runtimes**: ONNX Runtime, TensorFlow Lite, and PyTorch Mobile have different ARM optimization paths

Without automated optimization, developers must manually experiment with quantization, thread counts, and graph optimizations - a time-consuming process that requires deep expertise in both the model and the target hardware.

## Solution

**ARM Developer AutoPilot** is an autonomous AI optimization platform that allows developers to provide an AI/ML project or ONNX model and automatically:

1. Detect the execution hardware and verify ARM64 availability
2. Detect AI models inside the supplied project
3. Establish a baseline inference benchmark
4. Generate ARM-aware optimization candidates
5. Optimize the model using supported techniques (dynamic quantization)
6. Execute every candidate on the actual CPU
7. Measure latency, throughput, memory usage, and model size
8. Score and rank all candidates
9. Select the best configuration according to the selected optimization objective
10. Explain why the configuration was selected
11. Generate an ARM deployment package
11. Deploy and run the optimized model locally
12. Verify the optimized model with a second benchmark
13. Show baseline vs optimized results
14. Generate HTML/PDF reports
15. Provide a polished Streamlit dashboard

The central product workflow is:

```text
Developer Project / AI Model
            |
            v
     Hardware Detection
            |
            v
      ARM Verification
            |
            v
       Project Analysis
            |
            v
       Model Detection
            |
            v
      Baseline Benchmark
            |
            v
   ARM-Aware Candidate Search
            |
            v
    Model Optimization
            |
            v
   Real ARM CPU Benchmark
            |
            v
      Score + Ranking
            |
            v
    Best Configuration
            |
            v
     Deployment Package
            |
            v
       ARM Deployment
            |
            v
     Verification Run
            |
            v
  Baseline vs Optimized
            |
            v
       Final Report
```

The product communicates this as an autonomous optimization agent rather than merely a static benchmarking dashboard.

## Why ARM?

ARM architecture dominates the edge and mobile computing landscape:

- **Market Share**: Over 95% of mobile devices and most IoT edge devices use ARM
- **Energy Efficiency**: ARM CPUs deliver performance-per-watt that x86 cannot match
- **Ubiquity**: ARM is found in Raspberry Pi, Google Coral, NVIDIA Jetson, AWS Graviton, and custom SoCs
- **Software Ecosystem**: ONNX Runtime, TensorFlow Lite, and PyTorch all provide strong ARM support

**Constraints specific to ARM deployment:**

| Constraint | Impact | AutoPilot Solution |
|---|---|---|
| Limited RAM | INT8 quantization reduces model size 4-8x | Automatic INT8 dynamic quantization |
| Variable core count | Thread count must match core availability | Hardware-aware candidate generation |
| NEON SIMD capability | Accelerates math operations | Hardware NEON detection and reporting |
| CPU frequency | Lower than x86, requires optimization | Baseline benchmarking and optimization objectives |
| Model size | Must fit on device | INT8 + graph optimization pipeline |

## Architecture

```text
                    Developer
                        |
                        v
             ARM Developer AutoPilot
                        |
          +-------------+-------------+
          |                           |
          v                           v
 Hardware Profiler              Project Analyzer
          |                           |
          v                           v
 ARM Verification                AI Detector
          |                           |
          +-------------+-------------+
                        |
                        v
                Baseline Benchmark
                        |
                        v
             Optimization Agent
                        |
             +----------+----------+
             |          |          |
             v          v          v
         Quantization Threads   Graph Opt.
             |          |          |
             +----------+----------+
                        |
                        v
                ARM CPU Runtime
                        |
                        v
                  Benchmarking
                        |
                        v
                  Score + Rank
                        |
                        v
               Best Configuration
                        |
                        v
                Deployment Package
                        |
                        v
                  ARM Inference
                        |
                        v
                 Verification
                        |
                        v
               Final Report
```

## Agent Workflow

```text
Hardware Detection
    ↓
Model Detection
    ↓
Baseline Benchmark
    ↓
Candidate Search
    ↓
ARM Benchmarking
    ↓
Ranking
    ↓
Best Configuration
    ↓
Deployment
    ↓
Verification
    ↓
Final Report
```

## Features

| Feature | Description |
|---|---|
| **Hardware Profiler** | Detects architecture, CPU model, core count, RAM, NEON capability |
| **ARM Verification** | Validates ARM64 environment; displays clear badge status |
| **Model Detection** | Finds ONNX, TFLite, PyTorch models; extracts metadata |
| **Baseline Benchmark** | Warmup + benchmark iterations; mean/P95 latency, throughput, memory |
| **Optimization Search** | FP32/INT8 × BASIC/EXTENDED × thread configurations |
| **Hardware-Aware Candidates** | Thread counts constrained by detected core count |
| **Objective-Based Scoring** | Balanced, Lowest Latency, Lowest Memory, Smallest Model, Highest Throughput |
| **Transparent Scoring** | Min-max normalized metrics with explainable weights |
| **Baseline vs Optimized** | Improvement percentages for latency, memory, model size, throughput |
| **Explanation Engine** | Deterministic explanation of configuration selection |
| **Deployment Package** | config.json, benchmark.json, deployment.py, README.md |
| **Deployment Runner** | Validates ARM environment and runs inference verification |
| **Post-Department Verification** | Confirms optimized model runs correctly after deployment |
| **Benchmarking** | Configurable warmup/iterations; mean/median/P95 latency; throughput |
| **Streamlit Dashboard** | Full UI with all sections (hardware, project, baseline, candidates, etc.) |
| **CLI** | `profile`, `verify`, `optimize`, `deploy`, `report` commands |
| **Reporting** | HTML and PDF reports with complete optimization history |
| **Session Management** | Local JSON storage of optimization runs |

## Tech Stack

- **Python 3.13+**: Core programming language
- **ONNX Runtime 1.28+**: Inference engine with CPUExecutionProvider
- **ONNX 1.22+**: Model format for optimization and deployment
- **NumPy**: Numerical operations and input data generation
- **PSUtil**: System monitoring and memory measurement
- **Plotly**: Interactive charts in the dashboard
- **Rich**: Rich text formatting for CLI and logs
- **Streamlit 1.61+**: Web dashboard framework
- **ReportLab**: PDF report generation
- **Jinja2**: HTML template rendering (via Streamlit)

## Supported Runtime

- **ONNX Runtime CPU Execution Provider**: Primary inference runtime
- **Supported Models**: ONNX (.onnx), TensorFlow Lite (.tflite), PyTorch (.pt, .torchscript)
- **Quantization**: FP32 (original) → INT8 (dynamic quantization)
- **Graph Optimization**: BASIC (basic fusion) → EXTENDED (advanced optimizations)
- **Thread Configuration**: 1 to N threads based on physical core count

## Installation

```bash
# Clone the repository
git clone https://github.com/Goku-2307/arm-developer-autopilot.git
cd arm-developer-autopilot

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m src.cli profile
```

**Requirements:**

```
streamlit
onnx
onnxruntime
numpy
pandas
matplotlib
plotly
psutil
rich
pygithub
reportlab
jinja2
```

## Running on ARM

The application automatically detects the hardware environment:

```bash
# Check hardware profile
python -m src.cli profile

# Verify ARM environment
python -m src.cli verify
```

**On ARM64/aarch64:**
- ✅ ARM VERIFIED badge displayed
- ✅ Full optimization workflow runs
- ✅ Real ARM CPU benchmarks executed
- ✅ Results are genuine ARM performance

**On x86_64:**
- 🔴 ARM NOT DETECTED badge displayed
- ⚠️ Optimization runs in development mode
- ⚠️ Results are marked as x86 measurements, not ARM
- ⚠️ No false claims of ARM performance

> **Important**: The product clearly distinguishes between ARM and x86 environments. It never fabricates ARM results or reports x86 measurements as ARM performance.

## Running the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard opens at `http://localhost:8502` and contains:

### Section 1 — Hardware
- Architecture badge (🟢 ARM VERIFIED / 🔴 ARM NOT DETECTED)
- CPU model, core count, RAM size
- NEON capability, ONNX Runtime version

### Section 2 — Project
- Project name and language
- Models detected with framework info

### Section 3 — Baseline
- Latency (mean/P95), throughput, memory, model size
- Configurable warmup and benchmark iterations

### Section 4 — Optimization Objective
- Radio selection: Balanced / Lowest Latency / Lowest Memory / Smallest Model / Highest Throughput
- Weight breakdown and description

### Section 5 — Agent Progress
- Timeline of optimization stages with status indicators

### Section 6 — Candidate Table
- All candidates with quantization, threads, graph optimization
- Latency, P95, memory, model size, throughput, score
- Best candidate highlighted

### Section 7 — Best Configuration
- Visually prominent recommended configuration
- Latency, memory, model size metrics
- Baseline vs optimized improvement percentages

### Section 8 — Improvement Panel
- Side-by-side baseline vs optimized comparison
- Latency ↓%, Memory ↓%, Model Size ↓%, Throughput ↑%

### Section 9 — Why This Configuration?
- Deterministic explanation of configuration selection
- Key factors: quantization benefit, thread performance, graph optimization
- Reference to selected optimization objective

### Section 10 — Reports & Deployment
- Generate HTML + PDF reports
- Generate deployment package (config.json, benchmark.py, deployment.py, README.md)
- Run post-deployment verification

## CLI

```bash
# Show hardware profile
python -m src.cli profile

# Verify ARM environment
python -m src.cli verify

# Run optimization on a project
python -m src.cli optimize /path/to/project

# Deploy optimized model
python -m src.cli deploy <session_id>

# Generate report
python -m src.cli report <session_id>
```

**CLI Help:**

```bash
python -m src.cli --help
```

## Example

The example project demonstrates the complete workflow:

```
examples/sample_ai_project/
├── mobilenetv2.onnx     # MobileNetV2 model (13.3 MB, ONNX opset 12)
├── requirements.txt     # Python dependencies
└── README.md            # Example project description
```

The model is MobileNetV2, a lightweight image classification network suitable for edge deployment.

**Benchmark Results (on example model):**

| Metric | Value |
|---|---|
| Model Size | 13.32 MB |
| Input Shape | (1, 3, 224, 224) |
| Output Shape | (1, 1000) |
| Opset Version | 12 |
| Framework | ONNX Runtime |

*Note: Benchmark values vary depending on host hardware.*

## Project Structure

```
arm-developer-autopilot/
├── src/                   # Source code
│   ├── analysis/          # Project analysis and model detection
│   ├── arm/               # ARM hardware profiler and verifier
│   ├── benchmarking/      # Model benchmarking on ARM CPU
│   ├── optimization/      # Optimization engine, search, ranking
│   ├── deployment/        # Deployment package and runner
│   ├── reports/           # Report generation
│   ├── services/          # High-level orchestration services
│   ├── core/              # Session management
│   ├── cli.py             # Command-line interface
│   └── dashboard.py       # Streamlit web dashboard
├── config/                # Configuration files (ARM devices, optimization profiles)
├── examples/              # Example AI project for demonstration
│   └── sample_ai_project/
│       ├── mobilenetv2.onnx
│       ├── app.py
│       └── requirements.txt
├── tests/                 # Unit tests (35+ tests)
├── reports/               # Generated optimization reports
├── deployment/           # Generated deployment packages
├── dashboard.py           # Streamlit dashboard entry point
├── main.py                # Legacy entry point
├── dashboard.py           # Streamlit dashboard
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Testing

The project includes 35+ unit tests covering:

| Test Category | Tests |
|---|---|
| Hardware Profiler | 7 tests |
| ARM Verifier | 8 tests |
| Model Detector | 5 tests |
| Candidate Generator | 6 tests |
| Scoring Engine | 7 tests |
| Post-Department Verifier | 4 tests |

**Running tests:**

```bash
pytest tests/ -v
```

**ARM integration tests:**

```bash
pytest -m arm
```

These tests will be skipped on x86 with an explanatory message.

## Limitations

| Limitation | Description |
|---|---|
| **No Physical ARM Required for Unit Tests** | All unit tests mock hardware detection; can run on x86 |
| **INT8 Performance Varies** | INT8 dynamic quantization speedup depends on ARM chip NEON support |
| **Single Model Support** | Optimizes one model at a time (multiple models processed sequentially) |
| **No Cloud Dependency** | Core optimization runs locally; no internet required |
| **No Model Training** | Focuses on inference optimization, not training or transfer learning |
| **Benchmark Variability** | Results depend on host OS load, Python version, and ONNX Runtime version |

## Future Scope

- **TensorRT Integration**: NVIDIA GPU acceleration for compatible ARM devices
- **Advanced Quantization**: AQT, QAT quantization-aware training support
- **Multi-Model Optimization**: Simultaneous optimization of multiple models
- **Cross-Compilation**: Support for cross-compiling models between different ARM chips
- **Auto-Tuning**: Reinforcement learning-based thread and configuration auto-tuning
- **Dashboard Enhancements**: Real-time monitoring, historical comparison, multi-view charts
- **ONNX Model Zoo**: Pre-optimized models for common architectures (MobileNet, YOLO, etc.)

## Hackathon Demo

**Demo Duration**: 2-3 minutes

**Demo Sequence**:

1. **0:00-0:20** - Explain the problem: efficient AI deployment on ARM edge devices
2. **0:20-0:40** - Introduce ARM Developer AutoPilot and its value proposition
3. **0:40-1:00** - Show real ARM hardware detection (or x86 development mode with clear labeling)
4. **1:00-1:20** - Demonstrate model detection and baseline benchmark
5. **1:20-1:50** - Show candidate generation and optimization search
6. **1:50-2:10** - Display best configuration and improvement percentages
7. **2:10-2:30** - Deploy and verify the optimized model
8. **2:30-2:45** - Show measurable improvements with before/after comparison
9. **2:45-3:00** - Generate and download the optimization report

**Everything shown should be generated by the real application** - no fake animations or placeholder values.

## Benchmark Results

*(Include real benchmark data from your testing environment)*

See the dashboard and CLI output for real-time benchmark results. All values are generated from actual inference measurements on the host hardware.

## Project Structure

Refer to the "Project Structure" section above for a complete file listing.

## Testing

Refer to the "Testing" section above for test details and commands.

## Limitations

Refer to the "Limitations" section above for known constraints and assumptions.

## Future Scope

Refer to the "Future Scope" section above for planned features and enhancements.

## Hackathon Demo

Refer to the "Hackathon Demo" section above for the demo sequence and script.