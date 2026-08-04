ARM_DATABASE = {

    "ONNX": {

        "runtime": "ONNX Runtime",

        "quantization": "INT8",

        "thread_tuning": True,

        "graph_optimization": "Level 3",

        "neon": True,

        "arm_compute_library": True,

        "expected_speedup": "2.5x",

        "memory_reduction": "60%",

        "power_reduction": "35%"
    },

    "PyTorch": {

        "runtime": "TorchScript",

        "quantization": "Dynamic INT8",

        "thread_tuning": True,

        "graph_optimization": "JIT",

        "neon": True,

        "arm_compute_library": False,

        "expected_speedup": "1.8x",

        "memory_reduction": "40%",

        "power_reduction": "20%"
    },

    "TensorFlow Lite": {

        "runtime": "TensorFlow Lite",

        "quantization": "INT8",

        "thread_tuning": True,

        "graph_optimization": "Delegate",

        "neon": True,

        "arm_compute_library": True,

        "expected_speedup": "3x",

        "memory_reduction": "65%",

        "power_reduction": "45%"
    }

}