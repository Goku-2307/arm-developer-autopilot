import os
import platform
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class CPUInfo:
    model_name: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    frequency_mhz: Optional[float] = None
    flags: list = None


@dataclass
class MemoryInfo:
    total_mb: int = 0
    available_mb: int = 0


@dataclass
class HardwareProfile:
    architecture: str = ""
    machine: str = ""
    cpu_model: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    memory_total_mb: int = 0
    memory_available_mb: int = 0
    neon: bool = False
    os: str = ""
    kernel: str = ""
    python_version: str = ""
    onnxruntime_version: str = ""
    numpy_version: str = ""
    is_arm: bool = False
    process_architecture: str = ""
    hostname: str = ""


def _get_kernel() -> str:
    try:
        with open("/proc/version", "r") as f:
            return f.read().strip()
    except Exception:
        return platform.uname().release


def _detect_neon() -> bool:
    try:
        machine = platform.machine().lower()
        if "aarch64" in machine or "arm" in machine:
            return True
        # Check /proc/cpuinfo for NEON
        with open("/proc/cpuinfo", "r") as f:
            content = f.read()
        neon_keywords = ["neon", "asimd", "fphp"]
        return any(
            kw in content.lower() for kw in neon_keywords
        )
    except Exception:
        return False


def profile() -> HardwareProfile:
    """
    Generate a complete hardware profile for the current system.
    Returns structured hardware information suitable for ARM verification.
    """
    uname = platform.uname()

    architecture = uname.machine.lower()
    machine = uname.machine
    process_architecture = architecture

    # Detect if ARM64
    is_arm = architecture in ("aarch64", "arm64") or "arm" in architecture.lower()

    # CPU information
    cpu_model = ""
    physical_cores = 0
    logical_cores = 0
    frequency_mhz = None

    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()

        # Count physical and logical cores
        processors = len(re.findall(r"^processor\s+:", cpuinfo, re.MULTILINE))
        physical_ids = len(re.findall(r"^physical id\s+:", cpuinfo, re.MULTILINE))

        logical_cores = processors
        if physical_ids > 0:
            physical_cores = len(set(
                re.findall(r"^physical id\s+:\s+(\d+)", cpuinfo, re.MULTILINE)
            ))
        else:
            physical_cores = logical_cores

        # Get CPU model name (first one)
        model_match = re.search(
            r"^model name\s+:\s+(.+)$",
            cpuinfo, re.MULTILINE
        )
        if model_match:
            cpu_model = model_match.group(1).strip()

    except Exception:
        pass

    # Memory information
    memory_total_mb = 0
    memory_available_mb = 0
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()

        total_match = re.search(
            r"^MemTotal:\s+(\d+)", meminfo, re.MULTILINE
        )
        available_match = re.search(
            r"^MemAvailable:\s+(\d+)", meminfo, re.MULTILINE
        )

        if total_match:
            memory_total_mb = int(int(total_match.group(1)) / 1024)
        if available_match:
            memory_available_mb = int(int(available_match.group(1)) / 1024)
    except Exception:
        pass

    # NEON capability
    neon = _detect_neon()

    # OS and kernel
    os_name = uname.system
    kernel = _get_kernel()

    # Python version
    python_version = platform.python_version()

    # ONNX Runtime version
    onnxruntime_version = ""
    try:
        import onnxruntime
        onnxruntime_version = onnxruntime.__version__
    except Exception:
        pass

    # NumPy version
    numpy_version = ""
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        pass

    # Hostname
    try:
        hostname = platform.node()
    except Exception:
        hostname = ""

    return HardwareProfile(
        architecture=architecture,
        machine=machine,
        cpu_model=cpu_model if cpu_model else "Unknown",
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        memory_total_mb=memory_total_mb,
        memory_available_mb=memory_available_mb,
        neon=neon,
        os=os_name,
        kernel=kernel,
        python_version=python_version,
        onnxruntime_version=onnxruntime_version,
        numpy_version=numpy_version,
        is_arm=is_arm,
        process_architecture=process_architecture,
        hostname=hostname,
    )


def is_arm_environment(
    min_cores: int = 1,
    min_memory_mb: int = 256,
) -> Dict[str, Any]:
    """
    Check if the current environment meets minimum ARM requirements.

    Returns:
        Dict with 'is_arm', 'supported', and 'reason' keys.
    """
    profile_data = profile()

    profile_dict = {
        "architecture": profile_data.architecture,
        "machine": profile_data.machine,
        "cpu_model": profile_data.cpu_model,
        "physical_cores": profile_data.physical_cores,
        "logical_cores": profile_data.logical_cores,
        "memory_total_mb": profile_data.memory_total_mb,
        "is_arm": profile_data.is_arm,
        "neon": profile_data.neon,
        "os": profile_data.os,
    }

    if not profile_dict["is_arm"]:
        return {
            "is_arm": False,
            "architecture": profile_dict["architecture"],
            "supported": False,
            "reason": "ARM64/aarch64 hardware is required for real ARM benchmarking",
        }

    if profile_dict["physical_cores"] < min_cores:
        return {
            "is_arm": True,
            "architecture": profile_dict["architecture"],
            "supported": False,
            "reason": f"Detected {profile_dict['physical_cores']} core(s), minimum {min_cores} required",
        }

    if profile_dict["memory_total_mb"] < min_memory_mb:
        return {
            "is_arm": True,
            "architecture": profile_dict["architecture"],
            "supported": False,
            "reason": f"Detected {profile_dict['memory_total_mb']} MB RAM, minimum {min_memory_mb} MB required",
        }

    return {
        "is_arm": True,
        "architecture": profile_dict["architecture"],
        "supported": True,
        "reason": "ARM64 Linux environment detected and verified",
    }