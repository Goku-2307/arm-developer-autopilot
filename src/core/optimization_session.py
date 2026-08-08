from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class OptimizationSession:
    """
    Central object that stores everything related to
    one optimization run.
    """

    # -----------------------------
    # Project Information
    # -----------------------------

    project_name: str = ""

    project_path: str = ""

    language: str = ""

    # -----------------------------
    # AI Models
    # -----------------------------

    models: List[Dict[str, Any]] = field(default_factory=list)

    # -----------------------------
    # ARM Information
    # -----------------------------

    arm_device: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------
    # Optimization
    # -----------------------------

    optimization_candidates: List[Dict[str, Any]] = field(default_factory=list)

    benchmark_results: List[Dict[str, Any]] = field(default_factory=list)

    best_result: Dict[str, Any] = field(default_factory=dict)

    # -----------------------------
    # Timeline
    # -----------------------------

    events: List[Dict[str, Any]] = field(default_factory=list)

    # -----------------------------
    # Reports
    # -----------------------------

    report_path: str = ""

    # -----------------------------
    # GitHub
    # -----------------------------

    github_repo: str = ""

    github_commit: str = ""

    # -----------------------------
    # Statistics
    # -----------------------------

    total_time: float = 0

    optimization_score: float = 0

    improvement: float = 0