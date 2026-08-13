import multiprocessing
from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class Candidate:
    """Represents an optimization candidate configuration."""

    candidate_id: str
    quantization: str
    threads: int
    graph_optimization: str
    execution_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "quantization": self.quantization,
            "threads": self.threads,
            "graph_optimization": self.graph_optimization,
            "execution_mode": self.execution_mode,
        }


class CandidateGenerator:
    """
    Generates optimization candidates for ARM CPU inference.

    Candidates are generated based on actual hardware capabilities.
    Thread configurations are determined by the detected device's core count.

    Supported dimensions:

    ### Quantization

    - FP32: Original floating-point precision
    - INT8: 8-bit integer quantization (model size reduction)

    ### Graph Optimization

    - BASIC: Basic graph-level optimizations
    - EXTENDED: Extended graph-level optimizations (e.g., operator fusion)

    ### Threads

    Determined by hardware. Test only values that make sense
    for the detected device (e.g., do not test 8 threads on a 2-core device).
    """

    def __init__(self):

        self.quantizations = ["FP32", "INT8"]
        self.graph_optimizations = ["BASIC", "EXTENDED"]
        self.execution_modes = ["PARALLEL"]

        # Determine safe thread configurations based on detected hardware
        self._candidate_threads = self._determine_thread_config()

    def _determine_thread_config(self) -> List[int]:
        """
        Determine appropriate thread configurations based on detected hardware.

        Returns:
            List of thread counts to test, constrained by the physical core count.
        """
        try:
            cpu_count = multiprocessing.cpu_count()
        except Exception:
            cpu_count = 4

        # Check if this is an ARM environment
        from src.arm.hardware_profiler import is_arm_environment

        env_check = is_arm_environment(min_cores=1, min_memory_mb=256)

        candidates = [1]  # Always test 1 thread

        # Add thread counts that make sense for the device
        if cpu_count >= 2 and env_check["is_arm"]:
            candidates.append(2)
        if cpu_count >= 4 and env_check["is_arm"]:
            candidates.append(4)
        if cpu_count >= 8 and env_check["is_arm"]:
            candidates.append(8)
        if cpu_count >= 12 and env_check["is_arm"]:
            candidates.append(12)
        if cpu_count >= 16 and env_check["is_arm"]:
            candidates.append(16)

        # If not ARM, still allow reasonable thread counts but cap at 8
        if not env_check["is_arm"]:
            candidates = sorted(set(candidates))
            if 8 not in candidates and len(candidates) < 8:
                candidates.append(min(8, cpu_count))
            candidates = sorted(candidates)

        return sorted(set(candidates))

    def generate(self) -> List[Candidate]:
        """
        Generate all optimization candidates.

        Returns:
            List of Candidate objects representing all combinations
            of quantization, graph optimization, and threads.
        """

        candidates = []
        candidate_id = 1

        for quantization in self.quantizations:
            for graph in self.graph_optimizations:
                for threads in self._candidate_threads:

                    candidate = Candidate(

                        candidate_id=f"candidate_{candidate_id:02d}",

                        quantization=quantization,

                        threads=threads,

                        graph_optimization=graph,

                        execution_mode="PARALLEL",

                    )
                    candidates.append(candidate)
                    candidate_id += 1

        return candidates

    def generate_dicts(self) -> List[Dict[str, Any]]:
        """
        Generate candidate dicts (for compatibility with optimization search).

        Returns:
            List of candidate dicts.
        """

        candidates = []
        candidate_id = 1

        for quantization in self.quantizations:
            for graph in self.graph_optimizations:
                for threads in self._candidate_threads:

                    candidate = {

                        "candidate_id": f"candidate_{candidate_id:02d}",

                        "quantization": quantization,

                        "threads": threads,

                        "graph_optimization": graph,

                        "execution_mode": "PARALLEL"

                    }
                    candidates.append(candidate)
                    candidate_id += 1

        return candidates

    def print_candidates(self, candidates=None):
        """Print candidate configurations in a readable format."""
        if candidates is None:
            candidates = self.generate()

        print("=" * 60)
        print("Optimization Candidates")
        print("=" * 60)

        for candidate in candidates:
            print()

            if isinstance(candidate, Candidate):
                print(f"Candidate {candidate.candidate_id}")
                print("-" * 30)

                print(f"Quantization       : {candidate.quantization}")
                print(f"Threads            : {candidate.threads}")
                print(f"Graph Optimization : {candidate.graph_optimization}")
                print(f"Execution Mode     : {candidate.execution_mode}")
            else:
                print(f"Candidate {candidate['candidate_id']}")
                print("-" * 30)

                print(f"Quantization       : {candidate['quantization']}")
                print(f"Threads            : {candidate['threads']}")
                print(f"Graph Optimization : {candidate['graph_optimization']}")
                print(f"Execution Mode     : {candidate['execution_mode']}")

        print()
        print(f"Total Candidates : {len(candidates)}")
        print(f"Thread config    : {self._candidate_threads}")