class CandidateGenerator:
    """
    Generates optimization candidates.

    Optimized for hackathon demo:
    - Compare FP32 vs INT8
    - Compare BASIC vs EXTENDED graph optimization
    - Thread count is selected separately by RuntimeTuner
    """

    def __init__(self):

        self.quantizations = [
            "FP32",
            "INT8"
        ]

        self.graph_optimizations = [
            "BASIC",
            "EXTENDED"
        ]

        self.execution_modes = [
            "PARALLEL"
        ]

    def generate(self):

        candidates = []

        candidate_id = 1

        for quantization in self.quantizations:

            for graph in self.graph_optimizations:

                candidate = {

                    "candidate_id": candidate_id,

                    "quantization": quantization,

                    # RuntimeTuner will overwrite this value
                    "threads": 4,

                    "graph_optimization": graph,

                    "execution_mode": "PARALLEL"

                }

                candidates.append(candidate)

                candidate_id += 1

        return candidates

    def print_candidates(self):

        candidates = self.generate()

        print("=" * 60)
        print("Optimization Candidates")
        print("=" * 60)

        for candidate in candidates:

            print()

            print(f"Candidate {candidate['candidate_id']}")
            print("-" * 30)

            print(f"Quantization       : {candidate['quantization']}")
            print(f"Threads            : {candidate['threads']}")
            print(f"Graph Optimization : {candidate['graph_optimization']}")
            print(f"Execution Mode     : {candidate['execution_mode']}")

        print()
        print(f"Total Candidates : {len(candidates)}")