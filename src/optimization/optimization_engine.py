from src.optimization.candidate_generator import CandidateGenerator
from src.optimization.optimization_search import OptimizationSearch
from src.optimization.ranking_engine import RankingEngine


class OptimizationEngine:
    """
    Main optimization engine.

    Responsibilities
    ----------------
    1. Generate optimization candidates
    2. Execute every candidate
    3. Rank every candidate
    4. Recommend the best configuration
    """

    def __init__(self, model_path):

        self.model_path = model_path

    def optimize(self):

        print("\n" + "=" * 60)
        print("ARM Optimization Engine")
        print("=" * 60)

        # -----------------------------------
        # Generate Candidates
        # -----------------------------------

        generator = CandidateGenerator()

        candidates = generator.generate()

        print(f"\nGenerated {len(candidates)} optimization candidates")

        # -----------------------------------
        # Search
        # -----------------------------------

        search = OptimizationSearch(self.model_path)

        results = search.execute(candidates)

        if len(results) == 0:

            print("\nNo successful optimization candidates.")

            return None

        # -----------------------------------
        # Rank
        # -----------------------------------

        ranking = RankingEngine()

        ranked = ranking.rank(results)

        best = ranked[0]

        print("\n" + "=" * 60)
        print("BEST CONFIGURATION")
        print("=" * 60)

        print(f"Candidate ID       : {best['candidate_id']}")
        print(f"Quantization       : {best['quantization']}")
        print(f"Threads            : {best['threads']}")
        print(f"Graph Optimization : {best['graph_optimization']}")
        print(f"Execution Mode     : {best['execution_mode']}")
        print(f"Latency            : {best['latency']} ms")
        print(f"Memory             : {best['memory']} MB")
        print(f"Model Size         : {best['model_size']} MB")
        print(f"Score              : {best['score']}")

        return {

            "best_configuration": best,

            "ranking": ranked,

            "results": results

        }