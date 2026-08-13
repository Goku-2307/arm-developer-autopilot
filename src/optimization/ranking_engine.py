from typing import List, Dict, Any


class ScoringEngine:
    """
    Transparent scoring system for optimization candidates.

    Supports multiple optimization objectives with configurable weights.
    Normalizes metrics before scoring to ensure fair comparison.

    Objectives supported:
    - Balanced: Equal emphasis on latency, memory, and model size
    - Lowest Latency: Priority on minimizing latency
    - Lowest Memory: Priority on minimizing memory usage
    - Smallest Model: Priority on minimizing model size
    - Highest Throughput: Priority on maximizing throughput
    """

    # Default weights for Balanced objective
    DEFAULT_WEIGHTS = {
        "latency": 0.5,
        "memory": 0.3,
        "size": 0.2,
    }

    # Objective-specific weight configurations
    OBJECTIVE_WEIGHTS = {
        "balanced": {
            "latency": 0.5,
            "memory": 0.3,
            "size": 0.2,
        },
        "lowest_latency": {
            "latency": 0.8,
            "memory": 0.1,
            "size": 0.1,
        },
        "lowest_memory": {
            "latency": 0.2,
            "memory": 0.6,
            "size": 0.2,
        },
        "smallest_model": {
            "latency": 0.2,
            "memory": 0.2,
            "size": 0.6,
        },
        "highest_throughput": {
            "latency": 0.3,
            "memory": 0.3,
            "size": 0.1,
            "throughput": 0.3,
        },
    }

    @classmethod
    def rank(
        cls,
        candidates: List[Dict[str, Any]],
        objective: str = "balanced",
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates based on the selected optimization objective.

        Args:
            candidates: List of candidate result dicts from optimization search.
                Each dict should contain: latency, memory, model_size, throughput (if applicable).
            objective: Optimization objective name. One of:
                'balanced', 'lowest_latency', 'lowest_memory',
                'smallest_model', 'highest_throughput'

        Returns:
            List of candidate dicts with added 'score' field, sorted by score
            in descending order (highest score = best).
        """

        if objective not in cls.OBJECTIVE_WEIGHTS:
            print(
                f"Warning: Unknown objective '{objective}', "
                f"falling back to 'balanced'"
            )
            objective = "balanced"

        weights = cls.OBJECTIVE_WEIGHTS[objective]

        for candidate in candidates:

            latency = candidate.get("latency", float("inf"))
            memory = candidate.get("memory", float("inf"))
            size = candidate.get("model_size", float("inf"))
            throughput = candidate.get("throughput_fps", 0)

            # Normalize metrics using min-max normalization across all candidates
            # Lower is better for latency, memory, size
            # Higher is better for throughput

            all_latencies = [c.get("latency", float("inf")) for c in candidates]
            all_memories = [c.get("memory", float("inf")) for c in candidates]
            all_sizes = [c.get("model_size", float("inf")) for c in candidates]
            all_throughputs = [c.get("throughput_fps", 0) for c in candidates]

            min_latency = min(all_latencies) if all_latencies else latency
            max_latency = max(all_latencies) if all_latencies else latency
            min_memory = min(all_memories) if all_memories else memory
            max_memory = max(all_memories) if all_memories else memory
            min_size = min(all_sizes) if all_sizes else size
            max_size = max(all_sizes) if all_sizes else size
            max_throughput = max(all_throughputs) if all_throughputs else throughput
            min_throughput = min(all_throughputs) if all_throughputs else throughput

            # Avoid division by zero
            range_latency = max(max_latency - min_latency, 1e-10)
            range_memory = max(max_memory - min_memory, 1e-10)
            range_size = max(max_size - min_size, 1e-10)
            range_throughput = max(max_throughput - min_throughput, 1e-10)

            # Normalized scores (0 to 1, where 1 is best)
            # For "lower is better" metrics: (max - value) / range
            # For "higher is better" metrics: (value - min) / range

            norm_latency = (
                (max_latency - latency) / range_latency if range_latency else 1.0
            )
            norm_memory = (
                (max_memory - memory) / range_memory if range_memory else 1.0
            )
            norm_size = (
                (max_size - size) / range_size if range_size else 1.0
            )
            norm_throughput = (
                (throughput - min_throughput) / range_throughput
                if range_throughput
                else 1.0
            )

            # Weighted score contribution from each metric
            score_contribution = (
                weights.get("latency", 0) * norm_latency
                + weights.get("memory", 0) * norm_memory
                + weights.get("size", 0) * norm_size
            )

            # Add throughput contribution if objective supports it
            if "throughput" in weights:
                tp_contrib = weights.get("throughput", 0) * norm_throughput
                if isinstance(tp_contrib, (int, float)) and not isinstance(tp_contrib, bool):
                    if tp_contrib != tp_contrib:  # NaN check
                        tp_contrib = 0
                score_contribution += tp_contrib if isinstance(tp_contrib, (int, float)) and not isinstance(tp_contrib, bool) else 0

            # Check for NaN and fallback
            if score_contribution != score_contribution:  # NaN check
                score_contribution = 0.5  # Neutral fallback

            # Clamp to [0, 1] range and round
            clamped = max(min(float(score_contribution), 1.0), 0.0)
            candidate["score"] = round(clamped, 2)

        # Sort by score descending (higher is better)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates

    @classmethod
    def get_objective_info(cls, objective: str = "balanced") -> Dict[str, Any]:
        """
        Get information about an optimization objective.

        Args:
            objective: Objective name.

        Returns:
            Dict with weight breakdown and explanation.
        """
        if objective not in cls.OBJECTIVE_WEIGHTS:
            objective = "balanced"

        weights = cls.OBJECTIVE_WEIGHTS[objective]

        return {
            "objective": objective,
            "weights": weights,
            "description": cls._get_objective_description(objective),
        }

    @staticmethod
    def _get_objective_description(objective: str) -> str:
        """Get human-readable description of the objective."""
        descriptions = {
            "balanced": (
                "Equal emphasis on latency (50%), memory (30%), and model size (20%)"
            ),
            "lowest_latency": (
                "Priority on minimizing latency (80%), with secondary "
                "consideration of memory (10%) and model size (10%)"
            ),
            "lowest_memory": (
                "Priority on minimizing memory usage (60%), with secondary "
                "consideration of latency (20%) and model size (20%)"
            ),
            "smallest_model": (
                "Priority on minimizing model size (60%), with secondary "
                "consideration of latency (20%) and memory (20%)"
            ),
            "highest_throughput": (
                "Priority on maximizing throughput, with balanced consideration "
                "of latency (30%), memory (30%), and model size (10%)"
            ),
        }
        return descriptions.get(
            objective,
            "Balanced optimization objective",
        )


# Backward compatibility alias
RankingEngine = ScoringEngine