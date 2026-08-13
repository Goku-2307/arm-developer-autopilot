from typing import List, Dict, Any


class ExplanationEngine:
    """
    Deterministic explanation system for optimization results.

    Generates human-readable explanations for why a particular
    configuration was selected as the best, based on the optimization
    objective and benchmark results.

    The explanations are generated from the actual data, not from
    an LLM, ensuring reproducibility and transparency.
    """

    @staticmethod
    def generate(
        best_candidate: Dict[str, Any],
        baseline: Dict[str, Any],
        optimized: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        objective: str,
    ) -> str:
        """
        Generate a deterministic explanation for the selected configuration.

        Args:
            best_candidate: The best candidate dict that was selected.
            baseline: The baseline benchmark result dict.
            optimized: The optimized benchmark result dict (best candidate).
            candidates: List of all candidate result dicts.
            objective: The optimization objective name.

        Returns:
            String containing the formatted explanation.
        """
        lines = []

        # Header
        lines.append("WHY THIS CONFIGURATION?")
        lines.append("")
        quant = best_candidate.get("quantization", "?")
        threads = best_candidate.get("threads", "?")
        graph = best_candidate.get("graph_optimization", "?")
        lines.append(f"Configuration: {quant} quantization, {threads} threads, {graph} graph optimization")
        lines.append("")

        # Build explanation components
        reasons = []

        # Quantization reason
        if quant == "INT8":
            size_before = baseline.get("model_size_mb", 0)
            size_after = optimized.get("model_size_mb", 0)
            if size_before > 0:
                reduction = round(
                    ((size_before - size_after) / size_before) * 100, 2
                )
                reasons.append(
                    f"• INT8 reduced the model footprint significantly "
                    f"(model size {reduction}% smaller)"
                )

        # Thread reason
        threads_val = best_candidate.get("threads", 0)
        if threads_val > 0:
            latency_before = baseline.get("latency_mean_ms", float("inf"))
            latency_after = optimized.get("latency_mean_ms", float("inf"))
            if latency_before > 0 and latency_after > 0:
                improvement = round(
                    ((latency_before - latency_after) / latency_before) * 100, 2
                )
                if improvement != 0:
                    direction = "improved" if improvement > 0 else "worsened"
                    reasons.append(
                        f"• {threads_val} threads {direction} latency by {abs(improvement)}%"
                    )

        # Graph optimization reason
        graph_val = best_candidate.get("graph_optimization", "")
        if graph_val:
            reasons.append(
                f"• {graph_val} graph optimization was selected as "
                f"the graph optimization strategy"
            )

        # Objective reason
        reasons.append(
            f"• This configuration achieved the highest score under "
            f"the {objective} optimization objective"
        )

        # Tradeoffs
        lines.append("Key factors:")
        for reason in reasons:
            lines.append(reason)

        # Note about other candidates
        lines.append("")
        better_candidates = [
            c
            for c in candidates
            if c.get("candidate_id", "") != best_candidate.get("candidate_id", "")
            and c.get("score", 0) > best_candidate.get("score", 0)
        ]
        if better_candidates:
            lines.append(
                f"• Note: {len(better_candidates)} other candidate(s) "
                f"had higher raw metric values, but this configuration"
            )
            lines.append(
                f"  achieved the best overall score for the selected objective"
            )

        lines.append("")
        explanation = "\n".join(lines)
        return explanation