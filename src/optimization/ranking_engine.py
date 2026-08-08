class RankingEngine:
    """
    Ranks optimization candidates using a weighted score.
    """

    def rank(self, candidates):

        for candidate in candidates:

            latency = candidate["latency"]
            memory = candidate["memory"]
            size = candidate["model_size"]

            latency_score = 50 / latency
            memory_score = 20 / max(memory, 1)
            size_score = 30 / max(size, 1)

            score = (
                latency_score +
                memory_score +
                size_score
            )

            # Bonus for INT8 because it's the preferred
            # deployment format on resource-constrained devices.
            if candidate["quantization"] == "INT8":
                score += 15

            candidate["score"] = round(score, 2)

        candidates.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return candidates