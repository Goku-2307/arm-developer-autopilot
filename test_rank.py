from src.optimization.ranking_engine import RankingEngine

results = [

    {
        "candidate_id": 1,
        "latency": 12,
        "memory": 65,
        "model_size": 28
    },

    {
        "candidate_id": 2,
        "latency": 9,
        "memory": 60,
        "model_size": 14
    },

    {
        "candidate_id": 3,
        "latency": 8,
        "memory": 58,
        "model_size": 13
    }

]

engine = RankingEngine()

engine.print_ranking(results)