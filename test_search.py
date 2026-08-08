from src.optimization.candidate_generator import CandidateGenerator
from src.optimization.optimization_search import OptimizationSearch

generator = CandidateGenerator()

candidates = generator.generate()

search = OptimizationSearch("models/model.onnx")

results = search.execute(candidates[:3])

print(results)