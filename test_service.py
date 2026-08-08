from src.services.optimization_service import OptimizationService

service = OptimizationService()

result = service.optimize_project(

    "examples/sample_ai_project"

)

print()

print(result)