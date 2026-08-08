from src.services.optimization_service import OptimizationService

service = OptimizationService()

session = service.optimize_project(
    "examples/sample_ai_project"
)

print()

print("=" * 60)
print("SESSION SUMMARY")
print("=" * 60)

print(session.project_name)
print(session.language)
print(session.best_result)