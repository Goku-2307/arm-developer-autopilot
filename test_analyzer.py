from src.analysis.analyzer import ProjectAnalyzer

analyzer = ProjectAnalyzer(
    "examples/sample_ai_project"
)

print(analyzer.project_summary())