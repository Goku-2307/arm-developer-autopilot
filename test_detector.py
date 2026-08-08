from src.analysis.ai_detector import AIDetector

detector = AIDetector(
    "examples/sample_ai_project"
)

models = detector.detect_models()

print(models)