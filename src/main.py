from pipeline import OptimizationPipeline

pipeline = OptimizationPipeline(
    "examples/sample_ai_project",
    "Raspberry Pi 5"
)

results = pipeline.run()

print(results)