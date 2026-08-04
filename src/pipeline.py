from analyzer import ProjectAnalyzer
from ai_detector import AIDetector
from arm_advisor import ArmAdvisor
from benchmark import Benchmark
from optimizer import ModelOptimizer
from comparator import BenchmarkComparator
from runtime_tuner import RuntimeTuner


class OptimizationPipeline:

    def __init__(self, project_path, target_device="Generic ARM Cortex"):

        self.project_path = project_path
        self.target_device = target_device

    def run(self):

        results = {}

        # =====================================
        # PROJECT ANALYSIS
        # =====================================

        analyzer = ProjectAnalyzer(self.project_path)

        project_name = analyzer.project_name()
        language = analyzer.detect_language()

        results["project"] = project_name
        results["language"] = language
        results["target_device"] = self.target_device

        # =====================================
        # AI MODEL DETECTION
        # =====================================

        detector = AIDetector(self.project_path)

        models = detector.detect_models()

        if not models:

            results["status"] = "No AI Models Found"

            return results

        results["status"] = "Success"

        model_reports = []

        # =====================================
        # PROCESS EACH MODEL
        # =====================================

        for model in models:

            model_report = {}

            model_report["model_name"] = model["model_name"]
            model_report["model_type"] = model["model_type"]
            model_report["framework"] = model["framework"]

            model_path = model["model_path"]

            # =====================================
            # THREAD TUNING
            # =====================================

            tuner = RuntimeTuner(model_path)

            best_threads, thread_results = tuner.find_best_threads()

            model_report["best_threads"] = best_threads
            model_report["thread_results"] = thread_results

            # =====================================
            # ARM RECOMMENDATION
            # =====================================

            advisor = ArmAdvisor(model["model_type"])

            arm_plan = advisor.generate_plan()

            model_report["arm_plan"] = arm_plan

            # =====================================
            # ORIGINAL BENCHMARK
            # =====================================

            benchmark = Benchmark(model_path)

            before = benchmark.benchmark()

            model_report["before"] = before

            # =====================================
            # OPTIMIZATION
            # =====================================

            optimizer = ModelOptimizer(model_path)

            optimized_model = optimizer.quantize()

            model_report["optimized_model"] = optimized_model

            # =====================================
            # OPTIMIZED BENCHMARK
            # =====================================

            benchmark2 = Benchmark(optimized_model)

            after = benchmark2.benchmark()

            model_report["after"] = after

            # =====================================
            # COMPARISON
            # =====================================

            comparison = BenchmarkComparator().compare(
                before,
                after
            )

            model_report["comparison"] = comparison

            model_reports.append(model_report)

        results["models"] = model_reports

        return results