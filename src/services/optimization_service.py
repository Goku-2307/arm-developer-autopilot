from src.analysis.analyzer import ProjectAnalyzer
from src.analysis.ai_detector import AIDetector

from src.optimization.optimization_engine import OptimizationEngine

from src.core.session_manager import SessionManager


class OptimizationService:

    def __init__(self):

        self.manager = SessionManager()

        self.session = self.manager.get_session()

        self.logger = self.manager.get_logger()

    def optimize_project(self, project_path):

        # -------------------------
        # Analyze Project
        # -------------------------

        self.logger.start()

        analyzer = ProjectAnalyzer(project_path)

        self.session.project_name = analyzer.project_name()
        self.session.project_path = project_path
        self.session.language = analyzer.detect_language()

        self.logger.log(
            "Analysis",
            "Project Analysis Complete",
            "SUCCESS"
        )

        # -------------------------
        # Detect Models
        # -------------------------

        self.logger.start()

        detector = AIDetector(project_path)

        models = detector.detect_models()

        if not models:

            self.logger.log(
                "Detection",
                "No AI Models Found",
                "ERROR"
            )

            return self.session

        self.session.models = models

        self.logger.log(
            "Detection",
            f"{len(models)} Model(s) Found",
            "SUCCESS"
        )

        # -------------------------
        # Optimize Every Model
        # -------------------------

        self.session.benchmark_results = []

        for model in models:

            self.logger.start()

            engine = OptimizationEngine(
                model["model_path"]
            )

            result = engine.optimize()

            if result:

                self.session.best_result = result["best_configuration"]

                self.session.benchmark_results.extend(
                    result["ranking"]
                )

            self.logger.log(
                "Optimization",
                f"{model['model_name']} Optimized",
                "SUCCESS"
            )

        return self.session