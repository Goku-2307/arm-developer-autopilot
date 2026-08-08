from src.reports.report_generator import ReportGenerator


class ReportService:
    def __init__(self, output_dir="reports"):
        self.generator = ReportGenerator(output_dir)

    def generate(self, session):
        return self.generator.generate(session)
