from pathlib import Path


class ProjectAnalyzer:
    """
    Analyzes the selected AI project.
    """

    def __init__(self, project_path):

        self.project_path = Path(project_path)

    def project_name(self):

        return self.project_path.name

    def detect_language(self):

        extensions = []

        for file in self.project_path.rglob("*"):

            if file.is_file():

                extensions.append(file.suffix.lower())

        if ".py" in extensions:
            return "Python"

        if ".cpp" in extensions:
            return "C++"

        if ".c" in extensions:
            return "C"

        if ".java" in extensions:
            return "Java"

        return "Unknown"

    def project_summary(self):

        return {

            "project_name": self.project_name(),

            "language": self.detect_language(),

            "path": str(self.project_path)

        }