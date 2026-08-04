import os


class ProjectAnalyzer:

    def __init__(self, project_path):
        self.project_path = project_path

    def project_name(self):
        return os.path.basename(self.project_path)

    def total_files(self):
        count = 0
        for _, _, files in os.walk(self.project_path):
            count += len(files)
        return count

    def total_folders(self):
        count = 0
        for _, dirs, _ in os.walk(self.project_path):
            count += len(dirs)
        return count

    def folder_size(self):
        total = 0

        for path, _, files in os.walk(self.project_path):
            for file in files:
                filepath = os.path.join(path, file)

                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)

        return round(total / (1024 * 1024), 2)

    def detect_language(self):

        files = os.listdir(self.project_path)

        if "requirements.txt" in files:
            return "Python"

        if "package.json" in files:
            return "Node.js"

        if "pom.xml" in files:
            return "Java"

        return "Unknown"