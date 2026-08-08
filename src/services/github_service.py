from src.github.publisher import ProjectPublisher


class GitHubService:
    def publish(self, session, token, owner, repo_name, private=False, project_path=None):
        return ProjectPublisher().publish(
            session=session,
            token=token,
            owner=owner,
            repo_name=repo_name,
            private=private,
            project_path=project_path,
        )
