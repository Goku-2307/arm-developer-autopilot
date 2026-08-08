from pathlib import Path
from urllib.parse import urlparse

from src.github.github_service import GitHubService
from src.github.readme_generator import ReadmeGenerator


class ProjectPublisher:
    """Publishes the AutoPilot project directory to a GitHub repository."""

    def publish(self, session, token, owner, repo_name, private=False, project_path=None):
        project_dir = Path(project_path or Path.cwd()).resolve()
        if not project_dir.exists():
            return {"success": False, "message": f"Project directory not found: {project_dir}"}

        git = GitHubService(token=token, cwd=str(project_dir))

        code, _, err = git.init()
        if code != 0:
            return {"success": False, "message": f"git init failed: {err}"}

        ok, message, repo = git.create_or_get_repo(owner, repo_name, private=private)
        if not ok:
            return {"success": False, "message": message}

        clone_url = repo.get("clone_url") or f"https://github.com/{owner}/{repo_name}.git"
        code, _, err = git.set_remote(clone_url)
        if code != 0:
            return {"success": False, "message": f"Could not configure origin: {err}"}

        ReadmeGenerator().generate(session, str(project_dir / "README_GENERATED.md"))

        code, _, err = git.add()
        if code != 0:
            return {"success": False, "message": f"git add failed: {err}"}

        code, out, err = git.commit("Publish ARM optimization results")
        if code != 0 and "nothing to commit" not in (out + err).lower():
            return {"success": False, "message": f"git commit failed: {err or out}"}

        branch_code, branch, branch_err = git._git(["branch", "--show-current"])
        branch = branch if branch_code == 0 and branch else "master"

        code, out, err = git.push(branch)
        if code != 0:
            return {
                "success": False,
                "message": (
                    f"Push failed: {err or out}. "
                    "Check that the token has permission to push and that the repository is accessible."
                ),
            }

        session.github_repo = repo.get("html_url", f"https://github.com/{owner}/{repo_name}")
        return {
            "success": True,
            "message": "Project published successfully.",
            "url": session.github_repo,
            "commit": "Publish ARM optimization results",
        }
