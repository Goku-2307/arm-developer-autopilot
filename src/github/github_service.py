import base64
import os
import subprocess
from pathlib import Path
from typing import Optional

import requests


class GitHubService:
    """GitHub API + local git operations. Tokens are never written to git config."""

    def __init__(self, token: Optional[str] = None, cwd: Optional[str] = None):
        self.token = token.strip() if token else ""
        self.cwd = Path(cwd or os.getcwd()).resolve()

    def _git(self, args):
        env = os.environ.copy()
        if self.token:
            encoded = base64.b64encode(f"x-access-token:{self.token}".encode()).decode()
            env["GIT_CONFIG_COUNT"] = "1"
            env["GIT_CONFIG_KEY_0"] = "http.extraHeader"
            env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {encoded}"
        result = subprocess.run(
            ["git", *args], cwd=str(self.cwd), env=env,
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    def status(self):
        return self._git(["status", "--short", "--branch"])

    def init(self):
        return self._git(["init"])

    def add(self):
        return self._git(["add", "."])

    def commit(self, message):
        return self._git(["commit", "-m", message])

    def remote_url(self):
        code, out, err = self._git(["remote", "get-url", "origin"])
        return out if code == 0 else ""

    def set_remote(self, url):
        existing = self.remote_url()
        if existing:
            return self._git(["remote", "set-url", "origin", url])
        return self._git(["remote", "add", "origin", url])

    def push(self, branch="master"):
        return self._git(["push", "-u", "origin", branch])

    def create_or_get_repo(self, owner, name, private=False, description="ARM Developer AutoPilot optimization project"):
        if not self.token:
            return False, "GitHub token is required.", None

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        base = "https://api.github.com"
        response = requests.get(f"{base}/repos/{owner}/{name}", headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return True, "Repository already exists.", data

        if response.status_code != 404:
            return False, self._github_error(response), None

        response = requests.post(
            f"{base}/user/repos",
            headers=headers,
            json={"name": name, "private": bool(private), "description": description, "auto_init": False},
            timeout=30,
        )
        if response.status_code in (201, 200):
            return True, "Repository created.", response.json()
        return False, self._github_error(response), None

    @staticmethod
    def _github_error(response):
        try:
            data = response.json()
            return f"GitHub API error {response.status_code}: {data.get('message', response.text)}"
        except Exception:
            return f"GitHub API error {response.status_code}: {response.text}"
