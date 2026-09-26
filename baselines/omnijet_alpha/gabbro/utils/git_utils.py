"""Useful functions for git operations.

(e.g. getting git hash, last commit message, etc.)

The helpers return ``NOT_A_GIT_REPO`` instead of raising when the code is not inside
a git repository (e.g. a downloaded archive) or git is not installed, so that
training can still start.
"""

import subprocess  # nosec

NOT_A_GIT_REPO = "<not-a-git-repo>"


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")  # nosec
    except (subprocess.CalledProcessError, FileNotFoundError):
        return NOT_A_GIT_REPO


def get_git_status():
    try:
        cmd = "git diff -- . ':!*.ipynb' --color"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)  # nosec
        stdout, stderr = process.communicate()
        git_diff_output = stdout.decode("utf-8")
    except Exception:
        git_diff_output = NOT_A_GIT_REPO
    separator_start = f"\n{100 * '='}\n{'=' * 10} start git diff {'=' * 10}\n"
    separator_end = f"\n{'=' * 10} end git diff {'=' * 10}\n{100 * '='}\n"
    return separator_start + git_diff_output + separator_end


def get_last_commit_message():
    try:
        return (
            subprocess.check_output(  # nosec
                ["git", "log", "-1", "--pretty=%B"],
            )
            .strip()
            .decode("utf-8")
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return NOT_A_GIT_REPO
