"""Linux can't run a shell script with Windows line endings: bash reads
`set -euo pipefail\\r` and fails on an "invalid option name". This broke the first
Docker build, and it would break CI's vendoring step the same way."""
from pathlib import Path

import pytest


@pytest.mark.parametrize("path", sorted(Path("scripts").glob("*.sh")), ids=lambda p: p.name)
def test_shell_scripts_use_unix_line_endings(path):
    assert b"\r" not in path.read_bytes(), f"{path} has Windows line endings"
