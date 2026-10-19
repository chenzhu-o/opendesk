"""Suite-wide test isolation.

* ``OPENDESK_HOME`` points at a throwaway directory so the safety guard's
  policy / control / audit files never touch the real ``~/.opendesk``.
* ``OPENDESK_APPROVAL_CHAIN=none`` makes any ``confirm`` decision deny
  immediately instead of popping a real OS dialog or reading stdin.

Both are ``setdefault`` so a developer can override them from the shell.
"""

from __future__ import annotations

import os
import shutil
import tempfile

_TEST_HOME = tempfile.mkdtemp(prefix="opendesk-test-home-")
os.environ.setdefault("OPENDESK_HOME", _TEST_HOME)
os.environ.setdefault("OPENDESK_APPROVAL_CHAIN", "none")


def pytest_unconfigure(config):  # noqa: ARG001
    shutil.rmtree(_TEST_HOME, ignore_errors=True)
