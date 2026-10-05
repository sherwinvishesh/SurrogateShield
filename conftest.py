"""Repo-wide test setup, loaded before any test module.

Every test (and every legacy script run as a subprocess) gets a private
SURROGATESHIELD_HOME, so no test reads or writes the user's real
~/.surrogateshield (device key, conversations, settings).
"""

import atexit
import os
import shutil
import tempfile

if not os.environ.get("SURROGATESHIELD_TEST_HOME_SET"):
    _home = tempfile.mkdtemp(prefix="ss-test-home-")
    os.environ["SURROGATESHIELD_HOME"] = _home
    os.environ["SURROGATESHIELD_TEST_HOME_SET"] = "1"
    atexit.register(shutil.rmtree, _home, ignore_errors=True)


import sys

import pytest


@pytest.fixture(autouse=True)
def _library_defaults():
    """config() changes only what it is given, so restore the library's
    default settings and drop the context's session after every test."""
    yield
    if "surrogateshield" in sys.modules:
        import surrogateshield as ss
        from surrogateshield._state import Config, cfg
        ss.flush()
        for name, value in vars(Config()).items():
            setattr(cfg, name, value)
