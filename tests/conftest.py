"""Test setup for the library's own suite. Also usable from the sdist: it
needs no file outside this folder.

Every test gets a private SURROGATESHIELD_HOME, so none reads or writes the
user's real ~/.surrogateshield; and config() changes only what it is given,
so the default settings are restored after every test."""

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

LIB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LIB_ROOT))

if not os.environ.get("SURROGATESHIELD_TEST_HOME_SET"):
    _home = tempfile.mkdtemp(prefix="ss-test-home-")
    os.environ["SURROGATESHIELD_HOME"] = _home
    os.environ["SURROGATESHIELD_TEST_HOME_SET"] = "1"
    atexit.register(shutil.rmtree, _home, ignore_errors=True)


@pytest.fixture(autouse=True)
def _library_defaults():
    yield
    if "surrogateshield" in sys.modules:
        import surrogateshield as ss
        from surrogateshield._state import Config, cfg
        ss.flush()
        for name, value in vars(Config()).items():
            setattr(cfg, name, value)
