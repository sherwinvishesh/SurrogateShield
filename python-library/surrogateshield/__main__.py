"""``python -m surrogateshield`` — the same as the ``surrogateshield`` command."""

import sys

from .cli import main

sys.exit(main())
