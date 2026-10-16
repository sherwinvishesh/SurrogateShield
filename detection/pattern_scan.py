# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""Alias of ``surrogateshield.core.detection.pattern_scan`` (single implementation, audit F4).

The app imports this path; the code lives in the package. Importing this
module returns the package module itself, so patches apply to both.
"""

import sys

from surrogateshield.core.detection import pattern_scan as _impl

sys.modules[__name__] = _impl
