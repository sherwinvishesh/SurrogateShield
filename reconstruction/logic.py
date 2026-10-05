# Paper available on arXiv: https://arxiv.org/abs/2606.29567

"""ResolvePass for the app: ``surrogateshield.core.reconstruction.resolve``
with the fuzzy threshold taken from ``config.py`` (audit F4)."""

from typing import Dict, Optional

import config as _config
from surrogateshield.core.reconstruction import resolve as _impl
from surrogateshield.core.reconstruction.resolve import ResolutionFailure  # noqa: F401


class ResolvePass(_impl.ResolvePass):
    def resolve(
        self,
        response_text: str,
        shadow_map: Dict[str, str],
        fuzzy_threshold: Optional[int] = None,
    ) -> str:
        if fuzzy_threshold is None:
            fuzzy_threshold = _config.FUZZY_MATCH_THRESHOLD
        return super().resolve(response_text, shadow_map, fuzzy_threshold)


def __getattr__(name):
    return getattr(_impl, name)
