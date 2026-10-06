"""
An example detector plugin (V3 §3.6): employee numbers such as
``EMP-204816``, reported as ID, also when typed with look-alike characters
("ＥＭＰ-２０４８１６"): the plugin reads the message and its canonical view.

Register it and add it to a config::

    import surrogateshield as ss
    from employee_id_detector import factory

    ss.register_detector("employee_ids", factory)
    ss.config(detection=ss.preset("balanced").with_plugin(
        "employee_ids", types=["ID"], options={"prefix": "EMP"}))
    ss.mask("Ask EMP-204816 about the rota")

A package can declare it as an entry point instead, and list the stage in
its config file (``SURROGATESHIELD_DETECTION_CONFIG``)::

    [project.entry-points."surrogateshield.detectors"]
    employee_ids = "employee_id_detector:factory"
"""

from __future__ import annotations

import re
from typing import List

from surrogateshield import Candidate


class EmployeeIds:
    """``<prefix>`` then six digits, joined by a hyphen, a space or nothing."""

    def __init__(self, prefix: str = "EMP", min_score: float = 0.0):
        self.pattern = re.compile(rf"\b{re.escape(prefix)}[- ]?\d{{6}}\b", re.IGNORECASE)
        self.min_score = min_score

    def detect(self, text: str, view) -> List[Candidate]:
        found = {(m.start(), m.end()) for m in self.pattern.finditer(text)}
        if view is not None:            # the canonical view: look-alikes folded
            found |= {view.original(m.start(), m.end()) for m in self.pattern.finditer(view.text)}
        return [Candidate(s, e, "ID", 0.99) for s, e in sorted(found)]


def factory(stage) -> EmployeeIds:
    """The registry calls this with the config's Stage."""
    return EmployeeIds(prefix=stage.options.get("prefix", "EMP"))
