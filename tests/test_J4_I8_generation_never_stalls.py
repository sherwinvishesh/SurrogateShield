"""Gates J4 / J2 with audit I8 / I12 — two shapes from the blind dev3 split
crashed generate_all ("could not generate a surrogate … that differs from
every original"), aborting the whole message:

* "Failed password for invalid user admin": the auth-log pattern had no
  validator, so the default account "admin" became a handle, and a handle
  surrogate of a role name is the role name itself (it is kept for e-mail
  locals such as admin@);
* "Mr. & Mrs. Castellano": NER typed "&" as a PERSON, and a letterless
  value has no surrogate other than itself.
Model-free.
"""

import pytest

from surrogateshield.core.detection import pattern_scan as ps
from surrogateshield.core.detection import relation_gate as rg
from surrogateshield.core.entities import DetectedEntity
from surrogateshield.core.generation.mimic import MimicGen

LOG = ("Sep 30 02:14:51 web01 sshd[22817]: Failed password for invalid user {u} "
       "from 185.244.29.17 port 51122 ssh2")


@pytest.mark.parametrize("account", ["admin", "root", "ubuntu", "www-data", "git", "oracle"])
def test_I8_default_accounts_in_auth_logs_are_not_handles(account):
    assert not any(e.type == "handle" for e in ps.scan(LOG.format(u=account)))


def test_I8_real_user_in_auth_log_is_still_a_handle():
    assert ("handle", "jmorales") in [(e.type, e.text) for e in ps.scan(LOG.format(u="jmorales"))]


@pytest.mark.parametrize("value", ["&", "-", "+", "&&", "1/2", "'"])
def test_I12_letterless_names_are_junk(value):
    text = f"Our landlords (Mr. {value} Mrs. Castellano) want more rent"
    s = text.index(value)
    assert rg.is_junk(DetectedEntity(value, s, s + len(value), "PERSON", 0.9, "ner"), text)


@pytest.mark.parametrize("handle", ["admin", "@support", "info", "@security"])
def test_J4_role_handle_gets_a_surrogate_that_differs(handle):
    for seed in range(20):
        ent = DetectedEntity(handle, 0, len(handle), "handle", 1.0, "pattern")
        out = MimicGen(seed=seed).generate_all([ent], text=f"ping {handle} today")
        assert out[handle].lower() != handle.lower()
        assert out[handle].startswith("@") == handle.startswith("@")


def test_J4_role_email_local_is_still_kept():
    ent = DetectedEntity("admin@harborpoint.com", 0, 21, "email", 1.0, "pattern")
    assert MimicGen(seed=1).generate(ent).startswith("admin@")
