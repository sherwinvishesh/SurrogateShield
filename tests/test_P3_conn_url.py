"""Campaign phase 3: a database or server URL ("postgresql://user:pw@host")
gives its password a credential surrogate and a deployment's own host a
hostname surrogate that keeps the domain; a placeholder password, a public
or local host and an IP host (left to the IP pattern) do not. Model-free;
hosts and passwords are invented."""

import pytest

from surrogateshield.core.detection import pattern_scan as ps
from surrogateshield.core.generation.mimic import MimicGen


def found(text):
    return [(e.type, e.text) for e in ps.scan(text)]


@pytest.mark.parametrize("text, want", [
    ("DB=postgresql://app_user:Kq7vLm29x@orders-db-3.c9xk2a.eu-west-1.rds.amazonaws.com:5432/orders",
     [("credential", "Kq7vLm29x"), ("hostname", "orders-db-3.c9xk2a.eu-west-1.rds.amazonaws.com")]),
    ("mongodb+srv://svc:<password>@cluster7.ab1cd.mongodb.net/test?retryWrites=true",
     [("hostname", "cluster7.ab1cd.mongodb.net")]),
    ("uri: postgres://svc:...@db.billing.prod.kessmar.io/main", [("hostname", "db.billing.prod.kessmar.io")]),
    ("redis://:Zt8wQ2pLx@cache-4.east.kessmar.net:6380/0",
     [("credential", "Zt8wQ2pLx"), ("hostname", "cache-4.east.kessmar.net")]),
])
def test_the_password_and_a_deployments_host_are_claimed(text, want):
    assert found(text) == want


@pytest.mark.parametrize("text", [
    "redis://localhost:6379/0",
    "mysql://root:root@localhost:3306/shop",
    "postgres://user:password@db.example.com:5432/x",
    "postgresql://u:${DB_PASS}@db:5432/app",
    "sftp://deploy@files.example.org/in",
    "ftp://ftp.gnu.org/gnu/",
    "smtp is smtp.gmail.com and ldap://ldap.kessmar.io",
])
def test_placeholders_and_public_hosts_are_not(text):
    assert found(text) == []


def test_an_ip_host_is_left_to_the_ip_pattern():
    assert found("mysql://root:Tr0ub4dor@203.0.113.45:3306/shop") == [
        ("credential", "Tr0ub4dor"), ("ip_address", "203.0.113.45")]


def test_nothing_else_is_matched_inside():
    # the user part before "@" is no e-mail local part
    assert found("postgres://ops:s3cretpw9@db7.kessmar.io:5432/x") == [
        ("credential", "s3cretpw9"), ("hostname", "db7.kessmar.io")]


@pytest.mark.parametrize("host, kept", [
    ("orders-db-3.c9xk2a.eu-west-1.rds.amazonaws.com", "amazonaws.com"),
    ("cluster7.ab1cd.mongodb.net", "mongodb.net"),
    ("db7.billing.kessmar.co.uk", "kessmar.co.uk"),
])
def test_the_host_surrogate_keeps_the_domain_and_replaces_the_rest(host, kept):
    out = MimicGen(seed=7)._gen_hostname_like(host)
    assert out.endswith("." + kept)
    head, new_head = host[:-len(kept) - 1], out[:-len(kept) - 1]
    assert new_head != head and len(new_head) == len(head)
    assert [c.isdigit() for c in new_head] == [c.isdigit() for c in head]
    assert host.split(".")[0] not in out


def test_a_device_name_still_follows_its_owner():
    out = MimicGen(seed=7)._gen_hostname_like("Galaxy-S23-Ritika")
    assert out.startswith("Galaxy-S23-") and not out.endswith("Ritika")
