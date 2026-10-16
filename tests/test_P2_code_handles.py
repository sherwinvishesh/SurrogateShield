"""Campaign phase 2, N18 and N19: a handle pattern's value joined to code
(a call, an index, an assignment, a Vue event, a doc-comment tag, or a
platform word that runs on into a longer identifier) is not a handle; a
bare host inside a template expression or a call argument is a property
path, not a domain. Scan only; the names and texts are invented."""

import pytest

from surrogateshield.core.detection import pattern_scan as ps


def _handles(text):
    return [e.text for e in ps.scan(text) if e.type == "handle"]


def _urls(text):
    return [(text[s:e], personal) for s, e, personal in ps.find_urls(text)]


JSDOC = ("/**\n"
         " * @author Ana Ruiz <ana@example.org>\n"
         " * @phone +33 6 12 34 56 78\n"
         " * @site https://example.org\n"
         " */")


# ── N18: code is not a handle ────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    '<button @click="openMenu(item)">',              # a Vue event
    '<input @input="onType" @change="save">',
    'username = profile[x["name"]]',                 # an index after the value
    'handle=Login.get("x")',                         # a call after the value
    'alias gst="git status"',                        # a shell alias
    'take a snapshot_id of it',                      # "snap" runs on
    'instance_id = 4',                               # "insta" runs on
    'the redditor_99 replied',                       # "reddit" runs on
    'aliases_v2 = {}',                               # "alias" runs on
    'loginapi.get("x")',                             # "login" runs on
    JSDOC,                                           # doc-comment tags
    "/* legacy\n * @phone 555 0100\n */",
])
def test_code_is_not_a_handle(text):
    assert _handles(text) == []


@pytest.mark.parametrize("text, handles", [
    ("ping @tariq (the tall one) and @mira_k", ["@tariq", "@mira_k"]),
    ("* @tariq please check the draft", ["@tariq"]),             # a Markdown bullet
    ("files in src/* and\n* @tariq please check", ["@tariq"]),   # a glob is no comment
    ("/* old */ ping @tariq", ["@tariq"]),                       # a closed comment
    ("login name: nightowl_42", ["nightowl_42"]),
    ('{"handle": "@nightowl_42"}', ["@nightowl_42"]),
    ("my discord is nightowl_42", ["nightowl_42"]),
    ("my steam name is PixelMoth_77", ["PixelMoth_77"]),
    ("snap: nightowl_42", ["nightowl_42"]),                       # a separator after the keyword
])
def test_a_person_s_handle_still_is(text, handles):
    assert _handles(text) == handles


def test_a_doc_comment_is_still_an_address_and_a_number():
    types = {e.type for e in ps.scan(JSDOC)}
    assert "handle" not in types and "email" in types and any(t.startswith("phone") for t in types)


# ── N19: a property path is not a domain ─────────────────────────────────────

@pytest.mark.parametrize("text", [
    "`${user.name} ${user.surname}`",
    "$t('menu.home.link')",
    "{{ item.info || '-' }}",
    "title={user.name}",
    'x = obj["user.name"]',
])
def test_a_property_path_in_a_template_is_not_a_url(text):
    assert _urls(text) == []


@pytest.mark.parametrize("text, urls", [
    ("my site anaruiz.dev", [("anaruiz.dev", True)]),
    ("my site is anaruiz.dev.", [("anaruiz.dev", True)]),
    ("reach me at anaruiz.dev or ana.page", [("anaruiz.dev", False), ("ana.page", False)]),
    ("see `${base}/https://example.org/a` now", [("https://example.org/a", False)]),   # a full URL stays
    ("href={'https://example.org/a'}", [("https://example.org/a", False)]),
    ("{{ example.org/path }}", [("example.org/path", False)]),                          # a path stays
])
def test_a_domain_in_prose_or_a_full_url_still_is(text, urls):
    assert _urls(text) == urls
