"""EntityTrace maps the spaCy model's word-vector table from its file instead of
reading it into memory (en_core_web_lg: 411 MB, most of it never read), which
keeps the default pipeline inside the V3 peak-RSS budget. The values are the
file's, so entities do not change. Model-free except the last test (heavy)."""

import numpy
import pytest

spacy = pytest.importorskip("spacy")

from surrogateshield.core.detection import entity_trace as et  # noqa: E402

WORDS = ("apple", "pear", "plum")


@pytest.fixture
def model_dir(tmp_path):
    nlp = spacy.blank("en")
    for i, w in enumerate(WORDS):
        nlp.vocab.set_vector(w, numpy.arange(4, dtype="float32") + i)
    nlp.to_disk(tmp_path / "with_vectors")
    return tmp_path / "with_vectors"


def test_vectors_are_mapped_read_only_with_the_files_values(model_dir):
    nlp = et._load_nlp(str(model_dir))
    data = nlp.vocab.vectors.data
    assert isinstance(data, numpy.memmap)
    assert not data.flags.writeable
    for i, w in enumerate(WORDS):
        assert nlp.vocab[w].vector.tolist() == (numpy.arange(4) + i).tolist()
    whole = spacy.load(str(model_dir))
    assert numpy.array_equal(numpy.asarray(data), whole.vocab.vectors.data)
    assert nlp.vocab.vectors.key2row == whole.vocab.vectors.key2row


def test_model_without_vectors_loads(tmp_path):
    spacy.blank("en").to_disk(tmp_path / "plain")
    nlp = et._load_nlp(str(tmp_path / "plain"))
    assert nlp.vocab.vectors.shape == (0, 0)


def test_table_that_cannot_be_mapped_is_read_whole(model_dir, monkeypatch):
    def fail(_nlp):
        raise ValueError("no map")
    monkeypatch.setattr(et, "_map_vectors", fail)
    nlp = et._load_nlp(str(model_dir))
    assert not isinstance(nlp.vocab.vectors.data, numpy.memmap)
    assert nlp.vocab["plum"].vector.tolist() == [2.0, 3.0, 4.0, 5.0]


@pytest.mark.heavy
@pytest.mark.skipif(not spacy.util.is_package("en_core_web_lg"), reason="en_core_web_lg not installed")
def test_lg_entities_are_the_same_as_with_the_table_in_memory():
    texts = ["Alice Johnson moved from Springfield to Toronto and joined Initech last May.",
             "Dr. Priya Raman of the Mayo Clinic flew to Lagos via Heathrow.",
             "my landlord kwame asante says the flat on Rue Oberkampf is ready"]
    mapped = et._load_nlp("en_core_web_lg")
    assert isinstance(mapped.vocab.vectors.data, numpy.memmap)
    whole = spacy.load("en_core_web_lg")
    ents = lambda nlp, t: [(e.start_char, e.end_char, e.label_) for e in nlp(t).ents]
    for t in texts:
        assert ents(mapped, t) == ents(whole, t)
