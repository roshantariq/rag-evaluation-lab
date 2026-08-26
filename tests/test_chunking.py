"""Tests for chunking.

The span invariant - text[char_start:char_end] == chunk.text - is the
foundation the entire gold evaluation set rests on. If it breaks, every
retrieval metric is silently wrong.
"""

import random

from rageval.chunking.base import Chunk, chunk_document, fixed_size_chunks

# Deterministic word-count tokenizer so tests never touch the network.
count = lambda s: int(len(s.split()) * 1.35)  # noqa: E731


class TestSpanInvariant:
    def test_spans_reproduce_text_across_random_documents(self):
        random.seed(7)
        vocab = "convlstm radar nowcasting precipitation ERA5 transformer skill RMSE".split()
        for _ in range(100):
            text = " ".join(random.choice(vocab) for _ in range(random.randint(1, 600)))
            for target in (16, 64, 512):
                for overlap in (0, 8):
                    for a, b in fixed_size_chunks(text, target, overlap, count):
                        assert text[a:b]
                        assert text[a:b].strip() == text[a:b]

    def test_chunks_without_overlap_tile_the_document(self):
        text = " ".join(f"w{i}" for i in range(200))
        spans = list(fixed_size_chunks(text, 20, 0, count))
        assert " ".join(text[a:b] for a, b in spans) == text


class TestChunkerEdgeCases:
    def test_empty_text(self):
        assert list(fixed_size_chunks("", 512, 0, count)) == []

    def test_whitespace_only(self):
        assert list(fixed_size_chunks("   \n  ", 512, 0, count)) == []

    def test_text_shorter_than_target(self):
        spans = list(fixed_size_chunks("single", 512, 0, count))
        assert len(spans) == 1

    def test_single_token_larger_than_target_terminates(self):
        # Guards against an infinite loop when one word exceeds the budget.
        assert len(list(fixed_size_chunks("x" * 5000, 10, 0, count))) == 1

    def test_overlap_produces_more_chunks_and_terminates(self):
        text = " ".join(f"w{i}" for i in range(200))
        plain = list(fixed_size_chunks(text, 20, 0, count))
        overlapped = list(fixed_size_chunks(text, 20, 8, count))
        assert len(overlapped) > len(plain)
        assert len(overlapped) < 500  # would run away if the back-step were wrong
        assert overlapped[1][0] < overlapped[0][1]


class TestChunkProvenance:
    CHUNK = Chunk(arxiv_id="1506.04214v2", text="t", char_start=100, char_end=200,
                  section="3 Model", strategy="fixed_512")

    def test_chunk_id_is_span_derived(self):
        assert self.CHUNK.chunk_id == "1506.04214v2:100-200"

    def test_evidence_inside_chunk_overlaps(self):
        assert self.CHUNK.overlaps(150, 160)

    def test_evidence_outside_chunk_does_not(self):
        assert not self.CHUNK.overlaps(250, 260)

    def test_touching_boundary_is_not_overlap(self):
        # Half-open spans: [100,200) and [200,210) share no characters.
        assert not self.CHUNK.overlaps(200, 210)

    def test_straddling_evidence_overlaps(self):
        assert self.CHUNK.overlaps(90, 110)

    def test_overlap_fraction_is_share_of_evidence_covered(self):
        assert self.CHUNK.overlap_fraction(150, 250) == 0.5


class TestChunkDocument:
    DOC = {
        "arxiv_id": "1506.04214v2",
        "title": "Convolutional LSTM Network",
        "published": "2015-06-13",
        "text": "The ConvLSTM extends the fully connected LSTM with convolutions. " * 60,
        "sections": [],
    }

    def test_every_chunk_span_round_trips(self):
        chunks = chunk_document(self.DOC, target_tokens=64, count_tokens=count)
        assert chunks
        for c in chunks:
            assert self.DOC["text"][c.char_start:c.char_end] == c.text

    def test_chunk_ids_are_unique(self):
        chunks = chunk_document(self.DOC, target_tokens=64, count_tokens=count)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_metadata_propagates(self):
        c = chunk_document(self.DOC, target_tokens=64, count_tokens=count)[0]
        assert c.title == "Convolutional LSTM Network"
        assert c.strategy == "fixed_512"