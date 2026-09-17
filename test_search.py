import pytest

from search import Part, cosine_similarity, rank_parts


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_rank_parts_returns_best_match_first():
    parts = [
        Part("A1", "Brake disc", "BrandA", "Volvo FH"),
        Part("B1", "Fuel filter", "BrandB", "MAN TGX"),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    results = rank_parts([0.9, 0.1], parts, embeddings, limit=2)

    assert results[0][0].article == "A1"
    assert results[0][1] > results[1][1]


def test_rank_parts_rejects_mismatched_lengths():
    parts = [Part("A1", "Brake disc", "BrandA", "Volvo FH")]

    with pytest.raises(ValueError):
        rank_parts([1.0, 0.0], parts, [], limit=5)
