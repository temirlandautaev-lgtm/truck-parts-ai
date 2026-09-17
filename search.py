from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Part:
    article: str
    name: str
    brand: str
    vehicle: str

    def searchable_text(self) -> str:
        return f"{self.article} | {self.name} | {self.brand} | {self.vehicle}"


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def rank_parts(
    query_embedding: Sequence[float],
    parts: Sequence[Part],
    part_embeddings: Sequence[Sequence[float]],
    limit: int = 5,
) -> list[tuple[Part, float]]:
    if len(parts) != len(part_embeddings):
        raise ValueError("Each part must have exactly one embedding.")
    if limit <= 0:
        return []

    scored = [
        (part, cosine_similarity(query_embedding, embedding))
        for part, embedding in zip(parts, part_embeddings)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]
