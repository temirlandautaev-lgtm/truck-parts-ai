from __future__ import annotations

import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from search import Part, rank_parts

MODEL = "text-embedding-3-small"
CATALOG_PATH = Path(__file__).with_name("catalog.csv")


def load_catalog(path: Path = CATALOG_PATH) -> list[Part]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"article", "name", "brand", "vehicle"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"В catalog.csv не хватает колонок: {', '.join(sorted(missing))}")

        return [
            Part(
                article=row["article"].strip(),
                name=row["name"].strip(),
                brand=row["brand"].strip(),
                vehicle=row["vehicle"].strip(),
            )
            for row in reader
            if any((row["article"], row["name"], row["brand"], row["vehicle"]))
        ]


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=MODEL, input=texts)
    return [item.embedding for item in response.data]


def main() -> None:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Не найден OPENAI_API_KEY. Скопируйте .env.example в .env и добавьте API-ключ.")
        return

    parts = load_catalog()
    if not parts:
        print("Каталог пуст.")
        return

    client = OpenAI()

    print("Truck Parts AI Finder")
    print("Для выхода напишите: exit")
    print("Индексирую каталог...")

    part_embeddings = embed_texts(client, [part.searchable_text() for part in parts])

    while True:
        query = input("\nВведите запрос: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "выход"}:
            break

        query_embedding = embed_texts(client, [query])[0]
        results = rank_parts(query_embedding, parts, part_embeddings, limit=5)

        print("\nЛучшие совпадения:")
        for number, (part, score) in enumerate(results, start=1):
            print(
                f"{number}. {part.name}\n"
                f"   Артикул: {part.article}\n"
                f"   Бренд: {part.brand}\n"
                f"   Авто: {part.vehicle}\n"
                f"   Совпадение: {score:.3f}"
            )


if __name__ == "__main__":
    main()
