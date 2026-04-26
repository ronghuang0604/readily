import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 10


def load_records(records_path: Path):
    records = []

    with records_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {records_path} on line {line_number}: {exc}"
                ) from exc

    return records


def load_data():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    embeddings_path = project_root / "data" / "processed" / "embeddings.npy"
    records_path = project_root / "data" / "processed" / "embedded_pages.jsonl"

    embeddings = np.load(embeddings_path)
    records = load_records(records_path)

    if len(embeddings) != len(records):
        raise ValueError(
            "Embeddings and page records are misaligned: "
            f"{len(embeddings)} embeddings vs {len(records)} records."
        )

    return embeddings, records


def embed_question(question: str, model: SentenceTransformer):
    embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding


def retrieve_top_pages(question: str, embeddings, records, model: SentenceTransformer):
    question_embedding = embed_question(question, model)
    scores = embeddings @ question_embedding
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []
    for rank, index in enumerate(top_indices, start=1):
        record = records[index]
        results.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "file_name": record.get("file_name", ""),
                "page_number": record.get("page_number", ""),
                "clean_text": record.get("clean_text", ""),
            }
        )

    return results


def print_results(results):
    for result in results:
        preview = result["clean_text"][:700]
        print(f"rank: {result['rank']}")
        print(f"similarity score: {result['score']:.4f}")
        print(f"file_name: {result['file_name']}")
        print(f"page_number: {result['page_number']}")
        print(f"text: {preview}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve the most relevant policy pages for a user question."
    )
    parser.add_argument("question", help="User question to search for.")
    args = parser.parse_args()

    embeddings, records = load_data()
    model = SentenceTransformer(MODEL_NAME)
    results = retrieve_top_pages(args.question, embeddings, records, model)
    print_results(results)


if __name__ == "__main__":
    main()
