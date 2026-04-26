import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_FILE = "/data/processed/extracted_pages.jsonl"
OUTPUT_EMBEDDINGS = "/data/processed/embeddings.npy"
OUTPUT_RECORDS = "/data/processed/embedded_pages.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_pages(input_path: Path):
    pages = []
    total_loaded = 0

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping invalid JSON on line {line_number}: {exc}")
                continue

            total_loaded += 1
            clean_text = record.get("clean_text")

            if not isinstance(clean_text, str) or not clean_text.strip():
                continue

            try:
                pages.append(
                    {
                        "file_path": record["file_path"],
                        "file_name": record["file_name"],
                        "page_number": int(record["page_number"]),
                        "clean_text": clean_text,
                    }
                )
            except Exception as exc:
                print(f"Warning: skipping bad record on line {line_number}: {exc}")

    return total_loaded, pages


def save_records(records, output_path: Path):
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_embeddings(records):
    model = SentenceTransformer(MODEL_NAME)
    texts = [record["clean_text"] for record in records]
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def main():
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / INPUT_FILE
    embeddings_path = script_dir / OUTPUT_EMBEDDINGS
    records_path = script_dir / OUTPUT_RECORDS

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    total_loaded, pages = load_pages(input_path)

    if pages:
        embeddings = build_embeddings(pages)
    else:
        embeddings = np.empty((0, 384), dtype=np.float32)

    np.save(embeddings_path, embeddings)
    save_records(pages, records_path)

    print(f"total number of pages loaded: {total_loaded}")
    print(f"number of pages embedded: {len(pages)}")
    print(f"embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
