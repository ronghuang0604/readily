import argparse
import json
import re
from pathlib import Path

import fitz


def clean_text(raw_text: str) -> str:
    text = raw_text.strip()
    return re.sub(r"\s+", " ", text)


def extract_pdf_pages(root_folder: Path):
    for path in root_folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue

        try:
            with fitz.open(path) as document:
                for page_number, page in enumerate(document, start=1):
                    raw_text = page.get_text()
                    cleaned = clean_text(raw_text)

                    if not cleaned:
                        continue

                    yield {
                        "file_path": str(path.resolve()),
                        "file_name": path.name,
                        "page_number": page_number,
                        "raw_text": raw_text,
                        "clean_text": cleaned,
                    }
        except Exception as exc:
            print(f"Warning: skipping unreadable PDF {path}: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively extract text from PDF files into extracted_pages.jsonl."
    )
    parser.add_argument("folder_path", help="Root folder to scan for PDF files.")
    args = parser.parse_args()

    root_folder = Path(args.folder_path).expanduser().resolve()
    output_path = Path("extracted_pages.jsonl").resolve()

    if not root_folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a directory: {root_folder}")

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in extract_pdf_pages(root_folder):
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
