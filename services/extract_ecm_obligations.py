import argparse
import json
import os
from pathlib import Path

try:
    from services.generation import call_llm, parse_llm_response
except ImportError:
    from generation import call_llm, parse_llm_response


INPUT_FILE = "extracted_ecm_pages.jsonl"
OUTPUT_FILE = "ecm_obligations.jsonl"
PROMPT_TEMPLATE = """You are assisting a healthcare compliance analyst reviewing a Policy Guide. You are given text from one page of a Policy Guide. Your task is to extract concrete compliance obligations from the text.
{extracted_ecm_page}

Definition:
A concrete obligation is a specific requirement that the health plan must follow. This includes:
- required actions
- deadlines or timelines
- documentation requirements
- notification requirements
- reporting requirements
- operational responsibilities

Do NOT extract:
- background information
- definitions only
- examples only
- descriptive or explanatory text without a requirement
- vague or general statements that do not impose an obligation

Rules:
- Use ONLY the provided page text
- Do NOT invent obligations
- If no obligations are present, return an empty list
- Split multiple obligations into separate items
- Each obligation should be atomic and clearly stated
- Include an exact quote copied from the page text as evidence

Return ONLY valid JSON in this format:
{{
  "obligations": [
    {{
      "obligation_text": "...",
      "source_quote": "..."
    }}
  ]
}}"""


def get_default_paths():
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    return processed_dir / INPUT_FILE, processed_dir / OUTPUT_FILE


def load_pages(input_path: Path):
    pages = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping invalid JSON on line {line_number}: {exc}")
                continue

            clean_text = record.get("clean_text")
            if not isinstance(clean_text, str) or not clean_text.strip():
                print(f"Warning: skipping page with empty clean_text on line {line_number}")
                continue

            pages.append(record)

    return pages


def build_prompt(extracted_ecm_page: str):
    return PROMPT_TEMPLATE.format(extracted_ecm_page=extracted_ecm_page)


def extract_obligations_from_page(page_record):
    prompt = build_prompt(page_record["clean_text"])
    response_text = call_llm(prompt)
    response_json = parse_llm_response(response_text)
    obligations = response_json.get("obligations", [])

    if not isinstance(obligations, list):
        raise RuntimeError("LLM response field 'obligations' must be a list.")

    page_obligations = []
    for item in obligations:
        if not isinstance(item, dict):
            continue

        obligation_text = item.get("obligation_text")
        source_quote = item.get("source_quote")

        if not isinstance(obligation_text, str) or not obligation_text.strip():
            continue
        if not isinstance(source_quote, str) or not source_quote.strip():
            continue

        page_obligations.append(
            {
                "obligation_text": obligation_text.strip(),
                "source_quote": source_quote.strip(),
                "file_name": page_record.get("file_name", ""),
                "page_number": page_record.get("page_number", ""),
            }
        )

    return page_obligations


def save_obligations(input_path: Path, output_path: Path):
    pages = load_pages(input_path)
    total_saved = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for page_record in pages:
            file_name = page_record.get("file_name", "")
            page_number = page_record.get("page_number", "")
            print(f"Processing page {page_number} from {file_name}")

            try:
                page_obligations = extract_obligations_from_page(page_record)
            except Exception as exc:
                print(
                    f"Warning: failed to process {file_name} page {page_number}: {exc}"
                )
                continue

            for obligation in page_obligations:
                output_file.write(json.dumps(obligation, ensure_ascii=False) + "\n")
                total_saved += 1

            print(f"Obligations extracted from page: {len(page_obligations)}")
            print(f"Total obligations saved: {total_saved}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ECM compliance obligations from page-level JSONL."
    )
    parser.add_argument(
        "--input",
        help="Path to extracted_ecm_pages.jsonl",
    )
    parser.add_argument(
        "--output",
        help="Path to ecm_obligations.jsonl",
    )
    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

    default_input_path, default_output_path = get_default_paths()
    input_path = (
        Path(args.input).expanduser().resolve() if args.input else default_input_path
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path
    )

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    save_obligations(input_path, output_path)


if __name__ == "__main__":
    main()
