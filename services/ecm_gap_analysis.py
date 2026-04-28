import argparse
import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

try:
    from services import generation, retrieve
except ImportError:
    import generation
    import retrieve


INPUT_FILE = "ecm_obligations.jsonl"
OUTPUT_FILE = "ecm_gap_analysis.jsonl"
PROMPT_TEMPLATE = """You are assisting a healthcare compliance analyst. The analyst is reviewing whether an existing set of Public Policies (P&Ps) covers a compliance obligation extracted from a DHCS Policy Guide.

Compliance obligation:
{obligation_text}

Source quote from Policy Guide:
{source_quote}

You are given retrieved P&P chunks. Each chunk has:
- rank
- file_name
- page_number
- clean_text
{retrieved_chunks}

Your task:
1. Decide whether the retrieved P&P chunks cover the obligation.
2. Assign one status:
   - Covered: the P&P evidence clearly addresses the obligation.
   - Partial: the P&P evidence addresses part of the obligation, but something appears missing or incomplete.
   - Missing: the retrieved P&P evidence does not address the obligation.
   - Unclear: the retrieved P&P evidence is ambiguous or insufficient to decide.
3. Recommend an action.
4. Provide brief reasoning.
5. Select the most relevant P&P evidence chunks used in your decision.

Rules:
- Use only the provided obligation, source quote, and retrieved P&P chunks.
- Do not use outside knowledge.
- Do not guess.
- If the evidence is insufficient, use "Unclear" or "Missing".
- Quotes must be exact text copied from clean_text.
- Each evidence item must reference a valid rank from the retrieved chunks.
- Select 1–3 evidence items if relevant. If no useful evidence exists, return an empty evidence list.
- Return only valid JSON.

Return JSON in this exact format:

{{
  "status": "Covered | Partial | Missing | Unclear",
  "recommended_action": "No update needed | Update existing P&P | Write new P&P section | Human review needed",
  "reasoning": "...",
  "evidence": [
    {{
      "rank": 1,
      "file_name": "...",
      "page_number": 2,
      "quote": "...",
      "why_relevant": "..."
    }}
  ]
}}"""


def get_default_paths():
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    return processed_dir / INPUT_FILE, processed_dir / OUTPUT_FILE


def load_obligations(input_path: Path):
    obligations = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping invalid JSON on line {line_number}: {exc}")
                continue

            obligation_text = record.get("obligation_text")
            source_quote = record.get("source_quote")
            if not isinstance(obligation_text, str) or not obligation_text.strip():
                print(
                    f"Warning: skipping obligation with empty obligation_text on line {line_number}"
                )
                continue
            if not isinstance(source_quote, str) or not source_quote.strip():
                print(
                    f"Warning: skipping obligation with empty source_quote on line {line_number}"
                )
                continue

            obligations.append(record)

    return obligations


def format_chunks_for_prompt(chunks):
    prompt_chunks = []

    for chunk in chunks:
        prompt_chunks.append(
            {
                "rank": chunk["rank"],
                "file_name": chunk["file_name"],
                "page_number": chunk["page_number"],
                "clean_text": chunk["clean_text"],
            }
        )

    return json.dumps(prompt_chunks, ensure_ascii=False, indent=2)


def build_prompt(obligation_text: str, source_quote: str, retrieved_chunks):
    return PROMPT_TEMPLATE.format(
        obligation_text=obligation_text,
        source_quote=source_quote,
        retrieved_chunks=format_chunks_for_prompt(retrieved_chunks),
    )


def retrieve_chunks_for_obligation(
    obligation_text: str, embeddings, records, model: SentenceTransformer, top_k: int
):
    original_top_k = retrieve.TOP_K

    try:
        retrieve.TOP_K = top_k
        return retrieve.retrieve_top_pages(obligation_text, embeddings, records, model)
    finally:
        retrieve.TOP_K = original_top_k


def analyze_single_obligation(
    obligation, embeddings, records, model: SentenceTransformer, top_k: int
):
    retrieved_chunks = retrieve_chunks_for_obligation(
        obligation["obligation_text"], embeddings, records, model, top_k
    )
    prompt = build_prompt(
        obligation["obligation_text"],
        obligation["source_quote"],
        retrieved_chunks,
    )
    response_text = generation.call_llm(prompt)
    response_json = generation.parse_llm_response(response_text)

    status = response_json.get("status", "")
    recommended_action = response_json.get("recommended_action", "")
    reasoning = response_json.get("reasoning", "")
    evidence = response_json.get("evidence", [])

    if not isinstance(evidence, list):
        raise RuntimeError("LLM response field 'evidence' must be a list.")

    result = {
        "obligation_text": obligation["obligation_text"],
        "source_quote": obligation["source_quote"],
        "file_name": obligation.get("file_name", ""),
        "page_number": obligation.get("page_number", ""),
        "status": status,
        "recommended_action": recommended_action,
        "reasoning": reasoning,
        "evidence": evidence,
        "retrieved_chunks": retrieved_chunks,
    }

    return result


def analyze_obligations(
    input_path: Path, output_path: Path, limit: int, top_k: int
):
    obligations = load_obligations(input_path)
    obligations_to_process = obligations[:limit]
    embeddings, records = retrieve.load_data()
    model = SentenceTransformer(retrieve.MODEL_NAME)
    results = []

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, obligation in enumerate(obligations_to_process, start=1):
            try:
                result = analyze_single_obligation(
                    obligation, embeddings, records, model, top_k
                )
            except Exception as exc:
                print(f"Warning: failed to process obligation {index}: {exc}")
                continue

            results.append(result)
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"Current obligation number: {index}")
            print(f"Status returned: {result['status']}")
            print(f"Total results saved: {len(results)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ECM obligation gap analysis against retrieved P&P chunks."
    )
    parser.add_argument(
        "--input",
        help="Path to ecm_obligations.jsonl",
    )
    parser.add_argument(
        "--output",
        help="Path to ecm_gap_analysis.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Process only the first N obligations.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieved P&P chunks per obligation.",
    )
    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")

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

    analyze_obligations(input_path, output_path, args.limit, args.top_k)


if __name__ == "__main__":
    main()
