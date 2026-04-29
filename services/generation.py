import argparse
import json
import urllib.error
import urllib.request
import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    from services import retrieve
except ImportError:
    import retrieve

load_dotenv()

def get_google_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None

API_KEY = get_google_api_key()

if not API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. Add it to .env locally or Streamlit secrets in deployment."
    )

MODEL_NAME = "gemini-2.5-flash"
API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_NAME}:generateContent"
)
PROMPT_TEMPLATE = """You are assisting a compliance auditor. 
The auditor has asked this question:
{question}

You are given retrieved policy chunks. Each chunk has:
- rank
- similarity_score
- file_name
- page_number
- clean_text
{retrieved_chunks}

Your task:
1. Propose an answer: Yes, No, or Unclear.
2. Provide a brief reasoning based ONLY on the excerpts.
3. Select 2–3 most relevant chunks as evidence. If only one chunk clearly supports the answer, you may return one. Do not include more than 4.
4. For each evidence item:
   - reference the chunk by rank
   - include file_name, and page_number
   - include an exact quote copied from clean_text
   - explain why the quote is relevant
5. If there is ambiguity, missing information, exceptions, or possible conflicting language, mention it.

Rules:
- Use ONLY the provided chunks
- Do NOT use outside knowledge
- Do NOT guess
- If the retrieved chunks do not provide enough evidence, proposed_answer must be "Unclear".
- Quotes must be exact text copied from clean_text.
- Each evidence item must reference a valid rank from the retrieved chunks.
- Return only valid JSON

Return JSON in this exact format:

{{
  "proposed_answer": "Yes | No | Unclear",
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


def format_chunks_for_prompt(chunks):
    prompt_chunks = []

    for chunk in chunks:
        prompt_chunks.append(
            {
                "rank": chunk["rank"],
                "similarity_score": chunk["score"],
                "file_name": chunk["file_name"],
                "page_number": chunk["page_number"],
                "clean_text": chunk["clean_text"],
            }
        )

    return json.dumps(prompt_chunks, ensure_ascii=False, indent=2)


def build_prompt(question: str, chunks):
    return PROMPT_TEMPLATE.format(
        question=question,
        retrieved_chunks=format_chunks_for_prompt(chunks),
    )


def retrieve_chunks(question: str):
    embeddings, records = retrieve.load_data()
    model = SentenceTransformer(retrieve.MODEL_NAME)
    return retrieve.retrieve_top_pages(question, embeddings, records, model)


def call_llm(prompt: str):
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM request failed with status {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    try:
        response_json = json.loads(body)
        return response_json["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unexpected LLM response format: {body}") from exc


def parse_llm_response(response_text: str):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM did not return valid JSON: {response_text}") from exc


def generate_answer(question: str):
    chunks = retrieve_chunks(question)
    prompt = build_prompt(question, chunks)
    llm_response_text = call_llm(prompt)
    llm_response_json = parse_llm_response(llm_response_text)
    return llm_response_json, chunks


def print_retrieved_chunks(chunks):
    print("\nRetrieved chunks:")
    for chunk in chunks:
        print(json.dumps(chunk, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Generate a RAG answer using retrieved policy chunks."
    )
    parser.add_argument("question", help="User question to answer.")
    args = parser.parse_args()

    try:
        llm_response, chunks = generate_answer(args.question)
    except Exception as exc:
        print(f"Error: {exc}")
        return

    print(json.dumps(llm_response, ensure_ascii=False, indent=2))
    print_retrieved_chunks(chunks)

if __name__ == "__main__":
    main()
