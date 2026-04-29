# Compliance Assistant

## Overview
This project builds a prototype “Compliance Assistant” to help healthcare compliance analysts:
1. Answer regulatory questionnaires using existing P&P documents
2. Proactively review Policy Guides to identify gaps in current policies

This system is not intended to replace compliance analysts. Instead, it aims to reduce time spent searching documents, surface relevant evidence quickly, and highlight potential gaps early. The goal is to help the user make better decisions, not to make decisions for them.


## Features
### 1. Questionnaire Assistant

- Accepts natural language compliance questions  
- Retrieves relevant P&P sections using semantic search
- Generates a proposed answer (**Yes / No / Unclear**)  
- Provides:
  - reasoning
  - selected evidence (exact quotes with citations)
  - full retrieved chunks for transparency  
### 2. Policy Guide Review
- Extracts concrete compliance obligations from a ECM Policy Guidelines (sample pages for MVP)  
- Checks each obligation against existing P&P documents
- Flags:
  - Covered
  - Partial
  - Missing
  - Unclear
- Provides:
  - recommended action (e.g., update policy, write new section)
  - supporting evidence and reasoning  


## System Design
### Data Pipeline

P&P Documents:
PDF → text extraction → embeddings → semantic retrieval

ECM Policy Guidlines:
PDF → text extraction → obligation extraction → gap analysis

### Workflow

Task 1:
Question → retrieve P&P → LLM generates answer + evidence

Task 2:
ECM Policy Guidelines → extract obligations → retrieve P&P → LLM evaluates coverage

### Key Design Decisions
1. Human-in-the-loop design instead of fully automation
Rather than returning a black-box answer, the system provides recommendations and evidence. This allows the auditor to verify and trust the result, instead of blindly relying on the model.

2. Retrieval-Augmented Generation with Transparency
The system is built as a Retrieval-Augmented Generation (RAG) pipeline, where relevant policy sections are retrieved and used to ground LLM reasoning. 

    Unlike typical RAG implementations, this system emphasizes transparency and auditability by:
    - surfacing both selected evidence and full retrieved context
    - allowing human verification of model outputs
    - avoiding black-box responses in a high-risk compliance setting

3. Limited scope for MVP
Restricted analysis to a subset of pages / obligations to control latency and API cost.


## Tech Stack
BE:
- Python
- NumPy
- PyMuPDF (PDF extraction)

ML:
- Sentence Transformers (embeddings)
- Google Gemini API (LLM)

FE:
- Streamlit


## Tradeoffs
1. Chunking Strategy
    There is a tradeoff between precision and context. Smaller chunks are more precise but risk losing surrounding context. Larger chunks have more context but noisier and less precise.

    For this MVP, I chose page-level chunking because:
    - it is simple and reliable
    - it aligns with how auditors reference documents
    - it provides natural citation boundaries
2. Top-K Retrieval
    Another key tradeoff is precision vs recall.
    In this domain, recall is more important than precision. Missing a relevant policy could lead to compliance risk, while extra context can still be reviewed by the auditor.
    So I chose a higher K and relied on the LLM to highlight the most relevant evidence, while still exposing all retrieved results for transparency.



## Limitations

This prototype makes several intentional tradeoffs:

### Scope
- Processes a subset of pages and obligations instead of full-document batch processing

### Accuracy
- Page-by-page extraction may miss requirements spanning multiple pages
- Semantic retrieval may miss edge cases or return approximate matches
- No deep contradiction detection across documents

### Data
- Uses basic PDF text extraction. So structured content like tables, charts, and images may not be captured accurately


## Future Improvements


### Extraction
- Multi-page context windows
- Improved handling of structured content

### Retrieval & Reasoning
- Hybrid search (keyword + semantic)
- Reranking
- Better contradiction and exception detection
- Caching to reduce repeated LLM calls
- Add evaluation framework with labeled test cases

### Product
- Full document processing with batching
- Version comparison across Policy Guide updates
- Improved UI for review and export


## How to Run
1. Install dependencies:
   pip install -r requirements.txt
2. Set API key:
    export GOOGLE_API_KEY=your_key
3. Run app:
   streamlit run app.py

## Demo
Please find sample questions from data/processed/retrieval_eval_set.json

