import json

import streamlit as st

from services import ecm_gap_analysis
from services.generation import generate_answer


st.set_page_config(page_title="Compliance Assistant", layout="wide")
st.title("Compliance Assistant")

questionnaire_tab, policy_guide_tab = st.tabs(
    ["Questionnaire Assistant", "Policy Guide Review"]
)


def load_gap_analysis_results(results_path, limit=None):
    results = []

    with results_path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            results.append(json.loads(line))
            if limit is not None and len(results) >= limit:
                break

    return results

with questionnaire_tab:
    question = st.text_input("User question")
    submitted = st.button("Submit")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Retrieving policy chunks and generating answer..."):
                    llm_response, chunks = generate_answer(question.strip())
            except Exception as exc:
                st.error(f"Error: {exc}")
            else:
                st.subheader("Proposed Answer")
                st.write(llm_response.get("proposed_answer", ""))

                st.subheader("Reasoning")
                st.write(llm_response.get("reasoning", ""))

                st.subheader("Evidence")
                evidence_items = llm_response.get("evidence", [])
                if evidence_items:
                    for item in evidence_items:
                        st.markdown(f"**Quote**: {item.get('quote', '')}")
                        st.write(f"Rank: {item.get('rank', '')}")
                        st.write(f"File Name: {item.get('file_name', '')}")
                        st.write(f"Page Number: {item.get('page_number', '')}")
                        if item.get("why_relevant"):
                            st.write(f"Why Relevant: {item.get('why_relevant')}")
                        st.divider()
                else:
                    st.write("No evidence returned.")

                st.subheader("All Retrieved Chunks For Review")
                for chunk in chunks:
                    st.markdown(f"**Rank {chunk['rank']}**")
                    st.write(f"Score: {chunk['score']:.4f}")
                    st.write(f"File Name: {chunk['file_name']}")
                    st.write(f"Page Number: {chunk['page_number']}")
                    st.write(f"Preview: {chunk['clean_text'][:700]}")
                    st.divider()

with policy_guide_tab:
    sample_clicked = st.button("Use Sample Policy Guide (Pages 11–20)")
    st.button("Upload Policy Guide (Coming Soon)", disabled=True)

    if sample_clicked:
        try:
            input_path, output_path = ecm_gap_analysis.get_default_paths()
            obligations = ecm_gap_analysis.load_obligations(input_path)
        except Exception as exc:
            st.error(f"Error: {exc}")
        else:
            st.write(f"Total number of extracted obligations: {len(obligations)}")
            st.write("Prototype mode: analyzing first 10 obligations")

            try:
                if output_path.is_file():
                    results = load_gap_analysis_results(output_path, limit=10)
                else:
                    with st.spinner("Analyzing the first 10 obligations..."):
                        results = ecm_gap_analysis.analyze_obligations(
                            input_path=input_path,
                            output_path=output_path,
                            limit=10,
                            top_k=10,
                        )
            except Exception as exc:
                st.error(f"Error: {exc}")
            else:
                summary_rows = []
                for result in results:
                    summary_rows.append(
                        {
                            "Obligation Text": result.get("obligation_text", ""),
                            "Obligation Page Number": result.get("page_number", ""),
                            "Status": result.get("status", ""),
                            "Recommended Action": result.get(
                                "recommended_action", ""
                            ),
                        }
                    )

                st.subheader("Gap Analysis Results")
                st.dataframe(summary_rows, use_container_width=True, hide_index=True)

                for index, result in enumerate(results, start=1):
                    expander_label = (
                        f"Show reasoning details for obligation {index}"
                    )
                    with st.expander(expander_label):
                        st.markdown("**Source Quote**")
                        st.write(result.get("source_quote", ""))

                        st.markdown("**Reasoning**")
                        st.write(result.get("reasoning", ""))

                        st.markdown("**Selected P&P Evidence**")
                        evidence_items = result.get("evidence", [])
                        if evidence_items:
                            for item in evidence_items:
                                st.markdown(f"**Quote**: {item.get('quote', '')}")
                                st.write(f"File Name: {item.get('file_name', '')}")
                                st.write(f"Page Number: {item.get('page_number', '')}")
                                if item.get("why_relevant"):
                                    st.write(
                                        f"Why Relevant: {item.get('why_relevant', '')}"
                                    )
                                st.divider()
                        else:
                            st.write("No evidence returned.")

                        st.markdown("**Retrieved Chunks For Review**")
                        for chunk in result.get("retrieved_chunks", []):
                            st.write(f"Rank: {chunk.get('rank', '')}")
                            st.write(f"Score: {chunk.get('score', 0.0):.4f}")
                            st.write(f"File Name: {chunk.get('file_name', '')}")
                            st.write(f"Page Number: {chunk.get('page_number', '')}")
                            st.write(
                                f"Preview: {chunk.get('clean_text', '')[:700]}"
                            )
                            st.divider()
