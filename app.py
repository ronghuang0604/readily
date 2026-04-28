import streamlit as st

from services.generation import generate_answer


st.set_page_config(page_title="Policy RAG", layout="wide")
st.title("Compliance Analyst Assistant")

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

