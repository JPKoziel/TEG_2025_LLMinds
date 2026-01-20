import streamlit as st
import requests

API_URL = "http://localhost:8000/rag/naive"

st.set_page_config(page_title="LLMinds – GraphRAG", layout="centered")

st.title("🧠 GraphRAG Demo")

question = st.text_input(
    "Ask a question about the CV graph:",
    placeholder="How many people have Python skills?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Querying GraphRAG..."):
            response = requests.post(
                API_URL,
                json={"question": question},
                timeout=60
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            if data.get("context"):
                with st.expander("Context"):
                    st.write(data["context"])
        else:
            st.error(f"API error: {response.text}")
