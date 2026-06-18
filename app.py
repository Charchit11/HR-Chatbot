import os
import chromadb
import streamlit as st
import google.generativeai as genai

from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / "api.env")

st.title(
    "🤖 HR Policy Assistant"
)

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error(
        "Missing Gemini API key. Add GEMINI_API_KEY=your_key to api.env."
    )
    st.stop()

genai.configure(
    api_key=api_key
)

model_name = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

model = genai.GenerativeModel(
    model_name
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path=str(BASE_DIR / "chroma_db")
)

collection = client.get_collection(
    "hr_policies"
)

question = st.text_input(
    "Ask an HR Question"
)

if question:

    with st.spinner(
        "Generating answer..."
    ):

        query_embedding = (
            embedding_model.encode(
                question
            )
        )

        results = collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=2
        )

        context = "\n".join(
            results["documents"][0]
        )

        prompt = f"""
You are an HR assistant.

Answer ONLY using
the HR policy context below.

If the answer
does not exist,
say:

'I could not find that
information in the HR policies.'

Context:

{context}

Question:

{question}
"""

        response = model.generate_content(
            prompt
        )

    st.subheader(
        "Answer"
    )

    st.write(
        response.text
    )
