import os
import chromadb
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

genai.configure(
    api_key=os.getenv(
        "GEMINI_API_KEY"
    )
)

model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    "hr_policies"
)

st.title(
    "🤖 HR Policy Assistant"
)

question = st.text_input(
    "Ask an HR Question"
)

if question:

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

    st.subheader(
        "Source"
    )

    st.write(
        results["metadatas"][0][0]["source"]
    )