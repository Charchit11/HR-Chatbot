import os
import chromadb

from sentence_transformers import SentenceTransformer

DATA_FOLDER = "data"

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    name="hr_policies"
)

documents = []

for file in os.listdir(DATA_FOLDER):

    path = os.path.join(
        DATA_FOLDER,
        file
    )

    with open(path, "r", encoding="utf-8") as f:

        text = f.read()

        documents.append(
            (
                file,
                text
            )
        )

for idx, (filename, text) in enumerate(documents):

    embedding = model.encode(text)

    collection.add(
        ids=[str(idx)],
        documents=[text],
        embeddings=[embedding.tolist()],
        metadatas=[
            {
                "source": filename
            }
        ]
    )

print("Vector database created successfully!")