# HR Policy Assistant

## Project Overview

HR Policy Assistant is a Streamlit-based chatbot that answers employee HR policy questions using local policy documents. The system uses Retrieval-Augmented Generation (RAG): it first retrieves relevant HR policy content from a local vector database and then uses Google Gemini to generate a clear answer based only on that retrieved context.

The application is designed to reduce the time employees spend searching through HR documents and to provide quick, policy-grounded responses for common HR queries.

## Objectives

- Build an interactive HR chatbot interface.
- Store HR policy documents locally.
- Convert policy documents into vector embeddings for semantic search.
- Retrieve the most relevant policy content for a user question.
- Generate answers using Gemini while restricting responses to available HR policy data.
- Provide a simple and reusable project structure for future policy expansion.

## Technologies Used

- Python
- Streamlit
- Google Gemini API
- ChromaDB
- Sentence Transformers
- python-dotenv

## Main Features

- User-friendly web interface built with Streamlit.
- Supports natural language HR questions.
- Uses semantic search instead of keyword-only matching.
- Retrieves policy information from local HR documents.
- Generates answers using Gemini.
- Responds with a fallback message when the answer is not present in the policy documents.

## Project Structure

```text
Hr chatbot/
├── app.py
├── ingest.py
├── api.env
├── data/
│   ├── compensation_policy.txt
│   ├── leave_policy.txt
│   └── onboarding_policy.txt
├── chroma_db/
└── README.md
```

## File Description

| File or Folder | Description |
| --- | --- |
| `app.py` | Main Streamlit application. It accepts user questions, retrieves relevant policy data, and generates answers using Gemini. |
| `ingest.py` | Reads HR policy text files, creates embeddings, and stores them in ChromaDB. |
| `data/` | Contains HR policy source documents. |
| `chroma_db/` | Local persistent ChromaDB vector database created after running `ingest.py`. |
| `api.env` | Stores environment variables such as the Gemini API key. This file should not be shared publicly. |

## System Workflow

1. HR policy documents are stored as `.txt` files inside the `data` folder.
2. `ingest.py` reads each policy file.
3. The Sentence Transformer model `all-MiniLM-L6-v2` converts each policy document into an embedding.
4. The embeddings and original policy text are stored in a local ChromaDB collection named `hr_policies`.
5. The user enters an HR-related question in the Streamlit app.
6. The question is converted into an embedding using the same Sentence Transformer model.
7. ChromaDB retrieves the most relevant policy documents.
8. The retrieved context and user question are sent to Gemini.
9. Gemini generates an answer using only the retrieved HR policy context.

## Code Snippet and Explanation

The project has two main parts: document ingestion and chatbot response generation.

### 1. Creating the Vector Database

This code from `ingest.py` reads HR policy files, converts them into embeddings, and stores them in ChromaDB.

```python
model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="hr_policies"
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
```

Explanation:

- `SentenceTransformer("all-MiniLM-L6-v2")` loads the embedding model used to convert text into numerical vectors.
- `chromadb.PersistentClient(path="chroma_db")` creates a local persistent vector database.
- `get_or_create_collection(name="hr_policies")` creates or opens the collection where HR policy data is stored.
- `model.encode(text)` converts each policy document into an embedding.
- `collection.add(...)` stores the document text, embedding, ID, and source filename in ChromaDB.

### 2. Retrieving Context and Generating an Answer

This code from `app.py` takes the user's question, finds the most relevant HR policy content, and sends it to Gemini for answer generation.

```python
query_embedding = embedding_model.encode(question)

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

response = model.generate_content(prompt)
```

Explanation:

- `embedding_model.encode(question)` converts the user's question into an embedding.
- `collection.query(...)` compares the question embedding with stored policy embeddings and retrieves the top two most relevant documents.
- `context = "\n".join(results["documents"][0])` combines the retrieved policy text into one context block.
- The prompt instructs Gemini to answer only from the retrieved HR policy context.
- `model.generate_content(prompt)` generates the final response displayed in the Streamlit app.

### 3. Streamlit User Interface

This code creates the input field and displays the generated answer.

```python
question = st.text_input("Ask an HR Question")

if question:
    with st.spinner("Generating answer..."):
        response = model.generate_content(prompt)

    st.subheader("Answer")
    st.write(response.text)
```

Explanation:

- `st.text_input(...)` allows the user to enter an HR question.
- `st.spinner(...)` shows a loading message while the answer is being generated.
- `st.subheader("Answer")` displays a heading for the response section.
- `st.write(response.text)` prints the Gemini-generated answer on the web page.

## Installation and Setup

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install streamlit chromadb sentence-transformers google-generativeai python-dotenv
```

### 3. Configure API Key

Create an `api.env` file in the project root and add your Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
```

Optional model configuration:

```text
GEMINI_MODEL=gemini-2.5-flash
```

Important: Do not upload or submit your real API key in public repositories.

### 4. Create the Vector Database

Run the ingestion script:

```bash
python ingest.py
```

Expected output:

```text
Vector database created successfully!
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in a browser. Enter an HR policy question in the input field to receive an answer.

## Sample Questions

- How many annual leave days do employees get?
- When is payroll processed?
- Are employees eligible for performance bonuses?
- What do new employees receive during onboarding?
- When should mandatory training be completed?
- Can unused leave be carried forward?

## Sample Expected Answers

Question:

```text
How many sick leave days are provided?
```

Expected answer:

```text
Employees receive 10 sick leave days per year.
```

Question:

```text
When are salary revisions conducted?
```

Expected answer:

```text
Salary revisions are conducted annually in April.
```

Question:

```text
What is the travel reimbursement policy?
```

Expected answer:

```text
I could not find that information in the HR policies.
```

## Current Policy Data

The chatbot currently includes three policy areas:

- Leave Policy
- Compensation Policy
- Onboarding Policy

These policies cover annual leave, sick leave, maternity leave, salary revisions, bonuses, payroll processing, onboarding assets, HR orientation, and mandatory training timelines.

## Limitations

- The chatbot can only answer from the policy documents currently present in the `data` folder.
- If a policy is missing from the documents, the chatbot should return the fallback response.
- The current ingestion script stores each text file as a single document. For larger documents, chunking can improve retrieval accuracy.
- The app requires a valid Gemini API key to generate responses.

## Future Enhancements

- Add more HR policy files, such as travel, remote work, benefits, and disciplinary policies.
- Add document chunking for long policy files.
- Display source document names with each answer.
- Add chat history for better user experience.
- Add admin functionality to upload new policy documents from the interface.
- Add authentication for internal company use.

## Conclusion

HR Policy Assistant demonstrates how a RAG-based chatbot can help employees quickly access HR information. By combining local policy documents, vector search, and Gemini-generated responses, the project provides a practical example of using generative AI for workplace support and internal knowledge access.
# HR-Chatbot
