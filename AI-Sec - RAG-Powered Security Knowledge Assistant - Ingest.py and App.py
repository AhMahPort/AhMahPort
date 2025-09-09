# install dependencies

pip install -r requirements.txt

# requirements.txt should have > langchain, faiss-cpu, sentence-transformers, gradio, pypdf

import os
import glob
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DOCS_DIR = "./data/docs"
VECTOR_DIR = "./vectorstore"

def load_documents(docs_dir):
    pdfs = glob.glob(os.path.join(docs_dir, "*.pdf"))
    all_docs = []
    for pdf in pdfs:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} pages from {len(pdfs)} PDF(s).")
    return all_docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def build_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def save_vectorstore(chunks):
    embeddings = build_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    db.save_local(VECTOR_DIR)
    print(" Vectorstore saved!")

if __name__ == "__main__":
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    docs = load_documents(DOCS_DIR)
    if not docs:
        print(f"No PDFs found in {DOCS_DIR}. Please add files and try again.")
    else:
        chunks = split_documents(docs)
        save_vectorstore(chunks)





# -----------------------------------------------------------------------------------------------------------------

import os
import gradio as gr
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

VECTOR_DIR = "./vectorstore"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(VECTOR_DIR):
        return None
    db = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
    return db

db = load_vectorstore()

def ask_question(query):
    if db is None:
        return "Please run ingestion first (python ingest.py)."
    # Retrieve top relevant chunks
    docs = db.similarity_search(query, k=3)
    if not docs:
        return "No relevant information found."
    # Create a basic answer using retrieved snippets
    response = "### Answer (from retrieved documents):\n"
    response += "\n\n".join([f"- {d.page_content.strip()}" for d in docs])
    response += "\n\n**(Note: This is a basic retrieval answer. For full RAG, you can add a language model later.)**"
    return response

def upload_pdfs(files):
    os.makedirs("./data/docs", exist_ok=True)
    for f in files:
        with open(f"./data/docs/{f.name}", "wb") as out:
            out.write(f.read())
    return "Uploaded! Now run `python ingest.py` to process and index."

with gr.Blocks(title="RAG-Powered Security Knowledge Assistant") as demo:
    gr.Markdown("# 🔐 RAG-Powered Security Knowledge Assistant")
    gr.Markdown("Upload security PDFs (NIST, OWASP, MITRE), index them, and ask questions.")

    with gr.Row():
        file_uploader = gr.File(file_count="multiple", file_types=[".pdf"], label="Upload PDFs")
        upload_status = gr.Textbox(label="Upload Status")
    upload_button = gr.Button("Upload Files")
    upload_button.click(upload_pdfs, inputs=[file_uploader], outputs=[upload_status])

    question = gr.Textbox(label="Ask a cybersecurity question", placeholder="e.g., What does NIST recommend for incident response?")
    answer = gr.Markdown()
    submit = gr.Button("Search")
    submit.click(ask_question, inputs=[question], outputs=[answer])

demo.launch()
