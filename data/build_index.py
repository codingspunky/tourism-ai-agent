import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================
# LangChain Modern Imports
# =============================

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# If using Groq only for LLM, embeddings still need OpenAI or HF

# =============================
# CONFIG
# =============================

DATA_FOLDER = os.path.dirname(__file__)
FAISS_PATH = os.path.join(os.path.dirname(DATA_FOLDER), "faiss_index")

# =============================
# LOAD DOCUMENTS
# =============================

def load_documents():
    documents = []

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_FOLDER, filename)
            print(f"📄 Loading: {filename}")
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    return documents


# =============================
# SPLIT DOCUMENTS
# =============================

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return text_splitter.split_documents(documents)


# =============================
# BUILD FAISS INDEX
# =============================

def build_faiss_index():

    print("🚀 Starting FAISS index build...")

    docs = load_documents()

    if not docs:
        print("❌ No .txt files found inside data/ folder.")
        return

    split_docs = split_documents(docs)

    print(f"🔹 Total chunks created: {len(split_docs)}")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


    # Build FAISS
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Save locally
    vectorstore.save_local(FAISS_PATH)

    print(f"✅ FAISS index saved at: {FAISS_PATH}")


# =============================
# MAIN
# =============================

if __name__ == "__main__":
    build_faiss_index()
