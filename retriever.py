import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================
# PATH CONFIG
# ==============================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

# ==============================
# EMBEDDINGS
# ==============================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==============================
# LOAD VECTOR STORE
# ==============================

def load_vector_store():
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_PATH}. "
            f"Run build_index.py first."
        )

    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
