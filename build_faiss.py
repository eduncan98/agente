# build_faiss.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIGURACIÓN ===
CARPETA_MARKDOWN = "TSpec-LLM/3GPP-clean/Rel-15"  # Cambia esta ruta si es necesario
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FAISS = "faiss_tspec"

load_dotenv()

# === PASO 1: Cargar documentos ===
def cargar_documentos():
    print("📥 Cargando documentos Markdown...")
    loader = DirectoryLoader(CARPETA_MARKDOWN, glob="**/*.md", loader_cls=TextLoader, show_progress=True)
    documentos = loader.load()
    print(f"✅ Se cargaron {len(documentos)} documentos.")
    return documentos

# === PASO 2: Dividir documentos ===
def dividir_en_chunks(documentos):
    print("✂️ Dividiendo en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documentos)
    print(f"✅ Se generaron {len(chunks)} fragmentos.")
    return chunks

# === PASO 3: Construir FAISS ===
def construir_faiss(chunks):
    print("🧠 Generando embeddings en batches y construyendo FAISS...")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        encode_kwargs={
            "batch_size": 64,
        }
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_FAISS)

    print("✅ FAISS guardado.")
    return db

if __name__ == "__main__":
    documentos = cargar_documentos()
    chunks = dividir_en_chunks(documentos)
    construir_faiss(chunks)
