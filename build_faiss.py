# build_faiss.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

import os
import torch
from dotenv import load_dotenv
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIGURACIÓN ===
CARPETA_MARKDOWN = "TSpec-LLM/3GPP-clean/Rel-15"  # Cambia esta ruta si es necesario
MODELO_EMBEDDINGS = "BAAI/bge-large-en-v1.5"
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(documentos)
    chunks = [Document(page_content=f"passage: {chunk.page_content}", metadata=chunk.metadata) for chunk in chunks]    
    print(f"✅ Se generaron {len(chunks)} fragmentos.")
    return chunks

# === PASO 3: Construir FAISS ===
def construir_faiss(chunks):
    print("🧠 Generando embeddings en batches y construyendo FAISS...")

    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️ Usando dispositivo: {device}")
    
    embeddings = SentenceTransformerEmbeddings(model_name=MODELO_EMBEDDINGS)

    batch_size = 64
    progress = tqdm(total=len(chunks), desc="Indexando", ncols=80)

    first_batch = chunks[:batch_size]
    db = FAISS.from_documents(first_batch, embeddings)
    progress.update(len(first_batch))

    for i in range(batch_size, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        db.add_documents(batch)
        progress.update(len(batch))

    progress.close()
    db.save_local(INDEX_FAISS)

    print("✅ FAISS guardado.")
    return db

if __name__ == "__main__":
    documentos = cargar_documentos()
    chunks = dividir_en_chunks(documentos)
    construir_faiss(chunks)
