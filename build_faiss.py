import os
import sys
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIGURACIÓN ===
CARPETA_MARKDOWN = "TSpec-LLM/3GPP-clean/Rel-15"
MODELO_EMBEDDINGS = "BAAI/bge-large-en-v1.5"
INDEX_FAISS = "faiss_tspec"
ARCHIVO_PROGRESO = "progreso_faiss.txt"
BATCH_SIZE = 128

load_dotenv()

def cargar_documentos():
    print("📥 Cargando documentos Markdown...")
    loader = DirectoryLoader(CARPETA_MARKDOWN, glob="**/*.md", loader_cls=TextLoader, show_progress=True)
    documentos = loader.load()
    print(f"✅ Se cargaron {len(documentos)} documentos.")
    return documentos

def dividir_en_chunks(documentos):
    print("✂️ Dividiendo en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(documentos)
    chunks = [Document(page_content=f"passage: {chunk.page_content}", metadata=chunk.metadata) for chunk in chunks]
    print(f"✅ Se generaron {len(chunks)} fragmentos.")
    return chunks

def dividir_en_lotes(lista, tamaño_lote):
    for i in range(0, len(lista), tamaño_lote):
        yield lista[i:i + tamaño_lote]

def leer_último_progreso(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return int(f.read().strip())
    return 0

def guardar_progreso(path, index):
    with open(path, "w") as f:
        f.write(str(index))

def construir_faiss(chunks, fresh=False):
    if fresh:
        print("🔁 Rehaciendo desde cero...")
        if os.path.exists(ARCHIVO_PROGRESO):
            os.remove(ARCHIVO_PROGRESO)
        if os.path.exists(INDEX_FAISS):
            import shutil
            shutil.rmtree(INDEX_FAISS)

    print("🧠 Generando embeddings y construyendo FAISS (resumible)...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️ Usando dispositivo: {device}")
    model = SentenceTransformer(MODELO_EMBEDDINGS, device=device)
    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 128}
    )
    lotes = list(dividir_en_lotes(chunks, BATCH_SIZE))
    último = 0 if fresh else leer_último_progreso(ARCHIVO_PROGRESO)

    db = None
    progress = tqdm(total=len(lotes), desc="Indexando", initial=último, ncols=80)

    for i, lote in enumerate(lotes):
        if i < último:
            continue

        if db is None and i == 0:
            db = FAISS.from_documents(lote, embeddings)
        else:
            if db is None:
                db = FAISS.load_local(INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(lote)

        guardar_progreso(ARCHIVO_PROGRESO, i + 1)
        db.save_local(INDEX_FAISS)
        progress.update(1)

    progress.close()
    print("✅ FAISS guardado completamente.")

if __name__ == "__main__":
    fresh_flag = "--fresh" in sys.argv
    documentos = cargar_documentos()
    chunks = dividir_en_chunks(documentos)
    construir_faiss(chunks, fresh=fresh_flag)
