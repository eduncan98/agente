from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Nuevo import correcto
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
from tqdm import tqdm
import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIGURACIÓN ===
CARPETA_MARKDOWN = "TSpec-LLM/3GPP-clean/Rel-15"
MODELO_EMBEDDINGS = "BAAI/bge-large-en-v1.5"
INDEX_FAISS = "faiss_tspec"

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

def construir_faiss(chunks):
    print("🧠 Generando embeddings en batches y construyendo FAISS...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Usando dispositivo: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 128}
    )

    batch_size = 128
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
