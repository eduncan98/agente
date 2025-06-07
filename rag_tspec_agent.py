# rag_tspec_agent.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIGURACIÓN ===
CARPETA_MARKDOWN = "TSpec-LLM/3GPP-clean/Rel-15"  # Cambia esta ruta si es necesario
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FAISS = "faiss_tspec"
USAR_INDEX_EXISTENTE = True  # Cambia a False si quieres recrear el índice
from dotenv import load_dotenv
import sys

load_dotenv()  # Carga automáticamente el archivo .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ ERROR: No se encontró OPENAI_API_KEY en el archivo .env.")
    sys.exit(1)

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

# === PASO 3: Crear o cargar FAISS ===
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

def cargar_faiss():
    print("📂 Cargando FAISS existente...")
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    db = FAISS.load_local(INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
    return db

# === PASO 4: Crear agente y responder ===
def crear_agente(db):
    print("🤖 Inicializando agente...")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = db.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain

def hacer_pregunta(chain):
    print("\n💬 Agente listo. Escribe tu pregunta técnica (o 'salir'):")
    while True:
        query = input("🧠> ").strip()
        if query.lower() == "salir":
            break
        respuesta = chain.invoke({"query":query})["result"]
        print(f"\n📘 Respuesta:\n{respuesta}\n")

# === MAIN ===
if __name__ == "__main__":
    if USAR_INDEX_EXISTENTE and os.path.exists(f"{INDEX_FAISS}/index.faiss"):
        db = cargar_faiss()
    else:
        documentos = cargar_documentos()
        chunks = dividir_en_chunks(documentos)
        db = construir_faiss(chunks)

    agente = crear_agente(db)
    hacer_pregunta(agente)