# ask_agent.py

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import sys
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIGURACIÓN ===
INDEX_FAISS = "faiss_tspec"
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ ERROR: No se encontró OPENAI_API_KEY en el archivo .env.")
    sys.exit(1)

# === PASO 1: Cargar FAISS ===
def cargar_faiss():
    print("📂 Cargando FAISS existente...")
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    db = FAISS.load_local(INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
    return db

# === PASO 2: Crear agente y responder ===
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

        # Recuperar documentos relevantes
        try:
            docs = chain.retriever.invoke(query)
        except AttributeError:
            print("⚠️ El retriever no está expuesto correctamente en la cadena.")
            return

        print("\n🔎 Documentos recuperados por FAISS:\n")
        try:
            for i, doc in enumerate(list(docs)[:5]):
                print(f"--- Documento {i+1} ---")
                print(doc.page_content[:500])
                print()
        except Exception as e:
            print(f"❌ Error mostrando documentos: {e}")

        # Obtener respuesta
        try:
            respuesta = chain.invoke(query)
        except Exception:
            respuesta = chain.run(query)

        print(f"\n📘 Respuesta:\n{respuesta}\n")


if __name__ == "__main__":
    if not os.path.exists(f"{INDEX_FAISS}/index.faiss"):
        print("❌ ERROR: No se encontró el índice FAISS. Ejecuta primero build_faiss.py.")
        sys.exit(1)

    db = cargar_faiss()
    agente = crear_agente(db)
    hacer_pregunta(agente)
