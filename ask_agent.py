from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
import torch

import os
import sys
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === CONFIGURACIÓN ===
INDEX_FAISS = "faiss_tspec"
CARPETA_MARKDOWN = "TSpec-LLM/3GPP-clean/Rel-15"
MODELO_EMBEDDINGS = "BAAI/bge-large-en-v1.5"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("❌ ERROR: No se encontró OPENAI_API_KEY en el archivo .env.")
    sys.exit(1)

def cargar_faiss():
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️ Usando dispositivo: {device}")
    print("📂 Cargando FAISS existente...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDINGS,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    db = FAISS.load_local(INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
    return db

# === Función para cargar y dividir documentos para BM25 ===
def cargar_documentos_para_bm25():
    print("📄 Cargando documentos para BM25...")
    loader = DirectoryLoader(CARPETA_MARKDOWN, glob="**/*.md", loader_cls=TextLoader, show_progress=True)
    documentos = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_documents(documentos)
    print(f"✅ {len(chunks)} fragmentos listos para BM25.")
    return chunks

# === Función para crear retriever híbrido ===
def crear_retriever_hibrido(faiss_db, documentos_bm25):
    print("🔀 Combinando FAISS + BM25 retrievers...")
    faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 150})
    bm25_retriever = BM25Retriever.from_documents(documentos_bm25)
    bm25_retriever.k = 150
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )

# === Crear agente híbrido ===
def crear_agente(faiss_db):
    print("🤖 Inicializando agente híbrido con reranker...")
    documentos_bm25 = cargar_documentos_para_bm25()
    retriever_hibrido = crear_retriever_hibrido(faiss_db, documentos_bm25)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return llm, retriever_hibrido, reranker


def hacer_pregunta(llm, retriever, reranker):
    prompt_template = PromptTemplate.from_template(
        """Use the following technical context to answer the question.
Only respond with information found in the context.
If the answer is not in the context, say "The answer is not available in the provided information."

Context:
---------
{context}

Question:
---------
{question}

Answer:"""
    )

    print("\n💬 Agente listo. Escribe tu pregunta técnica (o 'salir'):")
    while True:
        query = input("🧠> ").strip()
        if query.lower() == "salir":
            break

        docs = retriever.invoke(query)
        
        # Aplicar reranking a los top-K documentos
        doc_scores = reranker.predict([[query, doc.page_content] for doc in docs])
        scored_docs = sorted(zip(docs, doc_scores), key=lambda x: x[1], reverse=True)
        
        # Seleccionar los top-5 mejor rankeados
        top_docs = [doc.page_content for doc, _ in scored_docs[:5]]
        contexto = "\n\n".join([d[:2000] for d in top_docs])  # Limitar longitud
        
        #contexto = "\n\n".join([doc.page_content[:2000] for doc in docs[:5]])

        prompt_final = prompt_template.format(context=contexto, question=query)

        for i, (doc, score) in enumerate(scored_docs[:5]):
            print(f"[{i+1}] Score: {score:.4f}\n{doc.page_content[:300]}\n")
        
        print("\n📄 Prompt enviado al LLM:\n")
        print(prompt_final)

        respuesta = llm.invoke(prompt_final)

        print("\n📘 Respuesta:\n")
        print(respuesta.content)
        print()

if __name__ == "__main__":
    if not os.path.exists(f"{INDEX_FAISS}/index.faiss"):
        print("❌ ERROR: No se encontró el índice FAISS. Ejecuta primero build_faiss.py.")
        sys.exit(1)

    db = cargar_faiss()
    llm, retriever, reranker = crear_agente(db)
    hacer_pregunta(llm, retriever, reranker)
