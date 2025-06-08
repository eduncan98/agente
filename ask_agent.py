from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
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

def cargar_faiss():
    print("📂 Cargando FAISS existente...")
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    db = FAISS.load_local(INDEX_FAISS, embeddings, allow_dangerous_deserialization=True)
    return db

def crear_agente(db):
    print("🤖 Inicializando agente...")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = db.as_retriever(search_kwargs={"k": 15})
    return llm, retriever

def hacer_pregunta(llm, retriever):
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
        contexto = "\n\n".join([doc.page_content[:2000] for doc in docs[:5]])

        prompt_final = prompt_template.format(context=contexto, question=query)

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
    llm, retriever = crear_agente(db)
    hacer_pregunta(llm, retriever)