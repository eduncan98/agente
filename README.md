# 🧠 TSpec RAG: Sistema de Preguntas Técnicas con FAISS + BM25 + Reranking

Este proyecto implementa un **sistema de recuperación de información híbrido** para responder preguntas técnicas a partir de documentos Markdown (como especificaciones 3GPP). Utiliza:

- 🔍 **FAISS + HuggingFace Embeddings** (`BAAI/bge-large-en-v1.5`)
- 📚 **BM25Retriever**
- 🧠 **CrossEncoder** para reranking (`ms-marco-MiniLM-L-6-v2`)
- 🤖 **ChatOpenAI (GPT-4)** para respuestas generadas con contexto relevante

---

## 📁 Estructura

- `build_faiss.py` → Construye el índice FAISS a partir de los `.md`
- `main.py` → Interfaz interactiva para hacer preguntas
- `.env` → Archivo de variables (necesario `OPENAI_API_KEY`)
- `TSpec-LLM/3GPP-clean/Rel-15/` → Carpeta de entrada con archivos Markdown

---

## ✅ Requisitos

- Python 3.9+
- GPU opcionalmente (`cuda`, `mps`)
- `.env` con tu clave de OpenAI:

### 📦 Dependencias

Instala con:

```bash
pip3 install -r requirements.txt