# 🛰️ Agente RAG para documentación 3GPP (TSpec-LLM)

Este proyecto implementa un agente de preguntas y respuestas técnicas usando RAG (Retrieval-Augmented Generation) sobre las especificaciones técnicas del estándar 3GPP, utilizando documentos en formato Markdown extraídos del dataset [TSpec-LLM](https://huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM).

---

## 🚀 ¿Qué hace este agente?

- ✅ Indexa especificaciones 3GPP Release 15 (u otras) usando FAISS
- ✅ Divide los documentos en fragmentos optimizados para recuperación
- ✅ Genera embeddings con modelos de Hugging Face (`MiniLM`)
- ✅ Recupera contexto relevante y responde usando `GPT-4` u otro LLM
- ✅ Funciona localmente o puede conectarse a OpenAI

---

## 📦 Requisitos

- Python 3.10 o superior
- Pip

Instalación de dependencias recomendadas:

```bash
pip install -U \
    langchain \
    langchain-community \
    langchain-core \
    langchain-openai \
    langchain-huggingface \
    sentence-transformers \
    faiss-cpu \
    python-dotenv \
    tqdm
```

---

## 🔐 Configuración del entorno

Crea un archivo `.env` en la raíz del proyecto:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 🧠 ¿Cómo se ejecuta?

1. Coloca los archivos `.md` del dataset 3GPP en la ruta definida en `TSpec-LLM/3GPP-clean/Rel-15`.
2. Construye el índice:

```bash
python build_faiss.py
```

3. Ejecuta el agente:

```bash
python ask_agent.py
```

4. Interactúa con el agente por consola:


---

## 🛠 Estructura de los scripts

- **build_faiss.py**
  - `cargar_documentos()` → Lee todos los archivos Markdown
  - `dividir_en_chunks()` → Divide el contenido en fragmentos de 1000 caracteres
  - `construir_faiss()` → Genera los embeddings y construye el índice FAISS
- **ask_agent.py**
  - `cargar_faiss()` → Carga el índice previamente guardado
  - `crear_agente()` → Crea el `RetrievalQA` con el modelo `ChatOpenAI`
  - `hacer_pregunta()` → Interfaz conversacional por consola

---

## ⚠️ Notas y advertencias

- Al usar modelos de OpenAI, asegúrate de no exceder tus límites de tokens.
- La carga de embeddings puede tomar varios minutos si hay muchos documentos (~700k fragmentos en Release 15).
- Para permitir deserialización segura de tu índice FAISS local, asegúrate de agregar `allow_dangerous_deserialization=True` al cargar el índice.

---

## ✨ Futuras mejoras sugeridas

- Interfaz web con `Gradio` o `Streamlit`
- Integración con modelos locales (`Ollama`, `LLaMA`, etc.)
- Soporte para múltiples versiones (`Rel-15`, `Rel-16`, `Rel-17`...)
- Exportar preguntas y respuestas a CSV

---

## 📄 Licencia

Este proyecto usa datos de TSpec-LLM, con licencia [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). El uso de modelos GPT de OpenAI está sujeto a sus [términos de uso](https://openai.com/policies/usage-policies).
