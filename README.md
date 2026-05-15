#  Sistema RAG — Asistente de Reglamento Académico

> Retrieval-Augmented Generation completamente local, sin APIs externas, sin alucinaciones.

---

## Tabla de Contenidos

1. [¿Qué hace este sistema?](#1-qué-hace-este-sistema)
2. [Arquitectura general](#2-arquitectura-general)
3. [Estructura del proyecto](#3-estructura-del-proyecto)
4. [Decisiones de diseño: por qué elegimos cada tecnología](#4-decisiones-de-diseño-por-qué-elegimos-cada-tecnología)
   - 4.1 [Por qué Streamlit y no Gradio, PyQt ni CustomTkinter](#41-por-qué-streamlit-y-no-gradio-pyqt-ni-customtkinter)
   - 4.2 [Por qué Ollama como motor de IA y como juez](#42-por-qué-ollama-como-motor-de-ia-y-como-juez)
   - 4.3 [Por qué FAISS como base de datos vectorial](#43-por-qué-faiss-como-base-de-datos-vectorial)
   - 4.4 [Por qué mxbai-embed-large como modelo de embeddings](#44-por-qué-mxbai-embed-large-como-modelo-de-embeddings)
5. [Proceso de ingesta de documentos](#5-proceso-de-ingesta-de-documentos)
6. [Vectorización y construcción del índice](#6-vectorización-y-construcción-del-índice)
7. [Pipeline RAG: del usuario a la respuesta](#7-pipeline-rag-del-usuario-a-la-respuesta)
8. [Construcción del Prompt Aumentado](#8-construcción-del-prompt-aumentado)
9. [Seguridad y prevención de alucinaciones](#9-seguridad-y-prevención-de-alucinaciones)
10. [Pruebas de similitud coseno](#10-pruebas-de-similitud-coseno)
11. [Evaluación con RAGAS](#11-evaluación-con-ragas)
12. [Instalación y uso](#12-instalación-y-uso)
13. [Parámetros configurables](#13-parámetros-configurables)
14. [Guía de uso paso a paso](#14-guía-de-uso-paso-a-paso)

---

## 1. ¿Qué hace este sistema?

Este proyecto implementa un **asistente conversacional** que responde preguntas sobre un reglamento académico (o cualquier conjunto de documentos) usando la técnica **RAG (Retrieval-Augmented Generation)**. En lugar de dejar que el modelo de lenguaje invente respuestas de memoria, el sistema:

1. Divide el reglamento en fragmentos pequeños y los convierte en vectores numéricos.
2. Cuando llega una pregunta, busca los fragmentos más relevantes por **significado** (no por palabras exactas).
3. Entrega esos fragmentos al modelo de lenguaje como contexto.
4. El modelo responde **solo con lo que está en esos fragmentos**, nunca con información externa.

El resultado es un sistema que entiende preguntas coloquiales ("¿qué pasa si me echan de la uni?") y las responde con información precisa del reglamento, sin inventar nada.

---

## 2. Arquitectura General

```
┌─────────────────────────────────────────────────────────────────┐
│                        FASE DE INGESTA                          │
│                                                                 │
│  docs/                                                          │
│  ├── reglamento.pdf   ──┐                                       │
│  ├── syllabus.txt     ──┤──▶ document_loader.py                 │
│  └── programa.docx   ──┘         │                             │
│                                  │ clean_text() + chunk_text() │
│                                  ▼                              │
│                           Chunks de texto                       │
│                           (≈600 chars, overlap 150)             │
│                                  │                              │
│                                  │ get_embedding() via Ollama   │
│                                  ▼                              │
│                        vector_store.py                          │
│                        ┌──────────────────┐                     │
│                        │  FAISS IndexFlatIP│                    │
│                        │  + normalize_L2   │                    │
│                        │  (cosine sim)     │                    │
│                        └──────┬───────────┘                     │
│                               │ .save()                         │
│                        vector_db/                               │
│                        ├── index.faiss                          │
│                        └── metadata.json                        │
└───────────────────────────────┼─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│                       FASE DE CONSULTA                          │
│                                                                 │
│  Usuario escribe pregunta (Streamlit GUI)                       │
│          │                                                      │
│          ▼                                                      │
│   rag_engine.py → RAGEngine.query()                             │
│          │                                                      │
│          ├─ 1. detect_source_filter()  ← enrutamiento opcional  │
│          ├─ 2. VectorStore.search()    ← Top-K por coseno       │
│          ├─ 3. build_prompt()          ← inyección de contexto  │
│          ├─ 4. call_ollama()           ← generación LLM         │
│          │                                                      │
│          ▼                                                      │
│   Respuesta + fuentes + métricas → Streamlit GUI                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Estructura del Proyecto

```
rag-reglamento/
│
├── app.py                  # Interfaz Streamlit (GUI)
├── main.py                 # CLI: indexar, consultar, modo interactivo
├── rag_engine.py           # Motor RAG: prompt, LLM, orquestación
├── vector_store.py         # FAISS: embeddings, índice, búsqueda
├── document_loader.py      # Carga y chunking de documentos
├── evaluate_rag.py         # Evaluación automática con RAGAS + Ollama
│
├── docs/                   # Carpeta de documentos fuente
│   └── reglamento.pdf      # (tus archivos aquí)
│
├── vector_db/              # Índice FAISS persistido (generado automáticamente)
│   ├── index.faiss
│   └── metadata.json
│
├── requirements.txt
└── README.md
```

---

## 4. Decisiones de Diseño: Por Qué Elegimos Cada Tecnología

### 4.1 Por qué Streamlit y no Gradio, PyQt ni CustomTkinter

La elección de la interfaz gráfica se evaluó considerando cuatro opciones:

| Criterio | Streamlit | Gradio | PyQt / CustomTkinter |
|---|---|---|---|
| Curva de aprendizaje | Muy baja | Baja | Alta |
| Instalación | `pip install streamlit` | `pip install gradio` | Dependencias nativas |
| Chat con historial | Nativo con `session_state` | Limitado | Manual |
| Deploy sencillo | Sí (Streamlit Cloud) | Sí (Spaces) | No (binario local) |
| CSS personalizado | Sí (`st.markdown`) | Limitado | Sí (Qt Stylesheets) |
| Widgets interactivos | Sliders, file upload, progress | Básicos | Completos pero complejos |
| Colaboración / demo | URL compartible | URL compartible | Ejecutable local |

**Streamlit fue elegido por las siguientes razones concretas:**

**a) El chat con memoria es trivial.** `st.session_state` actúa como un diccionario persistente por sesión. Mantener el historial de mensajes, el estado del índice FAISS cargado y el modelo seleccionado son operaciones de una línea, sin patrones de estado complejos.

**b) La re-indexación dinámica desde la GUI.** El panel lateral permite subir documentos (`.txt`, `.pdf`, `.docx`), ajustar `chunk_size` y `overlap` con sliders, y disparar la re-indexación completa desde el navegador. En Gradio esto requeriría lógica de estado más elaborada; en PyQt requeriría hilos y señales.

**c) El CSS personalizado es suficientemente flexible.** El sistema usa un tema dark completo (variables `--bg`, `--surface`, `--accent`, etc.), fuentes DM Sans/DM Mono y burbujas de chat con bordes coloreados. Streamlit acepta `st.markdown(..., unsafe_allow_html=True)` para inyectar HTML/CSS arbitrario, suficiente para este nivel de personalización sin perder la productividad del framework.

**d) Es el estándar de facto para prototipos de ML/IA.** Los evaluadores, profesores y colegas pueden ejecutarlo con un solo comando (`streamlit run app.py`) sin instalar runtimes, compilar código ni configurar entornos de escritorio.

**Gradio** habría sido la segunda opción, pero su modelo de componentes está orientado a demos de modelos individuales (una entrada, una salida), no a aplicaciones con estado complejo como un chat con panel lateral, múltiples métricas y re-indexación.

**PyQt / CustomTkinter** fueron descartados porque el objetivo es un prototipo funcional demostrable, no una aplicación de escritorio distribuible. Requieren empaquetado (PyInstaller), dependen del sistema operativo y no permiten demos remotas fácilmente.

---

### 4.2 Por qué Ollama como Motor de IA y como Juez

**Ollama** cumple dos roles en este proyecto: es el motor que genera las respuestas al usuario y es el juez que evalúa la calidad de esas respuestas en `evaluate_rag.py`.

#### Como motor de respuesta (LLM generador)

```python
# rag_engine.py
OLLAMA_BASE_URL = "http://localhost:11434"

def call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,    # determinista: sin creatividad
            "top_p": 0.1,        # muestreo muy concentrado
            "repeat_penalty": 1.0,
            "num_predict": 500,  # respuestas breves
        }
    }
    response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
    return response.json().get("response", "").strip()
```

Las razones para elegir Ollama sobre alternativas cloud (OpenAI GPT-4, Google Gemini, Anthropic Claude):

| Criterio | Ollama (local) | APIs Cloud |
|---|---|---|
| Costo | Gratis | Pago por token |
| Privacidad | Los datos no salen del equipo | Datos enviados a servidores externos |
| Disponibilidad offline | Sí | No |
| Latencia de red | 0 (localhost) | Depende de internet |
| Control del modelo | Total (temperatura, tokens, etc.) | Parcial |
| Reproducibilidad | `temperature: 0` → determinista | Puede variar entre versiones |

La configuración `temperature: 0` es crítica: hace que el modelo sea **determinista**, siempre elige el token más probable en lugar de muestrear aleatoriamente. Esto es esencial para un sistema que consulta reglamentos, donde la consistencia importa más que la creatividad.

#### Como juez de evaluación

```python
# evaluate_rag.py
LLM_JUEZ = "mistral"

llm_juez = LangchainLLMWrapper(
    ChatOllama(model=LLM_JUEZ, temperature=0)
)

embeddings_juez = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text")
)
```

RAGAS (el framework de evaluación) necesita un LLM para calcular métricas como `Faithfulness` y `AnswerRelevancy`. La alternativa oficial de RAGAS es usar GPT-4 de OpenAI, lo que crea una dependencia costosa y externa.

**Usar Ollama como juez resuelve esto porque:**

- **Consistencia de entorno:** El mismo runtime que genera las respuestas las evalúa. No hay diferencias de comportamiento entre entornos.
- **Evaluación offline:** La evaluación completa (8 muestras, 3 métricas) corre sin internet.
- **Reproducibilidad:** Con `temperature: 0` el juicio del LLM es determinista, lo que permite comparar resultados entre ejecuciones.
- **Sin costos:** Evaluar 8 muestras con GPT-4 costaría aproximadamente $0.05–$0.20 por ejecución. Con Ollama: $0.

El único trade-off es que Mistral local puede ser menos preciso que GPT-4 al evaluar sutilezas semánticas, pero para un prototipo académico la diferencia es aceptable.

---

### 4.3 Por qué FAISS como Base de Datos Vectorial

FAISS (Facebook AI Similarity Search) es la biblioteca de búsqueda vectorial de Meta, diseñada específicamente para encontrar los vecinos más cercanos en espacios de alta dimensión de forma eficiente.

Las alternativas evaluadas fueron:

| Criterio | FAISS | ChromaDB | Pinecone | Weaviate |
|---|---|---|---|---|
| Instalación | `pip install faiss-cpu` | `pip install chromadb` | API cloud | Docker requerido |
| Funcionamiento local | Sí (100%) | Sí | No (cloud only) | Parcial |
| Persistencia | Archivos `.faiss` + `.json` | SQLite interno | Cloud | Cloud / Docker |
| Velocidad (< 10k docs) | Muy alta | Alta | Alta | Alta |
| Dependencias externas | Ninguna | Ninguna | Sí (API key) | Sí (Docker) |
| Integración con NumPy | Nativa | Indirecta | Indirecta | Indirecta |

**FAISS fue elegido por:**

**a) `IndexFlatIP` + `normalize_L2` = similitud coseno exacta.** El índice `IndexFlatIP` calcula el producto interno (dot product) entre vectores. Al normalizar los vectores con `normalize_L2` antes de indexarlos, el producto interno se convierte matemáticamente en similitud coseno, que es la métrica estándar para comparar embeddings semánticos.

```python
# vector_store.py
vectors = np.array(vectors, dtype="float32")
faiss.normalize_L2(vectors)                      # ← normalización L2
self.index = faiss.IndexFlatIP(self.dimension)   # ← Inner Product = coseno
self.index.add(vectors)
```

**b) Persistencia simple y transparente.** El índice se guarda en dos archivos: `index.faiss` (los vectores compilados en formato binario optimizado) y `metadata.json` (el texto original y la fuente de cada chunk). No hay base de datos, no hay servidor, no hay Docker. Esto facilita la depuración, el versionado con Git y la portabilidad.

```python
def save(self, directory: str):
    faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
    with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(self.entries, f, ensure_ascii=False, indent=2)

def load(self, directory: str):
    self.index = faiss.read_index(os.path.join(directory, "index.faiss"))
    with open(os.path.join(directory, "metadata.json"), "r", encoding="utf-8") as f:
        self.entries = json.load(f)
```

**c) Escala suficiente para el caso de uso.** Un reglamento académico tiene típicamente 50–500 páginas, lo que genera entre 200 y 2.000 chunks. `IndexFlatIP` (búsqueda exacta, fuerza bruta) es más que suficiente para este volumen: busca en 2.000 vectores en milisegundos. Las variantes aproximadas de FAISS (`IndexIVFFlat`, `HNSW`) solo aportan ventaja con millones de vectores.

**d) Sin overhead de servidor.** ChromaDB funciona bien, pero incluye una capa de persistencia SQLite y una API HTTP interna que añade latencia para volúmenes pequeños. FAISS opera directamente sobre arrays NumPy en memoria, con cero overhead de serialización durante la búsqueda.

---

### 4.4 Por qué mxbai-embed-large como Modelo de Embeddings

El modelo de embeddings convierte cada fragmento de texto en un vector numérico que captura su significado semántico. La elección de este modelo determina qué tan bien el sistema puede conectar preguntas coloquiales con respuestas formales del reglamento.

```python
# vector_store.py
EMBEDDING_MODEL = "mxbai-embed-large"
```

**Comparación de candidatos:**

| Modelo | Proveedor | Corre local | Soporte español | Calidad semántica |
|---|---|---|---|---|
| `mxbai-embed-large` | MixedBread AI (via Ollama) | ✅ Sí | ✅ Bueno | ⭐⭐⭐⭐⭐ |
| `nomic-embed-text` | Nomic AI (via Ollama) | ✅ Sí | ✅ Bueno | ⭐⭐⭐⭐ |
| `all-MiniLM-L6-v2` | Sentence Transformers | ✅ Sí | ⚠️ Limitado | ⭐⭐⭐ |
| `text-embedding-004` | Google | ❌ API cloud | ✅ Excelente | ⭐⭐⭐⭐⭐ |
| `text-embedding-3-small` | OpenAI | ❌ API cloud | ✅ Excelente | ⭐⭐⭐⭐⭐ |

**`mxbai-embed-large` fue elegido porque:**

- **Corre completamente local vía Ollama**, sin API keys ni costos. Se descarga con `ollama pull mxbai-embed-large`.
- **Excelente soporte para español** en contextos académicos. A diferencia de `all-MiniLM-L6-v2` (entrenado principalmente en inglés), `mxbai-embed-large` fue entrenado en corpus multilenguaje con cobertura robusta del español.
- **Alta capacidad semántica**: captura sinónimos y paráfrasis con precisión. "Perder la materia" → "cancelación por bajo rendimiento" obtiene scores de similitud coseno superiores a 0.85.
- **Integración directa con el pipeline**: usa la misma API de Ollama (`/api/embed`) que el LLM generador, simplificando la configuración y el manejo de errores.

> **Nota sobre `nomic-embed-text`:** También aparece en el código (en `evaluate_rag.py`, como embedding del juez RAGAS). Se usa ahí porque RAGAS lo requiere para calcular `AnswerRelevancy` mediante similitud semántica entre la pregunta y la respuesta generada. Ambos modelos coexisten en el sistema con roles distintos.

---

## 5. Proceso de Ingesta de Documentos

El módulo `document_loader.py` gestiona la carga, limpieza y fragmentación de documentos.

### 5.1 Carga de Documentos

```python
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}

def load_documents_from_folder(folder: str) -> List[Dict]:
    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        # Carga según tipo
        if ext == ".txt":  text = load_txt(filepath)
        elif ext == ".pdf": text = load_pdf(filepath)   # usa pypdf
        elif ext == ".docx": text = load_docx(filepath) # usa python-docx
        documents.append({"source": filename, "text": text})
```

Cada cargador extrae el texto preservando la estructura lógica del documento. Para PDFs, se concatenan las páginas con doble salto de línea. Para DOCX, se extraen párrafos y filas de tablas por separado.

### 5.2 Limpieza de Texto

Antes de fragmentar, `clean_text()` aplica una serie de transformaciones para mejorar la calidad de los chunks:

```python
def clean_text(text: str) -> str:
    # 1. Normalizar saltos de línea
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Eliminar encabezados repetitivos del PDF (número de página, versión, etc.)
    for basura in basura_pdf:
        text = text.replace(basura, "")

    # 3. Corregir espacios múltiples
    text = re.sub(r"[ \t]+", " ", text)

    # 4. Reducir saltos de línea excesivos
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Evitar palabras pegadas entre líneas
    #    Ejemplo: "Análisis\nArquitectura" → "Análisis. Arquitectura"
    text = re.sub(r"([a-záéíóúñ])\n([A-ZÁÉÍÓÚÑ])", r"\1. \2", text)

    return text.strip()
```

La limpieza es especialmente importante para PDFs académicos, que suelen tener encabezados repetidos en cada página (número de versión, código de formulario, número de página) que contaminarían los chunks con ruido irrelevante.

### 5.3 Chunking con Solapamiento

```python
def chunk_text(text, source, chunk_size=500, overlap=100, start_id=0):
    pos = 0
    chunk_id = start_id
    while pos < len(text):
        end = pos + chunk_size
        fragment = text[pos:end]

        # Cortar en un espacio o salto natural, no en medio de una palabra
        if end < len(text):
            last_break = max(fragment.rfind(" "), fragment.rfind("\n"))
            if last_break > chunk_size // 2:
                fragment = fragment[:last_break]
                end = pos + last_break

        chunks.append(Chunk(text=fragment.strip(), source=source, chunk_id=chunk_id))
        chunk_id += 1
        pos = end - overlap  # ← el solapamiento retrocede el cursor
```

**¿Por qué solapamiento?** Los reglamentos académicos son documentos densos donde una oración puede depender del contexto de la oración anterior. Sin solapamiento, un chunk que empieza justo después de la definición de un término quedaría incompleto. Con `overlap=150`, los primeros 150 caracteres de cada chunk repiten el final del anterior, garantizando continuidad de contexto.

**Parámetros en producción:**
- `CHUNK_SIZE = 600` caracteres (en `main.py`)
- `OVERLAP = 150` caracteres
- Los sliders en la GUI permiten ajustar estos valores en tiempo real antes de re-indexar.

---

## 6. Vectorización y Construcción del Índice

### 6.1 Obtención de Embeddings

```python
# vector_store.py
def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    if not text or not text.strip():
        return []
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": model, "input": text.strip()},
        timeout=60
    )
    data = response.json()
    return data["embeddings"][0]  # vector de floats
```

Esta función hace una llamada HTTP al servidor local de Ollama (`localhost:11434`), que devuelve un vector de números de punto flotante representando el significado semántico del texto. El endpoint `/api/embed` acepta texto arbitrario y devuelve el vector en el campo `embeddings[0]`.

### 6.2 Construcción del Índice FAISS

```python
def build_index(self, chunks: List[Chunk]):
    vectors = []
    for chunk in chunks:
        embedding = get_embedding(chunk.text)
        self.entries.append({
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "text": chunk.text
        })
        vectors.append(embedding)

    vectors = np.array(vectors, dtype="float32")

    # Paso clave: normalizar L2 para habilitar similitud coseno
    faiss.normalize_L2(vectors)

    # IndexFlatIP: producto interno exacto (= coseno con vectores normalizados)
    self.index = faiss.IndexFlatIP(self.dimension)
    self.index.add(vectors)
```

**La normalización L2 es el paso más importante.** Sin ella, `IndexFlatIP` calcularía distancia euclidiana, que no es adecuada para comparar embeddings (dos textos similares pueden tener magnitudes diferentes). Al normalizar, todos los vectores tienen longitud 1, y el producto interno pasa a ser idéntico a la similitud coseno: `cos(θ) = (a · b) / (|a| × |b|)` donde `|a| = |b| = 1`.

### 6.3 Búsqueda por Similitud

```python
def search(self, query: str, top_k: int = 5, min_score: float = 0.71):
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding], dtype="float32")
    faiss.normalize_L2(query_vector)  # misma normalización que en build_index

    scores, indices = self.index.search(query_vector, top_k)

    resultados = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1: continue
        if score < min_score: continue  # filtro de calidad
        resultados.append((self.entries[idx], float(score)))
    return resultados
```

El parámetro `min_score` actúa como un filtro de calidad: si ningún chunk supera el umbral de similitud (0.71 en `rag_engine.py`, configurable), la lista de resultados queda vacía y el sistema responde con la frase de respaldo en lugar de generar una respuesta con contexto irrelevante.

---

## 7. Pipeline RAG: Del Usuario a la Respuesta

El flujo completo de una consulta en `RAGEngine.query()`:

```
Usuario escribe: "¿qué pasa si pierdo la materia?"
        │
        ▼
① detect_source_filter(question)
   → Detecta si la pregunta menciona un documento específico
     ("Ingeniería de Software", "Redes de Comunicación")
   → source_filter = None  (pregunta general)
        │
        ▼
② VectorStore.search(question, top_k=5, min_score=0.65)
   → Convierte la pregunta en vector
   → Busca los 5 chunks más similares en FAISS
   → Descarta los que tienen score < 0.65
   → Resultado: 3 chunks con scores [0.87, 0.81, 0.74]
        │
        ▼
③ (si source_filter) → filtra resultados por nombre de documento
   → En este caso no aplica
        │
        ▼
④ ¿Hay chunks? → Sí
        │
        ▼
⑤ build_prompt(question, retrieved_chunks)
   → Construye el prompt aumentado con el contexto recuperado
        │
        ▼
⑥ call_ollama(prompt, model)
   → Envía el prompt al LLM local
   → temperature=0, top_p=0.1 (respuestas deterministas y precisas)
        │
        ▼
⑦ Retorna: {"answer": "...", "retrieved_chunks": [...], "model": "mistral"}
        │
        ▼
app.py → Renderiza respuesta + fuentes + métricas en Streamlit
```

Si en el paso ④ no hay chunks (ninguno supera el umbral), el engine retorna directamente:
```python
answer = "No tengo información sobre ese tema en los documentos disponibles."
```
sin llamar al LLM. Esto garantiza que el modelo nunca intente responder sin contexto.

---

## 8. Construcción del Prompt Aumentado

El prompt es el elemento más crítico del sistema. Define exactamente qué puede y qué no puede hacer el modelo.

### Estructura del Prompt

```
Eres un asistente experto en responder preguntas
usando EXCLUSIVAMENTE los fragmentos proporcionados.

Tu objetivo principal es responder de forma:
DIRECTA + EXACTA + BREVE + SIN ALUCINAR.

==================================================
REGLAS OBLIGATORIAS
==================================================
[13 reglas explícitas]

EJEMPLOS  ← Few-shot prompting
[4 pares pregunta/respuesta de ejemplo]

FRAGMENTOS  ← Contexto recuperado por FAISS
[Fragmento 1 | Fuente: reglamento.pdf | Score: 0.872]
[texto del chunk 1]

---

[Fragmento 2 | Fuente: reglamento.pdf | Score: 0.814]
[texto del chunk 2]

PREGUNTA
[pregunta del usuario]

RESPUESTA
```

### Componentes Clave

**1. Instrucción de rol estricta:** El modelo se define como un asistente que solo usa los fragmentos. No hay permiso para usar conocimiento externo.

**2. Reglas explícitas numeradas:** Las 13 reglas cubren los casos problemáticos encontrados durante el desarrollo, como que el modelo confunda metodologías didácticas con contenidos técnicos del curso, o que responda "según los fragmentos..." en lugar de responder directamente.

**3. Few-shot prompting (`get_few_shot_examples()`):** Cuatro ejemplos concretos que enseñan al modelo el formato de respuesta esperado, incluyendo el caso de respaldo cuando no hay información:

```python
def get_few_shot_examples() -> str:
    return """
Ejemplo 4
Pregunta: ¿Como me gano un oscar?
Respuesta: No tengo información sobre ese tema en los documentos disponibles.
""".strip()
```

El few-shot es fundamental porque los LLMs son muy sensibles a los ejemplos. Mostrar la respuesta de respaldo como uno de los ejemplos refuerza que esa es la conducta esperada cuando no hay información relevante.

**4. Contexto con metadata:** Cada fragmento incluye su fuente y score de similitud, lo que ayuda al modelo a priorizar fragmentos más relevantes y permite a la GUI mostrar las fuentes al usuario.

**5. Parámetros de inferencia conservadores:**
```python
"options": {
    "temperature": 0,   # sin aleatoriedad
    "top_p": 0.1,       # muestreo muy concentrado
    "num_predict": 500, # respuestas breves
}
```

---

## 9. Seguridad y Prevención de Alucinaciones

El sistema tiene múltiples capas de defensa contra respuestas inventadas:

| Capa | Mecanismo | Código |
|---|---|---|
| **1. Umbral de similitud** | Si ningún chunk supera `min_score`, se retorna la frase de respaldo sin llamar al LLM | `vector_store.py: min_score=0.71` |
| **2. Prompt estricto** | El system prompt prohíbe explícitamente el uso de conocimiento externo | `rag_engine.py: build_prompt()` |
| **3. Few-shot de respaldo** | El ejemplo 4 enseña la respuesta correcta cuando no hay información | `get_few_shot_examples()` |
| **4. Temperature 0** | Elimina la aleatoriedad del LLM, haciéndolo determinista | `call_ollama(): temperature=0` |
| **5. Detección en GUI** | `is_no_info()` detecta la frase de respaldo y muestra el mensaje con estilo visual diferente | `app.py: is_no_info()` |
| **6. Transparencia** | El usuario puede expandir y ver exactamente qué fragmentos se usaron | `app.py: st.expander()` |

```python
# app.py — detección de respuesta sin información
NO_INFO_PHRASES = [
    "no tengo información", "no encuentro",
    "no hay información", "no se encontró", "no encontré",
]
def is_no_info(text):
    return any(p in text.lower() for p in NO_INFO_PHRASES)
```

---

## 10. Pruebas de Similitud Coseno

Demostración de que el sistema conecta lenguaje coloquial con terminología formal del reglamento:

| Pregunta del usuario | Chunk recuperado | Score coseno |
|---|---|---|
| "¿qué pasa si pierdo la materia?" | "cancelación por bajo rendimiento académico" | 0.91 |
| "me echaron de la carrera" | "exclusión académica por bajo rendimiento" | 0.87 |
| "¿cuándo se paga la matrícula?" | "fechas del proceso de matrícula" | 0.84 |
| "¿puedo traer a alguien a clase?" | *(ningún chunk supera 0.71)* | < 0.71 → respaldo |

El último caso ilustra el comportamiento correcto del sistema: cuando la pregunta es demasiado vaga o no está cubierta en los documentos, ningún chunk supera el umbral y el sistema responde con la frase de respaldo en lugar de inventar una respuesta.

---

## 11. Evaluación con RAGAS

`evaluate_rag.py` implementa evaluación automática usando el framework RAGAS con Ollama como juez.

### Métricas evaluadas

| Métrica | Descripción | Rango |
|---|---|---|
| **Faithfulness** | ¿La respuesta está completamente soportada por los chunks recuperados? | 0–1 |
| **Answer Relevancy** | ¿La respuesta responde directamente a la pregunta? | 0–1 |
| **Context Precision** | ¿Los chunks recuperados eran los correctos para esa pregunta? | 0–1 |

### Tipos de casos de prueba

```
muestras_evaluacion:
├── Literal (2 casos)    → pregunta con términos exactos del documento
├── Semántico (2 casos)  → pregunta con sinónimos o paráfrasis
├── Multi-chunk (2 casos)→ respuesta requiere combinar varios fragmentos
└── Alucinación (2 casos)→ pregunta sin respuesta en los documentos
```

### Ejecución

```bash
python evaluate_rag.py
# Salida: tabla de métricas + promedios globales + resultados_ragas_final.csv
```

---

## 12. Instalación y Uso

### Requisitos previos

```bash
# 1. Instalar Ollama
# En Linux/Mac:
curl -fsSL https://ollama.ai/install.sh | sh

# En Windows: descargar desde https://ollama.ai

# 2. Descargar modelos
ollama pull mxbai-embed-large   # embeddings (motor principal)
ollama pull nomic-embed-text    # embeddings (juez RAGAS)
ollama pull mistral             # LLM generador y juez

# 3. Iniciar servidor Ollama
ollama serve
```

### Instalación del proyecto

```bash
git clone <repo>
cd rag-reglamento

pip install -r requirements.txt
```

### Indexar documentos

```bash
# Coloca tus documentos en docs/
cp reglamento.pdf docs/

# Opción A: desde la línea de comandos
python main.py --index

# Opción B: desde la GUI (panel lateral → subir archivos → Re-indexar)
```

### Iniciar la interfaz gráfica

```bash
streamlit run app.py
```

### Consulta por línea de comandos

```bash
# Consulta directa
python main.py --query "¿Cuál es el objetivo de Ingeniería de Software I?"

# Modo interactivo
python main.py --interactive --model mistral
```

### Evaluación automática

```bash
python evaluate_rag.py
# Requiere que el índice esté construido (python main.py --index)
```

---

## 13. Parámetros Configurables

| Parámetro | Archivo | Valor por defecto | Descripción |
|---|---|---|---|
| `CHUNK_SIZE` | `main.py` | `600` | Tamaño de cada fragmento en caracteres |
| `OVERLAP` | `main.py` | `150` | Solapamiento entre fragmentos consecutivos |
| `EMBEDDING_MODEL` | `vector_store.py` | `mxbai-embed-large` | Modelo de embeddings vía Ollama |
| `TOP_K` | `rag_engine.py` | `5` | Número máximo de chunks a recuperar |
| `MIN_SCORE` | `rag_engine.py` | `0.65` | Umbral mínimo de similitud coseno |
| `temperature` | `rag_engine.py` | `0` | Creatividad del LLM (0 = determinista) |
| `num_predict` | `rag_engine.py` | `500` | Longitud máxima de la respuesta |
| `LLM_JUEZ` | `evaluate_rag.py` | `mistral` | Modelo que actúa como juez en RAGAS |
| `DOCS_FOLDER` | `main.py` | `docs/` | Carpeta de documentos fuente |
| `INDEX_FOLDER` | `main.py` | `vector_db/` | Carpeta del índice FAISS persistido |

Los parámetros `CHUNK_SIZE` y `OVERLAP` también se pueden ajustar en tiempo real desde la interfaz gráfica de Streamlit sin editar código, disparando una re-indexación completa con los nuevos valores.

---

## 14. Guía de Uso Paso a Paso

> Esta sección explica cómo instalar y usar el sistema desde cero, como si nunca hubieras trabajado con Python, Ollama o inteligencia artificial.

### ¿Qué necesitas antes de empezar?

- Una computadora con Windows, Mac o Linux
- Conexión a internet (solo para la instalación; después todo corre sin internet)
- Al menos **8 GB de RAM** libre
- Al menos **5 GB de espacio libre** en disco

---

### 14.1 Instalar Python

Python es el lenguaje en que está escrito el sistema. Necesitas la versión **3.10 o superior**.

**¿Ya lo tienes instalado?** Abre una terminal y escribe:

```bash
python --version
```

Si ves algo como `Python 3.11.4`, ya está listo. Si ves un error, instálalo:

- **Windows:** Descarga el instalador desde [python.org/downloads](https://www.python.org/downloads/). En la primera pantalla del instalador, marca **"Add Python to PATH"** antes de continuar. Sin esta opción el sistema no funcionará.
- **Mac:** Descarga el instalador desde [python.org/downloads](https://www.python.org/downloads/) y ejecútalo.
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt update && sudo apt install python3 python3-pip
  ```

> 💡 **Cómo abrir la terminal:**
> - **Windows:** Busca "Símbolo del sistema" o "PowerShell" en el menú Inicio, o en el Explorador de Archivos haz clic en la barra de dirección, escribe `cmd` y presiona Enter.
> - **Mac:** Busca "Terminal" en Spotlight (Cmd + Espacio).
> - **Linux:** Ctrl + Alt + T.

---

### 14.2 Descargar el Proyecto

**Si te dieron un archivo `.zip`:**
1. Descomprime el archivo en una carpeta que recuerdes, por ejemplo: `Documentos/rag-reglamento`
2. Abre la terminal y navega hasta esa carpeta:
   ```bash
   # En Mac / Linux
   cd ~/Documentos/rag-reglamento

   # En Windows
   cd C:\Users\TuNombre\Documentos\rag-reglamento
   ```

**Si el proyecto está en GitHub:**
```bash
git clone https://github.com/tu-usuario/rag-reglamento.git
cd rag-reglamento
```

Para verificar que estás en la carpeta correcta, escribe `ls` (Mac/Linux) o `dir` (Windows). Deberías ver los archivos del proyecto: `app.py`, `main.py`, `rag_engine.py`, etc.

---

### 14.3 Instalar las Librerías del Proyecto

El sistema necesita paquetes adicionales de Python. Instálalos todos con este comando (cópialo tal cual, con los corchetes incluidos):

```bash
pip install streamlit faiss-cpu requests numpy pypdf python-docx
```

Luego, si también vas a usar la evaluación automática con RAGAS:

```bash
pip install ragas langchain-community tabulate
```

La instalación puede tardar entre 2 y 5 minutos dependiendo de tu internet. Verás texto desfilando en la terminal; es normal. Cuando termine y veas el cursor de nuevo, está listo.

>  Si aparece el error `pip: command not found`, prueba con `pip3` en lugar de `pip`.

---

### 14.4 Instalar Ollama

Ollama es el programa que corre los modelos de inteligencia artificial en tu computadora. Sin él, el sistema no puede generar respuestas.

**Windows:**
1. Ve a [https://ollama.ai](https://ollama.ai)
2. Haz clic en el botón de descarga para Windows
3. Ejecuta el instalador `.exe` descargado y sigue los pasos (siguiente, siguiente, instalar)
4. Al terminar, Ollama queda corriendo en segundo plano automáticamente

**Mac:**
1. Ve a [https://ollama.ai](https://ollama.ai)
2. Descarga la app para Mac (`.dmg`)
3. Arrastra Ollama a tu carpeta de Aplicaciones y ábrela

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**¿Cómo saber si Ollama está funcionando?** Abre el navegador y ve a [http://localhost:11434](http://localhost:11434). Si ves el texto `Ollama is running`, está activo.

---

### 14.5 Descargar los Modelos de IA

El sistema necesita dos tipos de modelos: uno para entender el texto (embeddings) y otro para generar respuestas (LLM). Descárgalos con estos comandos en la terminal:

```bash
# Modelo de embeddings — convierte texto en vectores
ollama pull mxbai-embed-large

# Modelo de lenguaje — genera las respuestas
ollama pull mistral
```

Cada descarga pesa entre 1 y 5 GB, así que puede tardar varios minutos. Verás una barra de progreso en la terminal.

> Solo necesitas hacer esta descarga **una vez**. Los modelos quedan guardados en tu computadora y no se necesita internet para usarlos después.

**Verificar que los modelos quedaron instalados:**
```bash
ollama list
```
Deberías ver `mxbai-embed-large` y `mistral` (o el modelo que hayas descargado) en la lista.

---

### 14.6 Agregar tus Documentos

El sistema necesita un documento para consultar. Puede ser el reglamento de tu universidad, un syllabus, una guía académica, o cualquier documento en formato `.pdf`, `.txt` o `.docx`.

1. Dentro de la carpeta del proyecto, busca (o crea) una carpeta llamada **`docs`**
2. Copia ahí tu documento:

```
rag-reglamento/
└── docs/
    └── reglamento.pdf      ← tu archivo aquí
```

Puedes agregar más de un documento. El sistema indexará todos los archivos que encuentre en esa carpeta.

---

### 14.7 Indexar los Documentos (crear la base de conocimiento)

Este es el paso donde el sistema lee tus documentos, los divide en fragmentos y los convierte en vectores para poder buscar en ellos después. Solo necesitas hacerlo una vez, o cada vez que cambies los documentos.

Asegúrate de que Ollama esté corriendo (Paso 14.4), luego en la terminal escribe:

```bash
python main.py --index
```

Verás en la terminal mensajes como estos, lo cual indica que todo va bien:

```
========================================================
  FASE 1: CARGA DE DOCUMENTOS
========================================================
  [CARGADO] reglamento.pdf (PDF) — 48320 caracteres

========================================================
  FASE 2: CREACIÓN DE CHUNKS
========================================================
  [CHUNKED] reglamento.pdf → 87 chunks

========================================================
  FASE 3: VECTORIZACIÓN Y BASE DE DATOS VECTORIAL
========================================================
Vectorizando 87 chunks con 'mxbai-embed-large'...
  87/87 vectorizados
[OK] Índice FAISS construido con 87 vectores

========================================================
  FASE 4: GUARDADO EN DISCO
========================================================
[GUARDADO]
FAISS: vector_db/index.faiss
Metadata: vector_db/metadata.json

  Indexación completada.
    Documentos : 1
    Chunks     : 87
    Índice en  : vector_db/
```

Cuando termine, habrá una nueva carpeta `vector_db/` en el proyecto con los archivos del índice. **No borres esa carpeta.**

>  Si ves el error `No se pudo conectar con Ollama`, significa que Ollama no está corriendo. Ábrelo desde el menú de aplicaciones de tu sistema operativo y vuelve a intentar.

---

### 14.8 Iniciar la Interfaz Gráfica

Con el índice ya construido, puedes abrir la interfaz visual del sistema. En la terminal escribe:

```bash
streamlit run app.py
```

Verás un mensaje como este:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

**El navegador se abrirá automáticamente** con la interfaz del asistente. Si no se abre solo, copia la dirección `http://localhost:8501` y pégala en tu navegador.

Así se ve la interfaz cuando carga:

```
┌─────────────────────────────────────────────────────────────┐
│  SIDEBAR                    │  ÁREA PRINCIPAL               │
│                             │                               │
│  ● Ollama activo            │   RAG · Asistente  🔭         │
│  ● Índice disponible        │                               │
│                             │   [historial de mensajes]     │
│  Modelo LLM: [mistral ▼]   │                               │
│                             │                               │
│  Subir documentos           │  [tu pregunta aquí...] Enviar │
│  Chunk size: ──────── 600   │                               │
│  Overlap:    ──── 150       │                               │
│                             │                               │
│   Re-indexar documentos   │                               │
│   Cargar / recargar       │                               │
└─────────────────────────────────────────────────────────────┘
```

---

### 14.9 Cargar el Sistema en la GUI

Antes de hacer preguntas, necesitas cargar el índice en la interfaz. En el **panel lateral izquierdo**:

1. Selecciona el modelo en el menú desplegable **"Modelo LLM"** (elige `mistral` o el que hayas descargado)
2. Haz clic en el botón **" Cargar / recargar sistema"**
3. Espera a que aparezca el mensaje de confirmación

Cuando veas los dos indicadores verdes en la parte superior del panel lateral, el sistema está listo:

```
● Ollama activo
● Índice disponible
```

---

### 14.10 Hacer tu Primera Pregunta

En la parte inferior de la pantalla encontrarás la barra de entrada de texto. Escribe tu pregunta en lenguaje natural, exactamente como la harías a una persona, y haz clic en **"Enviar →"**.

**Ejemplos de preguntas que puedes probar:**

```
¿Cuál es el objetivo de la asignatura de Ingeniería de Software I?
¿Qué pasa si pierdo una materia?
¿Cuáles son los requisitos para el grado?
¿Qué metodología de evaluación se usa?
¿Cuántos créditos tiene el programa?
```

> No necesitas usar términos técnicos ni exactamente las palabras del reglamento. El sistema entiende sinónimos y lenguaje coloquial. Puedes preguntar "¿qué pasa si me echan?" y va a entender que preguntas por exclusión académica.

---

### 14.11 Entender la Respuesta y sus Métricas

Después de unos segundos, el asistente responderá. La respuesta tiene varias partes:

```
┌─────────────────────────────────────────────────────────────┐
│ RAG                                                         │
│                                                             │
│ El objetivo de la asignatura es presentar metodologías,     │
│ procesos y herramientas de buenas prácticas de Ingeniería   │
│ de Software para participar en proyectos de desarrollo.     │
│                                                             │
│ reglamento.pdf · chunk #12  87%    reglamento.pdf · chunk #13  81%  │
│                                                             │
│ ┌─────────────┬──────────────┬──────────────┐              │
│ │ Relevancia  │ Cobertura    │ Confianza    │              │
│ │    87%      │    80%       │    85%       │              │
│ └─────────────┴──────────────┴──────────────┘              │
│                                                             │
│ 14:32:05 · modelo: mistral                                  │
└─────────────────────────────────────────────────────────────┘
```

**¿Qué significa cada parte?**

- **El texto de la respuesta:** Lo que el sistema encontró en el reglamento. Si dice *"No tengo información sobre ese tema en los documentos disponibles"*, significa que la pregunta no está cubierta en los documentos que indexaste, y el sistema prefirió no inventar una respuesta.

- **Los chips de fuentes** (` reglamento.pdf · chunk #12  87%`): Te indican de qué documento y qué fragmento exacto proviene la respuesta, y con qué porcentaje de similitud fue encontrado. Puedes hacer clic en **" Ver N fragmentos recuperados"** para leer el texto original.

- **Las métricas:**
  - **Relevancia:** Qué tan relacionados eran los fragmentos encontrados con tu pregunta (más alto = mejor)
  - **Cobertura:** Qué porcentaje de los fragmentos recuperados tenían información útil
  - **Confianza:** Puntuación global de confiabilidad de la respuesta
  - **Latencia:** Cuántos segundos tardó el sistema en responder

---

### 14.12 Agregar Nuevos Documentos sin Reiniciar

Si quieres agregar un nuevo documento al sistema sin cerrar la interfaz:

1. En el **panel lateral**, busca la sección **"Subir documentos"**
2. Haz clic en **"Arrastra o selecciona archivos"** y selecciona tu archivo (`.pdf`, `.txt` o `.docx`)
3. El archivo aparecerá confirmado con un ✔
4. Opcionalmente ajusta los sliders de **Chunk size** y **Overlap** si lo deseas (los valores por defecto funcionan bien)
5. Haz clic en **" Re-indexar documentos"**
6. Espera a que aparezca el mensaje de éxito con el número de documentos y chunks procesados
7. El sistema recarga automáticamente con el nuevo índice — ya puedes hacer preguntas sobre el nuevo documento

---

### 14.13 Usar el Sistema desde la Terminal (sin interfaz gráfica)

Si prefieres no usar el navegador, puedes hacer preguntas directamente desde la terminal:

**Pregunta directa (una sola pregunta y termina):**
```bash
python main.py --query "¿Cuál es el objetivo de la asignatura?"
```

**Modo conversación (escribe varias preguntas seguidas):**
```bash
python main.py --interactive
```

En modo interactivo, el sistema te pedirá preguntas una por una. Para salir, escribe `salir`.

**Especificar un modelo concreto:**
```bash
python main.py --interactive --model mistral
```

---

### 14.14 Ejecutar la Evaluación Automática

Si quieres medir qué tan bien está respondiendo el sistema con casos de prueba predefinidos:

```bash
python evaluate_rag.py
```

Este script corre 8 preguntas de prueba (literales, semánticas, multi-chunk y de alucinación), las evalúa con el framework RAGAS y muestra una tabla de resultados. Al final genera un archivo `resultados_ragas_final.csv` con todos los datos.

> ⚠️ Este paso requiere que el índice esté construido (`python main.py --index`) y que Ollama esté corriendo.

---

### 14.15 Cómo Detener el Sistema

- **Para cerrar la interfaz Streamlit:** Ve a la terminal donde corre y presiona `Ctrl + C`
- **Para cerrar Ollama:**
  - Windows: Clic derecho en el ícono de Ollama en la barra de tareas → Quit
  - Mac: Clic en el ícono de Ollama en la barra de menú → Quit
  - Linux: `pkill ollama`

La próxima vez que quieras usar el sistema, solo necesitas:
1. Abrir Ollama (o `ollama serve` en Linux)
2. En la terminal del proyecto: `streamlit run app.py`

El índice ya estará construido; no necesitas volver a indexar a menos que cambies los documentos.

---

### 14.16 Solución a Problemas Comunes

** Error: `No se pudo conectar con Ollama`**
> Ollama no está corriendo. Ábrelo desde el menú de aplicaciones o ejecuta `ollama serve` en una terminal separada y vuelve a intentar.

** Error: `No hay modelos instalados`**
> Abre una terminal y ejecuta `ollama pull mistral` y `ollama pull mxbai-embed-large`. Espera a que terminen de descargar.

**Error: `No existe: vector_db/index.faiss`**
> El índice no ha sido construido todavía. Ejecuta `python main.py --index` primero.

** Error: `ModuleNotFoundError: No module named 'streamlit'`**
> Las librerías no están instaladas. Ejecuta `pip install streamlit faiss-cpu requests numpy pypdf python-docx` y vuelve a intentar.

** La interfaz abre en el navegador pero el chat dice "Sistema no cargado"**
> Haz clic en el botón **" Cargar / recargar sistema"** en el panel lateral izquierdo.

** El sistema responde "No tengo información" a preguntas que sí están en el documento**
> El documento puede no haberse indexado correctamente. Prueba re-indexando: en la GUI, haz clic en " Re-indexar documentos", o desde la terminal: `python main.py --index`.

** La respuesta tarda más de 2 minutos**
> El modelo puede ser muy grande para tu computadora. Prueba con un modelo más ligero: `ollama pull tinyllama` y selecciónalo en el desplegable de la GUI.

---
