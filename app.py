"""
app.py — Interfaz Streamlit para sistema RAG local con Ollama.

Uso:
    streamlit run app.py

Requisitos previos:
    1. ollama serve
    2. python main.py --index   (o indexar desde la propia app)
    3. pip install streamlit ragas langchain-community
"""

import streamlit as st
import time
import os
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG · Asistente",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #161921;
    --border:  #252a35;
    --accent:  #4f9cf9;
    --accent2: #7c5cfc;
    --ok:      #34d399;
    --warn:    #fbbf24;
    --err:     #f87171;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --user-bg: #1e2433;
    --bot-bg:  #161d2d;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}
section[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Header */
.rag-header {
    display: flex; align-items: center; gap: 16px;
    padding: 28px 0 16px 0;
    border-bottom: 1px solid var(--border); margin-bottom: 24px;
}
.rag-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; letter-spacing: -0.5px; margin: 0;
}
.rag-header .badge {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff; padding: 3px 10px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 500; text-transform: uppercase;
}

/* Burbujas */
.msg-user {
    background: var(--user-bg); border: 1px solid var(--border);
    border-radius: 14px 14px 4px 14px;
    padding: 14px 18px; margin: 10px 0;
    max-width: 75%; margin-left: auto; font-size: 0.95rem;
}
.msg-bot {
    background: var(--bot-bg); border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 4px 14px 14px 14px;
    padding: 16px 20px; margin: 10px 0;
    max-width: 82%; font-size: 0.95rem; line-height: 1.7;
}
.msg-bot.no-info { border-left-color: var(--warn); }
.msg-bot.error   { border-left-color: var(--err);  }

.role-label {
    font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 6px;
}
.role-label.bot  { color: var(--accent); }
.role-label.user { color: var(--muted); text-align: right; }

/* Chips fuentes */
.source-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.chip {
    background: #1a2235; border: 1px solid var(--border);
    border-radius: 20px; padding: 3px 10px;
    font-size: 0.72rem; font-family: 'DM Mono', monospace;
    color: var(--muted); display: inline-flex; align-items: center; gap: 5px;
}
.chip .score { color: var(--accent); font-weight: 500; }

/* Score bars */
.score-bar-wrap { margin-top: 10px; border-top: 1px solid var(--border); padding-top: 10px; }
.score-bar-label { font-size: 0.72rem; color: var(--muted); margin-bottom: 4px; }
.score-bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-bottom: 6px; }
.score-bar-fill { height: 100%; border-radius: 2px; background: linear-gradient(90deg, var(--accent), var(--accent2)); }

/* ── Métricas RAGAS ── */
.metrics-section {
    margin-top: 14px;
    border-top: 1px dashed var(--border);
    padding-top: 12px;
}
.metrics-title {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--muted);
    margin-bottom: 10px;
}
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}
.metric-card {
    background: #0f1520;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
}
.metric-label {
    font-size: 0.62rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: var(--muted); margin-bottom: 4px;
}
.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 1rem; font-weight: 500; color: var(--text);
}
.metric-bar {
    height: 3px; background: var(--border);
    border-radius: 2px; margin-top: 5px; overflow: hidden;
}
.metric-bar-fill { height: 100%; border-radius: 2px; }
.bar-ok   { background: var(--ok);   }
.bar-warn { background: var(--warn); }
.bar-err  { background: var(--err);  }

/* Inputs / botones */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 10px !important;
}
.stTextInput > div > div > input:focus { border-color: var(--accent) !important; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 500 !important;
    padding: 0.5rem 1.4rem !important; transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 500;
}
.pill.green  { background: rgba(52,211,153,0.12); color: var(--ok);  border: 1px solid rgba(52,211,153,0.3); }
.pill.yellow { background: rgba(251,191,36,0.12); color: var(--warn); border: 1px solid rgba(251,191,36,0.3); }
.pill.red    { background: rgba(248,113,113,0.12); color: var(--err); border: 1px solid rgba(248,113,113,0.3); }

div[data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--border) !important; color: var(--text) !important;
}
li[role="option"] { background: var(--surface) !important; }
li[role="option"]:hover { background: var(--user-bg) !important; }
details > summary { color: var(--muted) !important; font-size: 0.82rem !important; }
hr { border-color: var(--border) !important; }
.ts { font-size: 0.68rem; color: var(--muted); margin-top: 6px; font-family: 'DM Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Importaciones RAG
# ──────────────────────────────────────────────────────────────
RAG_AVAILABLE = True
IMPORT_ERROR  = ""

try:
    from vector_store import VectorStore
    from rag_engine import RAGEngine, get_available_models, TOP_K, MIN_SCORE
    from main import INDEX_FOLDER, CHUNK_SIZE, OVERLAP
except ImportError as e:
    RAG_AVAILABLE = False
    IMPORT_ERROR  = str(e)
    INDEX_FOLDER  = "vector_db"
    CHUNK_SIZE    = 500
    OVERLAP       = 50
    TOP_K         = 3
    MIN_SCORE     = 0.35

# ──────────────────────────────────────────────────────────────
# Importaciones RAGAS (juez Ollama)
# ──────────────────────────────────────────────────────────────
RAGAS_AVAILABLE = True
RAGAS_IMPORT_ERROR = ""

try:
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
except ImportError as e:
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = str(e)

# Modelo juez Ollama (mismo que el generador por defecto)
LLM_JUEZ       = "mistral"
EMBEDDING_JUDGE = "nomic-embed-text"

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def status_pill(label, kind):
    return f'<span class="pill {kind}">{label}</span>'

def render_sources(chunks):
    if not chunks:
        return ""
    chips = "".join(
        f'<span class="chip">📄 {e["source"]} · chunk #{e["chunk_id"]} '
        f'<span class="score">{int(s*100)}%</span></span>'
        for e, s in chunks
    )
    return f'<div class="source-chips">{chips}</div>'

def render_score_bars(chunks):
    if not chunks:
        return ""
    html = '<div class="score-bar-wrap">'
    for entry, score in chunks[:3]:
        pct  = int(score * 100)
        name = entry["source"][:32] + ("…" if len(entry["source"]) > 32 else "")
        html += (
            f'<div class="score-bar-label">{name} — {pct}%</div>'
            f'<div class="score-bar"><div class="score-bar-fill" style="width:{pct}%"></div></div>'
        )
    return html + "</div>"

def bar_color(value: float) -> str:
    if value >= 0.65: return "bar-ok"
    if value >= 0.40: return "bar-warn"
    return "bar-err"

# ──────────────────────────────────────────────────────────────
# RAGAS: construir juez y evaluar una sola muestra
# ──────────────────────────────────────────────────────────────
def _build_ragas_judge():
    """Crea el LLM y embedding wrapper de Ollama para RAGAS."""
    llm_juez = LangchainLLMWrapper(
        ChatOllama(model=LLM_JUEZ, temperature=0)
    )
    embeddings_juez = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=EMBEDDING_JUDGE)
    )
    return llm_juez, embeddings_juez


def compute_ragas_metrics(
    user_input: str,
    retrieved_chunks: list,
    answer: str,
    elapsed: float,
) -> dict:
    """
    Evalúa una respuesta RAG con las tres métricas RAGAS:
      - Faithfulness      : ¿la respuesta se apoya solo en el contexto?
      - Answer Relevancy  : ¿la respuesta responde la pregunta?
      - Context Precision : ¿los chunks recuperados son pertinentes?
      - Latencia          : tiempo total de respuesta en segundos.
    Devuelve un dict con las puntuaciones (0–1) o None si RAGAS no está disponible.
    """
    if not RAGAS_AVAILABLE or not retrieved_chunks:
        return {
            "faithfulness":      None,
            "answer_relevancy":  None,
            "context_precision": None,
            "latencia":          round(elapsed, 2),
            "ragas_error":       RAGAS_IMPORT_ERROR if not RAGAS_AVAILABLE else "Sin chunks",
        }

    try:
        llm_juez, embeddings_juez = _build_ragas_judge()

        contexts = [entry["text"] for entry, _ in retrieved_chunks]

        sample = SingleTurnSample(
            user_input=user_input,
            retrieved_contexts=contexts,
            response=answer,
            reference=answer,   # sin referencia externa; CP usará contexto vs respuesta
        )
        dataset = EvaluationDataset(samples=[sample])

        faithfulness_metric      = Faithfulness(llm=llm_juez)
        answer_relevancy_metric  = AnswerRelevancy(llm=llm_juez, embeddings=embeddings_juez)
        context_precision_metric = ContextPrecision(llm=llm_juez)

        resultados = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness_metric,
                answer_relevancy_metric,
                context_precision_metric,
            ],
        )

        df = resultados.to_pandas()
        row = df.iloc[0]

        def safe(key):
            val = row.get(key, None)
            if val is None:
                return None
            try:
                import math
                return None if math.isnan(float(val)) else round(float(val), 4)
            except Exception:
                return None

        return {
            "faithfulness":      safe("faithfulness"),
            "answer_relevancy":  safe("answer_relevancy"),
            "context_precision": safe("context_precision"),
            "latencia":          round(elapsed, 2),
            "ragas_error":       None,
        }

    except Exception as ex:
        return {
            "faithfulness":      None,
            "answer_relevancy":  None,
            "context_precision": None,
            "latencia":          round(elapsed, 2),
            "ragas_error":       str(ex),
        }


def render_metrics(m: dict) -> str:
    """Renderiza las tres métricas RAGAS + Latencia como tarjetas HTML."""
    if not m:
        return ""

    def fmt(val):
        """Formatea un valor 0–1 como porcentaje, o '—' si es None."""
        if val is None:
            return "—"
        return f"{val:.0%}"

    def card(label, val_str, pct, color):
        pct_css = int(min(max(pct if pct is not None else 0, 0), 1) * 100)
        fill_color = color if pct is not None else "bar-err"
        return (
            f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{val_str}</div>'
            f'<div class="metric-bar">'
            f'<div class="metric-bar-fill {fill_color}" style="width:{pct_css}%"></div>'
            f'</div>'
            f'</div>'
        )

    lat      = m["latencia"]
    lat_norm = max(0.0, 1.0 - lat / 20.0)
    lat_col  = "bar-ok" if lat < 5 else ("bar-warn" if lat < 15 else "bar-err")

    faith = m["faithfulness"]
    ar    = m["answer_relevancy"]
    cp    = m["context_precision"]

    html  = '<div class="metrics-section">'
    html += '<div class="metrics-title">⚖ Métricas RAGAS (juez Ollama)</div>'

    if m.get("ragas_error"):
        html += (
            f'<div style="font-size:0.72rem;color:var(--err);margin-bottom:8px;">'
            f'⚠ {m["ragas_error"]}</div>'
        )

    html += '<div class="metrics-grid">'
    html += card("Faithfulness",      fmt(faith), faith, bar_color(faith or 0))
    html += card("Answer Relevancy",  fmt(ar),    ar,    bar_color(ar or 0))
    html += card("Context Precision", fmt(cp),    cp,    bar_color(cp or 0))
    html += card("Latencia",          f"{lat}s",  lat_norm, lat_col)
    html += '</div></div>'
    return html


def check_ollama():
    try:
        return True, get_available_models()
    except Exception:
        return False, []

NO_INFO_PHRASES = [
    "no tengo información", "no encuentro",
    "no hay información", "no se encontró", "no encontré",
]
def is_no_info(text):
    return any(p in text.lower() for p in NO_INFO_PHRASES)

# ──────────────────────────────────────────────────────────────
# Estado de sesión
# ──────────────────────────────────────────────────────────────
for key, val in [
    ("messages",       []),
    ("store",          None),
    ("engine",         None),
    ("selected_model", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔭RAG · Sistema")
    st.markdown("---")

    # Estado
    st.markdown("**Estado del sistema**")
    if not RAG_AVAILABLE:
        st.markdown(status_pill("⚠ Módulos no encontrados", "red"), unsafe_allow_html=True)
        st.caption(f"Error: `{IMPORT_ERROR}`")
    else:
        ollama_ok, models = check_ollama()
        st.markdown(
            status_pill("● Ollama activo", "green") if ollama_ok
            else status_pill("● Ollama inactivo", "red"),
            unsafe_allow_html=True,
        )
        if not ollama_ok:
            st.caption("Ejecuta `ollama serve` en una terminal.")

        index_exists = os.path.exists(INDEX_FOLDER)
        st.markdown(
            status_pill("● Índice disponible", "green") if index_exists
            else status_pill("● Sin índice", "yellow"),
            unsafe_allow_html=True,
        )

    # Estado RAGAS
    st.markdown("---")
    st.markdown("**Evaluación RAGAS**")
    if RAGAS_AVAILABLE:
        st.markdown(
            status_pill("● RAGAS disponible", "green"),
            unsafe_allow_html=True,
        )
        st.caption(f"Juez LLM: `{LLM_JUEZ}` · Embedding: `{EMBEDDING_JUDGE}`")
    else:
        st.markdown(
            status_pill("⚠ RAGAS no instalado", "red"),
            unsafe_allow_html=True,
        )
        st.caption(f"`pip install ragas langchain-community`")

    st.markdown("---")

    if RAG_AVAILABLE:
        ollama_ok, models = check_ollama()

        if ollama_ok and models:

            # Modelo
            st.markdown("**Modelo LLM**")
            selected_model = st.selectbox(
                "Modelo", options=models,
                label_visibility="collapsed", key="model_select",
            )

            # ── Subida de documentos ──────────────────────────
            st.markdown("---")
            st.markdown("**Subir documentos**")
            DOCS_FOLDER = "docs"
            os.makedirs(DOCS_FOLDER, exist_ok=True)

            uploaded_files = st.file_uploader(
                "Arrastra o selecciona archivos",
                type=["txt", "pdf", "docx"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="file_uploader",
            )
            if uploaded_files:
                saved = []
                for uf in uploaded_files:
                    with open(os.path.join(DOCS_FOLDER, uf.name), "wb") as f:
                        f.write(uf.getbuffer())
                    saved.append(uf.name)
                st.success(f"✔ {len(saved)} archivo(s) guardado(s)")
                for n in saved:
                    st.caption(f" {n}")

            docs_in_folder = [
                f for f in os.listdir(DOCS_FOLDER)
                if os.path.splitext(f)[1].lower() in {".txt", ".pdf", ".docx"}
            ] if os.path.isdir(DOCS_FOLDER) else []

            if docs_in_folder:
                with st.expander(f"📁 docs/ ({len(docs_in_folder)} archivo(s))"):
                    for fname in sorted(docs_in_folder):
                        col_f, col_x = st.columns([4, 1])
                        with col_f:
                            ext = os.path.splitext(fname)[1].upper()[1:]
                            st.caption(f"`{ext}` {fname}")
                        with col_x:
                            if st.button("✕", key=f"del_{fname}"):
                                os.remove(os.path.join(DOCS_FOLDER, fname))
                                st.rerun()

            # ── Parámetros de indexación ──────────────────────
            st.markdown("---")
            st.markdown("**Parámetros de indexación**")

            chunk_size = st.slider(
                "Tamaño de chunk (tokens)",
                min_value=100, max_value=2000,
                value=CHUNK_SIZE, step=50,
                help="Fragmentos más grandes dan más contexto por chunk pero menor precisión de búsqueda.",
            )
            overlap = st.slider(
                "Overlap entre chunks (tokens)",
                min_value=0, max_value=min(500, chunk_size - 10),
                value=min(OVERLAP, chunk_size - 10), step=10,
                help="Tokens compartidos entre chunks consecutivos. Evita perder info en los bordes.",
            )
            st.markdown(
                f"<div style='font-size:0.72rem;color:#64748b;'>"
                f"chunk: <b>{chunk_size}</b> · overlap: <b>{overlap}</b> · "
                f"ratio: <b>{overlap/chunk_size:.0%}</b></div>",
                unsafe_allow_html=True,
            )

            if docs_in_folder:
                if st.button("⚡ Re-indexar documentos", use_container_width=True):
                    try:
                        from document_loader import load_documents_from_folder, build_chunks_from_documents
                        pb = st.progress(0, text="Cargando documentos…")
                        documents = load_documents_from_folder(DOCS_FOLDER)
                        if not documents:
                            st.error("No se encontraron documentos válidos.")
                        else:
                            pb.progress(25, text=f"{len(documents)} doc(s). Generando chunks…")
                            try:
                                chunks = build_chunks_from_documents(
                                    documents, chunk_size=chunk_size, overlap=overlap
                                )
                            except TypeError:
                                chunks = build_chunks_from_documents(documents)

                            pb.progress(55, text=f"{len(chunks)} chunks. Vectorizando…")
                            new_store = VectorStore()
                            new_store.build_index(chunks)

                            pb.progress(80, text="Guardando índice…")
                            new_store.save(INDEX_FOLDER)

                            pb.progress(100, text="¡Listo!")
                            st.success(
                                f"✔ {len(documents)} doc(s) · {len(chunks)} chunks · "
                                f"chunk={chunk_size} · overlap={overlap}"
                            )
                            if st.session_state.selected_model:
                                new_engine = RAGEngine(
                                    vector_store=new_store,
                                    model=st.session_state.selected_model,
                                    top_k=TOP_K,
                                )
                                st.session_state.store  = new_store
                                st.session_state.engine = new_engine
                                st.info("Engine recargado automáticamente.")
                    except Exception as ex:
                        st.error(f"Error durante la indexación: {ex}")

            # ── Parámetros de búsqueda ────────────────────────
            st.markdown("---")
            st.markdown("**Parámetros de búsqueda**")

            top_k = st.slider(
                "Chunks a recuperar (top-k)",
                min_value=1, max_value=10, value=TOP_K, step=1,
            )
            min_score = st.slider(
                "Score mínimo de similitud",
                min_value=0.20, max_value=0.95,
                value=MIN_SCORE, step=0.01, format="%.2f",
            )

            if st.button(" Cargar / recargar sistema", use_container_width=True):
                if not os.path.exists(INDEX_FOLDER):
                    st.error("Sin índice. Sube documentos y haz ⚡ Re-indexar.")
                else:
                    with st.spinner("Cargando base vectorial…"):
                        try:
                            store = VectorStore()
                            store.load(INDEX_FOLDER)
                            engine = RAGEngine(
                                vector_store=store,
                                model=selected_model,
                                top_k=top_k,
                            )
                            store._min_score_override       = min_score
                            st.session_state.store          = store
                            st.session_state.engine         = engine
                            st.session_state.selected_model = selected_model
                            st.success(f"Sistema listo · {selected_model}")
                        except Exception as ex:
                            st.error(f"Error al cargar: {ex}")
        else:
            st.warning("Ollama no disponible o sin modelos instalados.")

    # Config resumen
    st.markdown("---")
    if RAG_AVAILABLE:
        with st.expander("⚙ Config actual"):
            try:
                st.markdown(f"""
| Param | Valor |
|-------|-------|
| `chunk_size` | `{CHUNK_SIZE}` |
| `overlap` | `{OVERLAP}` |
| `top_k` | `{TOP_K}` |
| `min_score` | `{MIN_SCORE}` |
| `index_folder` | `{INDEX_FOLDER}` |
| `llm_juez` | `{LLM_JUEZ}` |
| `embedding_judge` | `{EMBEDDING_JUDGE}` |
""")
            except Exception:
                st.caption("No se pudo leer config de `main.py`.")

    st.markdown("---")
    if st.button("🗑 Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <span style="font-size:2.2rem"></span>
    <div>
        <h1>Asistente de Reglamento</h1>
        <span style="color:#64748b;font-size:0.85rem">RAG · Ollama · FAISS · Búsqueda semántica</span>
    </div>
    <span class="badge">RAG Local</span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Conversación
# ──────────────────────────────────────────────────────────────
with st.container():
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#64748b;">
            <div style="font-size:3rem;margin-bottom:16px"></div>
            <div style="font-size:1rem;margin-bottom:8px;color:#94a3b8">
                Haz tu primera pregunta sobre el reglamento.
            </div>
            <div style="font-size:0.82rem">
                El sistema buscará semánticamente en los documentos indexados.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown('<div class="role-label user">Tú</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="ts" style="text-align:right">{msg["ts"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="role-label bot">Asistente RAG</div>', unsafe_allow_html=True)

                is_error = msg.get("is_error", False)
                no_info  = is_no_info(msg["content"])
                css      = "msg-bot error" if is_error else ("msg-bot no-info" if no_info else "msg-bot")

                sources_html = render_sources(msg.get("chunks", []))
                scores_html  = render_score_bars(msg.get("chunks", [])) if msg.get("chunks") else ""
                metrics_html = render_metrics(msg.get("metrics")) if msg.get("metrics") else ""

                st.markdown(
                    f'<div class="{css}">'
                    f'{msg["content"]}'
                    f'{sources_html}'
                    f'{scores_html}'
                    f'{metrics_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="ts">{msg["ts"]} · modelo: {msg.get("model","—")}</div>',
                    unsafe_allow_html=True,
                )

                if msg.get("chunks"):
                    with st.expander(f"Ver {len(msg['chunks'])} fragmentos recuperados"):
                        for i, (entry, score) in enumerate(msg["chunks"], 1):
                            st.markdown(
                                f"**Fragmento {i}** — `{entry['source']}` · "
                                f"chunk `#{entry['chunk_id']}` · score `{score:.4f}`"
                            )
                            st.markdown(
                                f"> {entry['text'][:500]}"
                                f"{'…' if len(entry['text']) > 500 else ''}"
                            )
                            if i < len(msg["chunks"]):
                                st.markdown("---")

# ──────────────────────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────────────────────
st.markdown("---")
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_query = st.text_input(
        "Pregunta",
        placeholder="¿Cuáles son los temas del módulo 1?",
        label_visibility="collapsed",
        key="query_input",
    )
with col_btn:
    send_clicked = st.button("Enviar →", use_container_width=True)

st.markdown(
    "<div style='font-size:0.78rem;color:#64748b;margin-top:6px'>"
    "Ejemplos: <i>¿Qué pasa si pierdo la materia? · ¿Cuántos créditos tiene el curso? · "
    "¿Cuál es el objetivo de la asignatura?</i></div>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Procesamiento de la consulta
# ──────────────────────────────────────────────────────────────
if send_clicked and user_query.strip():
    ts = datetime.now().strftime("%H:%M:%S")

    st.session_state.messages.append({
        "role": "user", "content": user_query.strip(), "ts": ts,
    })

    if st.session_state.engine is None:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠ Sistema no cargado. Usa ** Cargar / recargar sistema** en el panel lateral.",
            "chunks": [], "metrics": None, "model": "—", "ts": ts, "is_error": True,
        })
        st.rerun()

    with st.spinner("Buscando en el reglamento…"):
        try:
            t0      = time.time()
            result  = st.session_state.engine.query(user_query.strip())
            elapsed = time.time() - t0

            answer  = result["answer"]
            chunks  = result["retrieved_chunks"]
            model   = result["model"]

        except Exception as ex:
            st.session_state.messages.append({
                "role":     "assistant",
                "content":  f" Error al procesar la consulta:\n\n`{ex}`",
                "chunks":   [], "metrics": None, "model": "—",
                "ts":       ts, "is_error": True,
            })
            st.rerun()

    # Evaluación RAGAS (puede tardar; se muestra spinner aparte)
    with st.spinner("⚖ Evaluando con RAGAS (Faithfulness · Answer Relevancy · Context Precision)…"):
        metrics = compute_ragas_metrics(
            user_input=user_query.strip(),
            retrieved_chunks=chunks,
            answer=answer,
            elapsed=elapsed,
        )

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  answer,
        "chunks":   chunks,
        "metrics":  metrics,
        "model":    model,
        "ts":       f"{ts} ({elapsed:.1f}s)",
        "is_error": False,
    })

    st.rerun()