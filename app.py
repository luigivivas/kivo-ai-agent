from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.agent import KivoAgent

load_dotenv()

BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "index"

EXAMPLE_QUESTIONS = [
    "¿Cuánto cuesta el smartwatch Kivo Fit 2 y qué garantía tiene?",
    "Compré unos audífonos y no me gustaron, ¿los puedo devolver?",
    "¿A qué países hacen envíos y cuánto tardan?",
    "¿Qué pasa si mi pedido llega dañado?",
]

st.set_page_config(page_title="Asistente Kivo", page_icon="💬", layout="centered")


@st.cache_resource(show_spinner=False)
def load_agent() -> KivoAgent | None:
    try:
        return KivoAgent(INDEX_DIR)
    except Exception as exc:  # noqa: BLE001 - se muestra en la UI, no en logs
        st.session_state["load_error"] = str(exc)
        return None


with st.sidebar:
    st.markdown("### Kivo")
    st.caption("Tienda de tecnología y accesorios")
    st.write(
        "Este asistente responde preguntas sobre pedidos, envíos, devoluciones, garantía y "
        "el catálogo de productos de Kivo, usando como única fuente los documentos oficiales "
        "de la tienda."
    )
    st.markdown("**Documentos que conoce el agente**")
    st.markdown(
        "- Política de envíos, devoluciones y garantía\n"
        "- Preguntas frecuentes\n"
        "- Términos y condiciones\n"
        "- Catálogo de productos (CSV)"
    )
    st.markdown("---")
    st.markdown("**Preguntas de ejemplo**")
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- {q}")

st.title("Asistente Kivo")
st.caption("Preguntas sobre pedidos, envíos, devoluciones, garantía y productos")

agent = load_agent()

if agent is None:
    st.error(
        "No se pudo iniciar el agente: "
        + st.session_state.get("load_error", "error desconocido")
        + ".\n\nSi es la primera vez que corres el proyecto, revisa el README: falta generar "
        "el índice (`python ingest.py`) o configurar `GEMINI_API_KEY`."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "previous_interaction_id" not in st.session_state:
    st.session_state.previous_interaction_id = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Fuentes: " + ", ".join(message["sources"]))

question = st.chat_input("Escribe tu pregunta sobre Kivo...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consultando los documentos de Kivo..."):
            try:
                answer, sources, interaction_id = agent.ask(
                    question, st.session_state.previous_interaction_id
                )
                st.session_state.previous_interaction_id = interaction_id
            except Exception as exc:  # noqa: BLE001
                answer = f"Ocurrió un error consultando al modelo: {exc}"
                sources = []
        st.markdown(answer)
        if sources:
            st.caption("Fuentes: " + ", ".join(sources))

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
