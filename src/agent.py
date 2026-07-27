"""Orquestación del agente: recuperación de contexto + generación de la respuesta."""
from pathlib import Path

from . import gemini_client as gc
from .store import VectorStore

SYSTEM_INSTRUCTION = """Eres el asistente virtual de Kivo, una tienda en línea de tecnología y \
accesorios que despacha a Colombia, México, Perú y Chile.

Reglas:
- Responde únicamente con base en el CONTEXTO que se te entrega en cada mensaje. Ese contexto \
viene de las políticas oficiales de Kivo y del catálogo de productos.
- Si la respuesta no está en el contexto, dilo con honestidad ("no tengo esa información en mis \
documentos") y sugiere escribir a soporte@kivo-tienda.com. No inventes precios, plazos ni políticas.
- Responde en español, en tono cercano y profesional, como lo haría un asesor de servicio al \
cliente. Sé breve y directo: 2 a 5 frases, salvo que la pregunta pida una lista.
- Si la pregunta menciona un producto, usa los datos exactos del catálogo (precio, stock, \
garantía) cuando estén disponibles en el contexto.
- No repitas literalmente el contexto ni menciones que "el documento dice"; simplemente responde \
como si conocieras la información."""

TOP_K = 5


def _format_context(hits: list[tuple]) -> str:
    blocks = []
    for chunk, _score in hits:
        blocks.append(f"[Fuente: {chunk.source}, {chunk.location}]\n{chunk.text}")
    return "\n\n".join(blocks)


class KivoAgent:
    def __init__(self, index_dir: Path):
        self.client = gc.get_client()
        self.store = VectorStore.load(index_dir)

    def ask(self, question: str, previous_interaction_id: str | None = None):
        query_vector = gc.embed_query(self.client, question)
        hits = self.store.search(query_vector, top_k=TOP_K)
        context = _format_context(hits)

        user_input = f"CONTEXTO:\n{context}\n\nPREGUNTA DEL CLIENTE:\n{question}"
        answer, interaction_id = gc.ask(
            self.client,
            system_instruction=SYSTEM_INSTRUCTION,
            user_input=user_input,
            previous_interaction_id=previous_interaction_id,
        )
        sources = sorted({f"{c.source} ({c.location})" for c, _ in hits})
        return answer, sources, interaction_id
