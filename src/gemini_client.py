"""Wrapper delgado sobre el SDK google-genai: embeddings + generación de texto."""
import os

import numpy as np
from google import genai

EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-2")
CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.6-flash")

_EMBED_BATCH = 32


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta GEMINI_API_KEY. Define la variable de entorno o crea un archivo .env "
            "a partir de .env.example."
        )
    return genai.Client(api_key=api_key)


def _embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [e.values for e in result.embeddings]


def embed_documents(client: genai.Client, chunks: list[str], titles: list[str]) -> np.ndarray:
    """Embebe texto de documentos usando el formato asimétrico recomendado para gemini-embedding-2."""
    prefixed = [f"title: {title} | text: {chunk}" for chunk, title in zip(chunks, titles)]
    vectors: list[list[float]] = []
    for i in range(0, len(prefixed), _EMBED_BATCH):
        batch = prefixed[i : i + _EMBED_BATCH]
        vectors.extend(_embed_batch(client, batch))
    return np.array(vectors, dtype="float32")


def embed_query(client: genai.Client, question: str) -> np.ndarray:
    prefixed = f"task: search result | query: {question}"
    vectors = _embed_batch(client, [prefixed])
    return np.array(vectors[0], dtype="float32")


def ask(
    client: genai.Client,
    system_instruction: str,
    user_input: str,
    previous_interaction_id: str | None = None,
):
    kwargs = {}
    if previous_interaction_id:
        kwargs["previous_interaction_id"] = previous_interaction_id

    interaction = client.interactions.create(
        model=CHAT_MODEL,
        system_instruction=system_instruction,
        input=user_input,
        **kwargs,
    )
    return interaction.output_text, interaction.id
