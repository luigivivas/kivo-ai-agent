"""Wrapper delgado sobre el SDK google-genai: embeddings + generación de texto."""
import os

import numpy as np
from google import genai

EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-2")
CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-3.6-flash")


def get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Falta GEMINI_API_KEY. Define la variable de entorno o crea un archivo .env "
            "a partir de .env.example."
        )
    return genai.Client(api_key=api_key)


def _embed_one(client: genai.Client, text: str) -> list[float]:
    # embed_content con una lista de varios textos no devuelve un embedding por texto
    # (se probó en la práctica y solo entrega uno por llamada), así que se embebe de a uno.
    result = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return result.embeddings[0].values


def embed_documents(client: genai.Client, chunks: list[str], titles: list[str]) -> np.ndarray:
    """Embebe texto de documentos usando el formato asimétrico recomendado para gemini-embedding-2."""
    vectors = [
        _embed_one(client, f"title: {title} | text: {chunk}")
        for chunk, title in zip(chunks, titles)
    ]
    return np.array(vectors, dtype="float32")


def embed_query(client: genai.Client, question: str) -> np.ndarray:
    vector = _embed_one(client, f"task: search result | query: {question}")
    return np.array(vector, dtype="float32")


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
