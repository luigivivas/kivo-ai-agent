"""Construye el índice vectorial a partir de los documentos en data/.

Uso:
    python ingest.py            # construye si el índice no existe
    python ingest.py --force    # reconstruye aunque ya exista
"""
import argparse
from pathlib import Path

from src import gemini_client as gc
from src.loaders import load_all
from src.store import VectorStore

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"


def build_index() -> None:
    chunks = load_all(DATA_DIR)
    if not chunks:
        raise SystemExit(f"No se encontraron documentos en {DATA_DIR}")

    print(f"Documentos troceados en {len(chunks)} fragmentos. Generando embeddings...")
    client = gc.get_client()
    texts = [c.text for c in chunks]
    titles = [c.source for c in chunks]
    vectors = gc.embed_documents(client, texts, titles)

    store = VectorStore(chunks, vectors)
    store.save(INDEX_DIR)
    print(f"Índice guardado en {INDEX_DIR} ({len(chunks)} fragmentos, dim={vectors.shape[1]}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="reconstruye el índice aunque ya exista")
    args = parser.parse_args()

    if VectorStore.exists(INDEX_DIR) and not args.force:
        print(f"El índice ya existe en {INDEX_DIR}. Usa --force para reconstruirlo.")
    else:
        build_index()
