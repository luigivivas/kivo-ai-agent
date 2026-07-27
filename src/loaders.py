"""Lectura y troceado (chunking) de los documentos fuente del agente."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str
    location: str


def _split_lines(text: str, max_chars: int = 800, overlap_lines: int = 2) -> list[str]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        if current_len + len(line) > max_chars and current:
            chunks.append(" ".join(current))
            current = current[-overlap_lines:] if overlap_lines else []
            current_len = sum(len(l) for l in current)
        current.append(line)
        current_len += len(line)

    if current:
        chunks.append(" ".join(current))
    return chunks


def load_pdf(path: Path) -> list[Chunk]:
    reader = PdfReader(str(path))
    chunks: list[Chunk] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for piece in _split_lines(text):
            chunks.append(Chunk(text=piece, source=path.name, location=f"página {page_num}"))
    return chunks


def load_csv(path: Path) -> list[Chunk]:
    df = pd.read_csv(path)
    chunks: list[Chunk] = []
    for row_num, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns]
        text = f"Producto del catálogo Kivo -> " + ", ".join(parts)
        chunks.append(Chunk(text=text, source=path.name, location=f"fila {row_num + 2}"))
    return chunks


def load_all(data_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        chunks.extend(load_pdf(pdf_path))
    for csv_path in sorted(data_dir.glob("*.csv")):
        chunks.extend(load_csv(csv_path))
    return chunks
