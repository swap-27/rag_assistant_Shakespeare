
import json
import os
import pickle
from pathlib import Path

import faiss
from dotenv import load_dotenv
from langchain_huggingface.chat_models import ChatHuggingFace
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = 1000, 150, 5
MIN_SIMILARITY = 0.25

INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.pkl"
MANIFEST_FILE = "manifest.json"


def get_pdf_files(folder):
    """Find all PDFs recursively, excluding macOS metadata folders."""
    return sorted(
        p for p in Path(folder).rglob("*.pdf")
        if "__MACOSX" not in p.parts
    )


def get_manifest(pdf_files, folder):
    """Detect added, removed, or modified PDFs."""
    return [
        {
            "path": str(p.relative_to(folder)),
            "size": p.stat().st_size,
            "modified": p.stat().st_mtime_ns,
        }
        for p in pdf_files
    ]


def extract_pdf(path, source):
    """Extract page text while preserving source filename and page number."""
    pages = []

    for number, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "text": " ".join(text.split()),
                "source": source,
                "page": number,
            })

    return pages


def chunk_pages(pages):
    """Split pages into overlapping chunks and preserve metadata."""
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))

            chunks.append({
                "text": text[start:end],
                "source": page["source"],
                "page": page["page"],
            })

            if end == len(text):
                break

            start = end - CHUNK_OVERLAP

    return chunks


def build_index(chunks, model):
    """Embed all chunks and build an exact FAISS similarity index."""
    vectors = model.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


def load_or_build_index(folder, model):
    """Load the cached index or rebuild it when PDFs change."""
    folder = Path(folder).resolve()
    pdf_files = get_pdf_files(folder)

    if not pdf_files:
        raise SystemExit(f"No PDFs found in: {folder}")

    manifest = get_manifest(pdf_files, folder)

    index_path = folder / INDEX_FILE
    metadata_path = folder / METADATA_FILE
    manifest_path = folder / MANIFEST_FILE

    # Reuse the index if the PDF collection has not changed.
    if all(p.exists() for p in (
        index_path, metadata_path, manifest_path
    )):
        with open(manifest_path, encoding="utf-8") as f:
            saved_manifest = json.load(f)

        if saved_manifest == manifest:
            index = faiss.read_index(str(index_path))

            with open(metadata_path, "rb") as f:
                chunks = pickle.load(f)

            if index.ntotal == len(chunks) and index.ntotal > 0:
                print(f"Loaded cached index: {index.ntotal} chunks")
                return chunks, index

    # Rebuild if any PDF was added, removed, or modified.
    print(f"Indexing {len(pdf_files)} PDF documents...")
    chunks = []

    for i, pdf in enumerate(pdf_files, start=1):
        source = str(pdf.relative_to(folder))
        print(f"[{i}/{len(pdf_files)}] Reading {source}")

        pages = extract_pdf(pdf, source)
        chunks.extend(chunk_pages(pages))

    if not chunks:
        raise SystemExit(
            "No extractable text found. Scanned PDFs may require OCR."
        )

    print(f"Created {len(chunks)} chunks.")
    print("Generating embeddings and building FAISS index...")

    index = build_index(chunks, model)

    # Save index and metadata for future runs.
    faiss.write_index(index, str(index_path))

    with open(metadata_path, "wb") as f:
        pickle.dump(chunks, f)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved index and metadata in {folder}")

    return chunks, index


def retrieve(question, chunks, index, model):
    """Retrieve the top-k chunks and their source metadata."""
    vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(vector, min(TOP_K, index.ntotal))

    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            item = dict(chunks[int(idx)])
            item["score"] = float(score)
            results.append(item)

    return results


def answer_question(question, passages, llm):
    """Generate an answer using only the retrieved evidence."""
    if not passages or max(p["score"] for p in passages) < MIN_SIMILARITY:
        return (
            "The provided documents do not contain enough relevant "
            "evidence to answer this question."
        )

    context = "\n\n".join(
        f"[{i}] Source: {p['source']}, PDF page {p['page']}\n"
        f"{p['text']}"
        for i, p in enumerate(passages, start=1)
    )
    for i, p in enumerate(passages, start=1):
        print("="*80)
        print(f"Source identified:{p['source']}")
        print(f"PDF page: {p['page']}")
        print(f"Chunk: {p['text']}")

    prompt = f"""You are a careful research assistant.

Answer using ONLY the supplied passages.
Do not use outside knowledge or invent facts.
If evidence is insufficient, explicitly say so.
Cite factual claims using passage numbers such as [1] and [2].
If sources conflict, explain the conflict.

PASSAGES:
{context}

QUESTION: {question}

Answer:"""

    '''response = llm.invoke(prompt)
    return response.content'''
    return


def main():

    folder = Path(__file__).resolve().parent / "Shakespeare_document"

    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("API_KEY")
    )

    if not hf_token:
        raise SystemExit(
            "Set HF_TOKEN in your .env file to access Hugging Face."
        )

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    chunks, index = load_or_build_index(folder, embedding_model)

    print("Connecting to the Hugging Face inference endpoint...")
    llm = ChatHuggingFace.from_model_id(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        backend="endpoint",
        huggingfacehub_api_token=hf_token,
        temperature=0.2,
        max_new_tokens=256,
    )

    print(f"\nReady. Indexed {len(chunks)} chunks.")
    print("Ask a question, or type 'exit' to quit.\n")

    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not question:
            continue

        passages = retrieve(
            question, chunks, index, embedding_model
        )
        answer = answer_question(question, passages, llm)

        print("\nAnswer\n------")
        print(answer)

        print("\nRetrieved sources\n-----------------")
        for i, p in enumerate(passages, start=1):
            print(
                f"[{i}] {p['source']} | PDF page {p['page']} "
                f"| similarity {p['score']:.3f}"
            )
            print(p["text"], "\n")

        print("=" * 72)


if __name__ == "__main__":
    main()
