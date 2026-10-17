
import pickle
from pathlib import Path

import faiss
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Configuration
BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / "Shakespeare_document"
ARTIFACTS_FOLDER = BASE_DIR / "artifacts"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "phi3:latest"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5
MIN_SIMILARITY = 0.25

INDEX_PATH = ARTIFACTS_FOLDER / "faiss.index"
METADATA_PATH = ARTIFACTS_FOLDER / "metadata.pkl"


def extract_pdf(pdf_path):
    """Extract text from every PDF page with source metadata."""
    pages = []

    for page_number, page in enumerate(PdfReader(pdf_path).pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "text": " ".join(text.split()),
                "source": pdf_path.name,
                "page": page_number,
            })

    return pages


def chunk_pages(pages):
    """Split page text into overlapping chunks."""
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
    """Create a FAISS index from normalized text embeddings."""
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


def load_or_build_index(model):
    """Load existing artifacts or ask to create them."""
    ARTIFACTS_FOLDER.mkdir(parents=True, exist_ok=True)

    if INDEX_PATH.exists() and METADATA_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))

        with open(METADATA_PATH, "rb") as file:
            chunks = pickle.load(file)

        if index.ntotal == len(chunks) and index.ntotal > 0:
            print(f"Loaded existing index: {index.ntotal} chunks")
            return chunks, index

        print("The index and metadata do not match.")

    choice = input(
        "FAISS index or metadata missing/invalid. "
        "Create the index now? (y/n): "
    ).strip().lower()

    if choice not in {"y", "yes"}:
        raise SystemExit("Index creation cancelled.")

    if not PDF_FOLDER.is_dir():
        raise SystemExit(f"PDF folder not found: {PDF_FOLDER}")

    pdf_files = sorted(PDF_FOLDER.rglob("*.pdf"))

    if not pdf_files:
        raise SystemExit(f"No PDF files found in: {PDF_FOLDER}")

    chunks = []

    for number, pdf_path in enumerate(pdf_files, start=1):
        print(f"Reading PDF {number}/{len(pdf_files)}: {pdf_path.name}")
        chunks.extend(chunk_pages(extract_pdf(pdf_path)))

    if not chunks:
        raise SystemExit("No extractable PDF text found.")

    print(f"Created {len(chunks)} chunks.")
    print("Building FAISS index...")

    index = build_index(chunks, model)

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "wb") as file:
        pickle.dump(chunks, file)

    print(f"Index and metadata saved in: {ARTIFACTS_FOLDER}")

    return chunks, index


def retrieve(question, chunks, index, model):
    """Retrieve the most relevant chunks for a question."""
    vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(vector, min(TOP_K, index.ntotal))

    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            result = dict(chunks[int(idx)])
            result["score"] = float(score)
            results.append(result)

    return results


def answer_question(question, passages):
    """Generate an evidence-based answer using local Ollama."""
    if not passages or max(p["score"] for p in passages) < MIN_SIMILARITY:
        return "The documents do not contain enough relevant evidence to answer."

    context = "\n\n".join(
        f"[{i}] Source: {p['source']}, PDF page {p['page']}\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )

    prompt = f"""Answer using ONLY the passages below.
Do not invent facts or use outside knowledge.
Cite factual claims using passage numbers, such as [1].
If the evidence is insufficient, say so.

PASSAGES:
{context}

QUESTION: {question}

ANSWER:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.2,
            "num_ctx": 2048,
            "num_predict": 128,
        },
    )

    return response["message"]["content"]


def main():
    #print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    chunks, index = load_or_build_index(embedding_model)

    #print(f"\nRAG is ready. Indexed {len(chunks)} chunks.")
    #print("Ask a question, or type 'exit' to quit.\n")

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

        passages = retrieve(question, chunks, index, embedding_model)
        answer = answer_question(question, passages)

        print("\nAnswer\n------")
        print(answer)

        '''print("\nSources")
        for i, passage in enumerate(passages, start=1):
            print(
                f"[{i}] {passage['source']}, "
                f"page {passage['page']}, "
                f"similarity: {passage['score']:.3f}"
            )'''


if __name__ == "__main__":
    main()
