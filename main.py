# import base64
# import io
import hashlib
from pathlib import Path

# import pymupdf
from diskcache import Cache
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader

# from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from PIL import Image

load_dotenv()

DOCUMENT_LOADERS_DIR = Path("documentLoaders")
CACHE_DIR = ".cache/summaries"

# Bounds how many chunks are summarized at once. Matches Ollama's default
# OLLAMA_NUM_PARALLEL, so this saturates the server without piling up
# requests it would just queue anyway.
MAX_CONCURRENCY = 4

# Pages with fewer real characters than this are treated as scanned/handwritten
# (pypdf found no usable text layer) and routed through OCR instead.
# MIN_TEXT_LAYER_CHARS = 20

# vision_model = ChatOllama(model="qwen2.5vl:3b")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# def render_page_to_image(pdf_path: str, page_number: int, zoom: float = 2.0) -> Image.Image:
#     doc = pymupdf.open(pdf_path)
#     try:
#         pix = doc[page_number].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
#         return Image.open(io.BytesIO(pix.tobytes("png")))
#     finally:
#         doc.close()
#
#
# def ocr_with_vision_llm(image: Image.Image) -> str:
#     """OCR fallback for pages with no usable embedded text layer (e.g.
#     handwritten/scanned pages), using a local vision-language model to
#     transcribe the page image."""
#     buf = io.BytesIO()
#     image.save(buf, format="PNG")
#     b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
#
#     message = HumanMessage(
#         content=[
#             {
#                 "type": "text",
#                 "text": (
#                     "Transcribe all text in this image exactly as written, "
#                     "including handwriting. Do not correct spelling, syntax, "
#                     "or errors in code - transcribe it verbatim, even if it "
#                     "looks wrong. If a word or character is illegible, mark "
#                     "it as [unclear] rather than guessing. Output only the "
#                     "transcribed text."
#                 ),
#             },
#             {
#                 "type": "image_url",
#                 "image_url": {"url": f"data:image/png;base64,{b64_image}"},
#             },
#         ]
#     )
#     return vision_model.invoke([message]).content.strip()


def extract_pdf_text(pdf_path: str) -> list[str]:
    """Extract each page's embedded text layer. OCR for pages with no usable
    text layer (e.g. handwritten/scanned pages) is disabled for now - see the
    commented-out vision-LLM OCR code above."""
    pages = PyPDFLoader(pdf_path).load()
    return [page.page_content.strip() for page in pages]


excel_paths = sorted(DOCUMENT_LOADERS_DIR.rglob("*.xlsx"))
docs = []
for excel_path in excel_paths:
    docs.extend(UnstructuredExcelLoader(str(excel_path), mode="elements").load())
docs = text_splitter.split_documents(docs)

pdf_paths = sorted(DOCUMENT_LOADERS_DIR.rglob("*.pdf"))
pdf_pages = []
for pdf_path in pdf_paths:
    pdf_pages.extend(extract_pdf_text(str(pdf_path)))
pdf_chunks = text_splitter.split_text("\n\n".join(pdf_pages))

template = ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarizes the text "), ("human", "{data}")]
)

model = ChatOllama(model="gemma2:2b")

cache = Cache(CACHE_DIR)


def _cache_key(model_name: str, text: str) -> str:
    return hashlib.sha256(f"{model_name}|{text}".encode()).hexdigest()


def summarize_chunks(texts: list[str], label: str) -> list[str]:
    """Summarize each text with `model`, reusing cached results across runs
    and running uncached chunks concurrently. Prints progress as chunks
    finish so long runs show they're actually moving."""
    keys = [_cache_key(model.model, text) for text in texts]
    results: list[str | None] = [cache.get(key) for key in keys]

    pending = [i for i, summary in enumerate(results) if summary is None]
    total, cached = len(texts), len(texts) - len(pending)
    print(f"[{label}] {cached}/{total} chunks already cached, {len(pending)} left to summarize")

    if not pending:
        return results  # type: ignore[return-value]

    prompts = [template.format_prompt(data=texts[i]) for i in pending]
    done = cached
    for position, output in model.batch_as_completed(
        prompts, config={"max_concurrency": MAX_CONCURRENCY}
    ):
        idx = pending[position]
        summary = output.content
        results[idx] = summary
        cache.set(keys[idx], summary)
        done += 1
        print(f"[{label}] {done}/{total} chunks summarized ({done * 100 // total}%)")

    return results  # type: ignore[return-value]


excel_summaries = summarize_chunks([chunk.page_content for chunk in docs], "excel")
pdf_summaries = summarize_chunks(pdf_chunks, "pdf")

print("\n\n".join(excel_summaries))
print("\n\n".join(pdf_summaries))
