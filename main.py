import base64
import io

import pymupdf
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from PIL import Image

load_dotenv()

# Pages with fewer real characters than this are treated as scanned/handwritten
# (pypdf found no usable text layer) and routed through OCR instead.
MIN_TEXT_LAYER_CHARS = 20

vision_model = ChatMistralAI(model="mistral-small-2506")


def render_page_to_image(pdf_path: str, page_number: int, zoom: float = 2.0) -> Image.Image:
    doc = pymupdf.open(pdf_path)
    try:
        pix = doc[page_number].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return Image.open(io.BytesIO(pix.tobytes("png")))
    finally:
        doc.close()


def ocr_with_vision_llm(image: Image.Image) -> str:
    """OCR fallback for pages with no usable embedded text layer (e.g.
    handwritten/scanned pages), using Mistral's multimodal model to
    transcribe the page image."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Transcribe all text in this image exactly as written, "
                    "including handwriting. Do not correct spelling, syntax, "
                    "or errors in code - transcribe it verbatim, even if it "
                    "looks wrong. If a word or character is illegible, mark "
                    "it as [unclear] rather than guessing. Output only the "
                    "transcribed text."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
            },
        ]
    )
    return vision_model.invoke([message]).content.strip()


def extract_pdf_text(pdf_path: str) -> list[str]:
    """Extract text per page, falling back to vision-LLM OCR for pages with
    no usable embedded text layer, e.g. handwritten or scanned pages."""
    pages = PyPDFLoader(pdf_path).load()

    page_texts = []
    for i, page in enumerate(pages):
        text = page.page_content.strip()
        if len(text) < MIN_TEXT_LAYER_CHARS:
            image = render_page_to_image(pdf_path, i)
            text = ocr_with_vision_llm(image)
        page_texts.append(text)

    return page_texts


data = UnstructuredExcelLoader("documentLoaders/Striver Sheet.xlsx", mode="elements")
docs = data.load()

pdf_pages = extract_pdf_text("documentLoaders/DSA documents/3. Sorting.pdf")
pdf_text = "\n\n".join(pdf_pages)

template = ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarizes the text "), ("human", "{data}")]
)

template2 = ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarizes the text "), ("human", "{data}")]
)

model = ChatMistralAI(model="mistral-small-2506")

prompt = template.format_prompt(data=docs[0].page_content)

result = model.invoke(prompt)

model2 = ChatMistralAI(model="mistral-small-2506")

prompt2 = template2.format_prompt(data=pdf_text)

result2 = model2.invoke(prompt2)


print(result.content)
print(result2.content)
