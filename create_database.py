from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PDF_PATH = "documentLoaders/DSA documents/3. Sorting.pdf"
EXCEL_PATH = "documentLoaders/Striver Sheet.xlsx"
CHROMA_PERSIST_DIR = "./chroma_langchain_db"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

embeddings = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma(
    collection_name="coursemate_documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
)

pdf_docs = PyPDFLoader(PDF_PATH).load()
excel_docs = UnstructuredExcelLoader(EXCEL_PATH, mode="elements").load()

chunks = text_splitter.split_documents(pdf_docs + excel_docs)

vector_store.add_documents(documents=chunks)

print(f"Stored {len(chunks)} chunks in Chroma at {CHROMA_PERSIST_DIR}")
