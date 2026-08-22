from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = UnstructuredExcelLoader("documentLoaders/Striver Sheet.xlsx", mode="elements")
docs= data.load()

template=ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarizes the text "),
     ("human", "{data}")

    ]
)

model=ChatMistralAI(model= "mistral-small-2506")

result=model.invoke("hello")

print(result.content)


