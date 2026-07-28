from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = TextLoader("documentLoaders/Striver Sheet.xlsx")
docs= data.load()

template=ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarizes the text "),
     ("human", "{data}")

    ]
)

model=ChatMistralAI(model= "mistral-small-2506")

result=model.invoke("hello")

print(result.content)


