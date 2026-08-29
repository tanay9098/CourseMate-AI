from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import UnstructuredExcelLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = UnstructuredExcelLoader("documentLoaders/Striver Sheet.xlsx", mode="elements")
docs= data.load()

data2 = PyPDFLoader("documentLoaders/DSA documents/3. Sorting.pdf")
docs2 = data2.load()

template=ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarizes the text "),
     ("human", "{data}")

    ]
)

template2=ChatPromptTemplate.from_messages(
    [("system", "you are an AI that summarizes the text "),
     ("human", "{data}")

    ]
)

model=ChatMistralAI(model= "mistral-small-2506")

prompt=template.format_prompt(data=docs[0].page_content)

result=model.invoke(prompt)

model2=ChatMistralAI(model= "mistral-small-2506")

prompt2=template2.format_prompt(data=docs2[0].page_content)

result2=model2.invoke(prompt2)



print(result.content)
print(result2.content)

