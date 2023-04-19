from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader


def configure():
    load_dotenv()

configure()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


URL = "https://nuclearsafety.gc.ca/eng/acts-and-regulations/regulatory-documents/published/html/regdoc2-5-2/index.cfm"
loader = WebBaseLoader(URL)
data = loader.load()
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
texts = text_splitter.split_documents(data)
print (f'Now you have {len(texts)} documents')

from langchain.vectorstores import Milvus