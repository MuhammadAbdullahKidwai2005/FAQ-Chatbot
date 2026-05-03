from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

files = ["data/terms.pdf", "data/privacy.pdf"]
docs = []

for file in files:
    loader = PyPDFLoader(file)
    docs.extend(loader.load())

text_files = ["data/emails.txt", "data/chat_logs.txt"]

for file in text_files:
    loader = TextLoader(file)
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))