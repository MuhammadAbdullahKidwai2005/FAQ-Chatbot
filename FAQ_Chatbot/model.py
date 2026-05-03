import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv

load_dotenv()

# Lightweight embedding model — replaces TensorFlow/USE
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load OpenRouter API key from environment
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenRouter model
local_llm = ChatOpenRouter(
    model="openai/gpt-oss-120b:free",
    temperature=0.3,
    max_tokens=512,
    openrouter_api_key=openrouter_api_key
)

def build_vectorstore(chunks):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Vector database created successfully")
    return vectorstore