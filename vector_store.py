from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def build_vector_store(documents):

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_db = FAISS.from_documents(
        documents,
        embeddings
    )

    vector_db.save_local("faiss_index")

    return vector_db
