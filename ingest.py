from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorestores/db_faiss"


def create_vector_db():
    # Loading and splitting PDF data
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text = text_splitter.split_documents(documents)

    # embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="bert-base-uncased",
        model_kwargs={"device": "cuda"},
    )

    # create vector and save
    db = FAISS.from_documents(text, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    create_vector_db()
