from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

# safety imports

from model import final_result

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class item(BaseModel):
    query: str


@app.post("/")
async def query(item: item):
    print(item.query)
    ans = final_result(item.query)
    return ans.get("result")


# uvicorn app:app --host 0.0.0.0 --port 8000
