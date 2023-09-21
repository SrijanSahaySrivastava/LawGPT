from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = "vectorestores/db_faiss"

custom_prompt_template = """Use the Following Piece of information to answer the user's question.If you don't know the answer, please just say you don't know the answer, don't try to make up an answer.

Context: {context}
Question: {question}

Only returns the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    prompt template for QA retrieval for each vector stores
    """

    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return prompt


def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_token=512,
        temperature=0.5,
    )
    return llm


def retireval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retireval_qa_chain(llm, qa_prompt, db)
    return qa


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({"query": query})
    return response


# # API calls
# from fastapi import FastAPI
# from pydantic import BaseModel

# app = FastAPI()


# class item(BaseModel):
#     query: str


# @app.post("/")
# async def query(item: item):
#     print(item.query)
#     ans = final_result(item.query)
#     return ans.get("result")


# uvicorn app:app --host 0.0.0.0 --port 8000


# x = "Draft a contract for a sale of a car between two parties."
# ans = final_result(x)
# print(ans)


# Chain lit
@cl.on_chat_start
async def on_chat_start():
    files = None
    chain = qa_bot()
    msg = cl.Message(content="Starting Bharatgpt!")
    await msg.send()
    msg.content = "Hi, welcome to the BHARATGPT. what is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["Answer:"],
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\n\nSource document: " + str(sources)
    else:
        answer += f"\nNo source document found"

    await cl.Message(content=answer).send()


# DB_FAISS_PATH = "vectorestores/db_faiss"
# @cl.on_file_upload(accept=["application/pdf"], max_files=3, max_size_mb=2)
# async def upload_file(files: any):
#     """
#     Handle uploaded files.

#     Args:
#         files (list): List of uploaded file data.

#     Example:
#         [{
#             "name": "example.txt",
#             "content": b"File content as bytes",
#             "type": "text/plain"
#         }]
#     """
#     for file_data in files:
#         with tempfile.NamedTemporaryFile() as tmp:
#             tmp.write(file_data["content"])
#             temp_file_name = tmp.name
#         loader = DirectoryLoader(temp_file_name, glob="*.pdf", loader_cls=PyPDFLoader)
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         text = text_splitter.split_documents(documents)
#         embeddings = HuggingFaceEmbeddings(
#             model_name="sentence-transformers/all-MiniLM-L6-v2",
#             model_kwargs={"device": "cpu"},
#         )

#         # create vector and save
#         db = FAISS.from_documents(text, embeddings)
#         db.save_local(DB_FAISS_PATH)
