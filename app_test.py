from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class item(BaseModel):
    query: str


# @app.get("/")
# def read_root():
#     print("response sent!")
#     return {"Hello": "bharat"}
ans = {
    "query": "If I make a contract with an unsound mind person then is it contract\xa0valid?",
    "result": "According to section 12 of the Indian Contract Act, 1872, a person who is usually of unsound mind but occasionally of sound mind may make a contract when they are of sound mind. However, if the person making the contract is of unsound mind at the time of contract formation, the contract may not be valid. It is important to note that the Act does not provide any specific definition or test for determining whether a person is of sound mind or unsound mind. Therefore, in the absence of any evidence to the contrary, it can be presumed that the person making the contract was of sound mind at the time of contract formation.",
    "source_documents": {
        "title": "Indian Contract Act, 1872",
    },
}


@app.post("/")
async def query(item: item):
    print(item.query)
    return ans
