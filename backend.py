from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.vectorstore.chroma_store import get_chroma_retriever
from src.rag.llms import get_openai_llm, get_local_llm
from src.rag.prompt import build_rag_pipeline
# from src.viz.plot import plot
import json

# ----- FastAPI Setup -----
app = FastAPI()

# Allow frontend (React) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev, later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Initialize RAG -----
retriever = get_chroma_retriever(path="./data/chroma")
llm = get_openai_llm()  # or get_local_llm("mistral")
rag = build_rag_pipeline(retriever, llm)

# ----- Request Schema -----
class ChatRequest(BaseModel):
    query: str

# ----- Endpoints -----
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    query = req.query.lower()
    res = rag.invoke({"query": req.query})
    answer = res['result']
    import re
    match = re.search(r'```json\s*(\{.*\})\s*```', answer, re.DOTALL)
    if match:
        import json
        data = json.loads(match.group(1))
        answer_text = data['answer']
    else:
        answer_text = answer
    
    return{
        "answer": answer_text,
        "sources": [doc.metadata for doc in res["source_documents"]],
    }

    # if "plot" in query or "map" in query:
    #     fig = plot() 
    #     return {
    #         "answer": "Here is the temperature/salinity profile.",
    #         "visualization": {"type": "plot", "data": fig.to_dict()}
    #     }
    # else:
    #     res = rag.invoke({"query": req.query})
    #     return {
    #         "answer": res["result"],
    #         "sources": [doc.metadata for doc in res["source_documents"]],
    #     }


# @app.get("/profile/{float_id}")
# def profile_endpoint(float_id: str):
#     fig = plot_profile(float_id)
#     return json.loads(fig.to_json())

# @app.get("/map/{float_id}")
# def map_endpoint():
#     fmap = plot_float_map()
#     return {"html": fmap._repr_html_()}

# @app.get("/salinity/{float_id}")
# def salinity_endpoint(float_id: str):
#     fig = plot_salinity(float_id)
#     return json.loads(fig.to_json())
