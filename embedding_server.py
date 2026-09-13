from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
model_path = "/storage/models/bge-m3"
# 强制CPU加载
model = SentenceTransformer(model_path, device="cpu")

class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = "bge-m3"

@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingRequest):
    texts = req.input
    embeddings = model.encode(texts, convert_to_tensor=False)
    if isinstance(texts, str):
        texts = [texts]
        embeddings = [embeddings]
    res_data = []
    for idx, emb in enumerate(embeddings):
        res_data.append({
            "object": "embedding",
            "embedding": emb.tolist(),
            "index": idx
        })
    return {
        "object": "list",
        "data": res_data,
        "model": req.model
    }

if __name__ == "__main__":
    uvicorn.run("run_embedding:app", host="192.168.1.200", port=8001)
