"""
本地 OpenAI 兼容 embedding server。
端口 8765，模型名 all-MiniLM-L6-v2。
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import uvicorn, numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class EmbedRequest(BaseModel):
    input: Union[str, list]
    model: str = "all-MiniLM-L6-v2"

@app.post("/v1/embeddings")
def embeddings(req: EmbedRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    vecs = model.encode(texts, normalize_embeddings=True).tolist()
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": i, "embedding": v}
                 for i, v in enumerate(vecs)],
        "model": req.model,
        "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": 0},
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
