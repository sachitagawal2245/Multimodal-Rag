#rertriever.py
import numpy as np
import faiss
from sentence_transformers import CrossEncoder
from core.embedder import embed_query

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
def retrieve(query: str, index, metadata: list, top_k: int = 10, rerank_top_n: int = 5) -> list:
    """Retriver function to get top-k relevant items from the index based on cosine similarity"""
    query_embed=embed_query(query).astype("float32")
    scores,indices=index.search(query_embed.reshape(1,-1),top_k)
    candidates=[]
    for score,idx in zip(scores[0],indices[0]):
        if idx == -1:
            continue
        item = metadata[idx].copy()
        item["faiss_score"] = float(score)
        candidates.append(item)
    if not candidates:
        return []
    rerank_pairs=[]
    for items in candidates:
        if items["type"]=="text":
            rerank_pairs.append((query, " " + items["text"]))
        else:
            rerank_pairs.append((query, f"Image on page {items['page']}"))
    rerank_scores=reranker.predict(rerank_pairs)
    for item, rscore in zip(candidates, rerank_scores):
        item["rerank_score"] = float(rscore)
    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:rerank_top_n]


def separate_results(results: list) -> tuple:
    """
    Splits retrieved results into text chunks and images.
    Useful for building an LLM prompt separately.
    """
    texts = [r for r in results if r["type"] == "text"]
    images = [r for r in results if r["type"] == "image"]
    return texts, images 
