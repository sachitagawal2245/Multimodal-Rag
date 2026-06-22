#i dexer.py
import faiss
import numpy as np
import pickle
import os

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.pkl"

def build_index(text_chunks: list[dict], images: list[dict])->tuple:
    """Build a FAISS index from text chunks and images."""
    all_item=text_chunks+images
    embedding=np.array([item["embedding"] for item in all_item]).astype("float32")
    embedding_dim=embedding.shape[1]
    index=faiss.IndexFlatIP(embedding_dim)
    index.add(embedding)
    meta_data=[]
    for items in all_item:
        meta={k:v for k,v in items.items() if k not in ["embedding"]}
        meta_data.append(meta)   
    print(f"Built FAISS index with {index.ntotal} items.")
    return index,meta_data

def save_index(index,meta_data):
    """saves the FAISS index and metadata to disk"""
    faiss.write_index(index,INDEX_PATH)
    with open(METADATA_PATH,"wb") as f:
        pickle.dump(meta_data,f)
    print(f"Saved FAISS index to {INDEX_PATH} and metadata to {METADATA_PATH}")

def load_index() -> tuple:
    """Loads FAISS index and metadata from disk."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Index not found. Please upload and process a PDF first.")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    print(f" Loaded index with {index.ntotal} vectors")
    return index, metadata

