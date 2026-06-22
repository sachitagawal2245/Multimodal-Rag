#embedeer.py
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import base64
import io
import torch
clip_model=None
clip_processor=None
def load_model():
    global clip_model,clip_processor
    if clip_model is None or clip_processor is None:
        clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_path:str):
    load_model()
    """Embeds an image using the CLIP model."""
    if isinstance(image_path, str):
        image=Image.open(image_path).convert("RGB")
    else:
        image=image_path
    inputs=clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():

        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text:str):
    """Embeds text using the CLIP model."""
    load_model()
    inputs = clip_processor(
        text=text, 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=77  
    )
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text_chunks(text_chunks: list[dict]) -> list[dict]:
    """Embeds all text chunks and adds embedding + type to each dict."""
    for chunk in text_chunks:
        chunk["embedding"] = embed_text(chunk["text"])
        chunk["type"] = "text"
    return text_chunks


def embed_images(images: list[dict]) -> list[dict]:
    """Embeds all images and adds embedding + type to each dict."""
    for img in images:
        img["embedding"] = embed_image(img["image_path"])
        img["type"] = "image"
    return images


def embed_query(query: str) -> np.ndarray:
    """Embeds user query — same CLIP text encoder as chunks."""
    load_model()
    return embed_text(query)