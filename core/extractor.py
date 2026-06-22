
import fitz  
import base64
from pathlib import Path
from PIL import Image
import io

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
        RecursiveCharacterTextSplitter = None

def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list:
    """Splits the input text into chunks of specified size with overlap.
    """
    if not text:
        return []
    if RecursiveCharacterTextSplitter is not None:
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            return splitter.split_text(text)
        except Exception:
            pass
    step = max(1, chunk_size - chunk_overlap)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), step)]
    return chunks



def extract_text_from_pdf(pdf_path: str, image_dir: str = "extracted_images") -> tuple:
    """
    Extracts text and images from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
        image_dir (str): The directory to save extracted images.

    Returns:
        tuple: A tuple of (text_chunks, images). Each text chunk is a dict
               with keys `text` and `page`. Each image is a dict with
               `image_path`, `page`, `type`, and `image_b64`.
    """
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    text_chunks = []
    images = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        chunks = chunk_text(text)
        for c in chunks:
            text_chunks.append({
                "text": c,
                "page": i + 1,
                "source": Path(pdf_path).name,
            })
        image_list = page.get_images(full=True)
        for image_index, image in enumerate(image_list):
            xref = image[0]
            base_image = doc.extract_image(xref)
            image_byte = base_image["image"]
            image_ext = base_image["ext"]
            try:
                image_pil=Image.open(io.BytesIO(image_byte)).convert("RGB")
            except Exception as e:
                continue
            image_filename=f"{Path(image_dir)}/{Path(pdf_path).stem}_page_{i+1}_img_{image_index+1}.{image_ext}"
            image_pil.save(image_filename)
            buffered_image=io.BytesIO()
            image_pil.save(buffered_image, format="JPEG")
            b64 = base64.b64encode(buffered_image.getvalue()).decode("utf-8")
            images.append({
                "image_path": image_filename,
                "page": i + 1,
                "type": "image",
                "image_b64": b64,
                "source": Path(pdf_path).name,
            })
    doc.close()
    print(f"Extracted {len(text_chunks)} text chunks and {len(images)} images from {pdf_path}")
    return text_chunks, images
