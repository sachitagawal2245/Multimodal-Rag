#  Multimodal RAG — PDF Chatbot with Text + Image Understanding

A Retrieval-Augmented Generation (RAG) system that understands both **text and images** from PDF documents. Ask questions about charts, diagrams, and text — all in one conversational interface.

---

##  Demo

| Query | Type |
|---|---|
| "Who is the CEO?" | Text retrieval |
| "What does the quarterly revenue chart show?" | Image/chart understanding |
| "Explain the pipeline architecture diagram" | Diagram interpretation |
| "What is the cricket score?" | Out-of-context rejection ✅ |

---

##  Architecture

```
PDF
 ├── Text → RecursiveCharacterTextSplitter → CLIP Text Embeddings
 └── Images → PIL → CLIP Image Embeddings
                        ↓
                   FAISS Index (IndexFlatIP)
                        ↓
              Query → CLIP Embed → FAISS Search (top-10)
                        ↓
               CrossEncoder Reranking (top-5)
                        ↓
          Llama 4 Scout Vision via Groq (text + image context)
                        ↓
                  Streamlit Chat UI
```

---

## 🔧 Tech Stack

| Component | Tool |
|---|---|
| PDF Parsing | PyMuPDF (fitz) |
| Text Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | CLIP ViT-B-32 (text + image, unified vector space) |
| Vector Store | FAISS (cosine similarity) |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| LLM | Llama 4 Scout Vision via Groq API |
| Chain | LangChain LCEL with conversational memory |
| UI | Streamlit |

---

## Project Structure

```
multimodal-rag/
├── core/
│   ├── extractor.py      # PDF → text chunks + images
│   ├── embedder.py       # CLIP embeddings (text + image)
│   ├── indexer.py        # FAISS index build/save/load
│   ├── retriever.py      # Dense retrieval + CrossEncoder reranking
│   └── generator.py      # LangChain LCEL chain + Groq vision LLM
├── app.py                # Streamlit UI
├── requirements.txt
├── .env.example
└── .gitignore
```

---

##  Setup

**1. Clone the repo**
```bash
git clone https://github.com/sachitagawal2245/multimodal-rag.git
cd multimodal-rag
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**4. Add your Groq API key**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```
Get your free API key at [console.groq.com](https://console.groq.com)

**5. Run the app**
```bash
python -m streamlit run app.py
```

---

## Key Design Decisions

**Why CLIP for both text and images?**
CLIP maps text and images into the same vector space. This means a text query like "show me the revenue chart" can directly retrieve relevant images — no separate image captioning pipeline needed.

**Why CrossEncoder reranking?**
FAISS retrieves by approximate similarity (recall). The CrossEncoder re-scores the top-10 candidates against the query for precision, significantly improving answer quality.

**Why two generation paths?**
- Text-only queries → LangChain LCEL chain with `MessagesPlaceholder` for clean memory handling
- Multimodal queries → Raw `HumanMessage` with base64 image blocks (OpenAI-compatible format that Groq Vision expects)

---

##  Requirements

```
streamlit
python-dotenv
pymupdf
pillow
transformers
torch
sentence-transformers
faiss-cpu
langchain
langchain-core
langchain-groq
langchain-community
groq
```

---

##  Future Improvements

- Image captioning before indexing for better image retrieval
- Metadata filtering (retrieve by page number, section)
- Support for multiple PDFs simultaneously
- Deployement on Streamlit Cloud / HuggingFace Spaces

---

## 👤 Author

**Sachit Agarwal**
- GitHub: [@sachitagawal2245](https://github.com/sachitagawal2245)
- LinkedIn: [Sachit Agarwal](https://www.linkedin.com/in/sachit-agarwal)

---

⭐ If you found this useful, consider giving it a star!
