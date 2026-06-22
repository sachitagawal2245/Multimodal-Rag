import streamlit as st
import os
import tempfile
from core.extractor import extract_text_from_pdf
from core.embedder import embed_text_chunks, embed_images, load_model
from core.indexer import build_index, save_index, load_index
from core.retriever import retrieve, separate_results
from core.generator import generate_answer, update_chat_history

st.set_page_config(page_title="PDF Chatbot", page_icon=":brain",layout="wide")
st.title("PDF Chatbot ")
st.caption("Upload a PDF file and ask questions about its content.")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        if not st.session_state.pdf_processed:
            with st.spinner("Processing PDF... this may take a minute"):
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                text_chunks, image_metadata = extract_text_from_pdf(tmp_file_path)
                load_model()
                text_embeddings = embed_text_chunks(text_chunks)
                image_embeddings = embed_images(image_metadata)
                index, metadata = build_index(text_embeddings, image_embeddings)
                st.session_state.index = index
                st.session_state.metadata = metadata
                st.session_state.pdf_processed = True
                os.unlink(tmp_file_path)
                st.success("Pdf processed successfully you can now ask questions about the  content.")
        else:
            st.info("Pdf already processed you can ask questions about the  content.")

    if st.button("reset"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.session_state.index = None
        st.session_state.metadata = None
        st.session_state.pdf_processed = False
        st.success("Chatbot has been reset You can upload a new Pdf.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.text_input("Ask a question about the Pdf content:")
if query:
    if not st.session_state.pdf_processed:
        st.warning("Please upload and process a Pdf first.")
    else:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            st.markdown("Generating answer...")
            results = retrieve(query, st.session_state.index, st.session_state.metadata)
            text_results, image_results = separate_results(results)
            generated_answer = generate_answer(query, text_results, image_results, st.session_state.chat_history)
            st.markdown(generated_answer)

        st.session_state.chat_history = update_chat_history(st.session_state.chat_history,query,generated_answer,)
        st.session_state.messages.append({"role": "assistant", "content": generated_answer,})
