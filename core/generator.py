# core/generator.py
import os
import base64
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

_llm = None
_output_parser = None
_prompt = None
_chain = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ-API-KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Please export it or create a .env file with GROQ_API_KEY=<your_api_key>."
            )
        _llm = ChatGroq(
            api_key=api_key,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2,
            max_tokens=1024
        )
    return _llm


def get_output_parser() -> StrOutputParser:
    global _output_parser
    if _output_parser is None:
        _output_parser = StrOutputParser()
    return _output_parser


def get_prompt() -> ChatPromptTemplate:
    global _prompt
    if _prompt is None:
        _prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on 
     the provided document context.
     If the answer is not in the context, say so clearly.
     Always consider the conversation history for follow-up questions.
     
     Context:
     {context}
     """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
    return _prompt


def get_chain():
    global _chain
    if _chain is None:
        _chain = (
            RunnablePassthrough.assign(
                context=lambda x: x["context"],
                chat_history=lambda x: x.get("chat_history", []),
                input=lambda x: x["input"]
            )
            | get_prompt()
            | get_llm()
            | get_output_parser()
        )
    return _chain


def results_to_documents(text_results: list, image_results: list) -> list[Document]:
    """
    Converts retrieved chunks and image metadata to LangChain Documents.
    """
    docs = []
    for r in text_results:
        docs.append(Document(
            page_content=r["text"],
            metadata={
                "page": r["page"],
                "type": "text",
                "source": r.get("source", "unknown")
            }
        ))

    for img in image_results:
        if "image_b64" not in img and "image_path" in img:
            with open(img["image_path"], "rb") as f:
                img["image_b64"] = base64.b64encode(f.read()).decode("utf-8")
        docs.append(Document(
            page_content=f"[Image on page {img['page']}]",
            metadata={
                "page": img["page"],
                "type": "image",
                "source": img.get("source", "unknown"),
                "image_path": img.get("image_path", ""),
                "image_b64": img.get("image_b64", "")
            }
        ))

    return docs


def format_docs(docs: list[Document]) -> str:
    """Formats text documents into a single context string."""
    return "\n\n".join([
        f"[Page {d.metadata['page']}]: {d.page_content}"
        for d in docs if d.metadata["type"] == "text"
    ])


def create_multimodal_message(query: str,docs: list[Document],chat_history: list) -> HumanMessage:
    """
    Builds Groq-compatible multimodal HumanMessage
    with chat history + text context + images + query.
    """
    content = []
    if chat_history:
        history_text = ""
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Assistant: {msg.content}\n"
        content.append({
            "type": "text",
            "text": f"Chat History:\n{history_text}\n"
        })

    text_docs = [d for d in docs if d.metadata["type"] == "text"]
    image_docs = [d for d in docs if d.metadata["type"] == "image"]
    if text_docs:
        text_context = "\n\n".join([
            f"[Page {d.metadata['page']}]: {d.page_content}"
            for d in text_docs
        ])
        content.append({
            "type": "text",
            "text": f"Text Context:\n{text_context}\n"
        })
    for doc in image_docs:
        b64 = doc.metadata.get("image_b64")
        if b64:
            content.append({
                "type": "text",
                "text": f"\n[Image from page {doc.metadata['page']}]:\n"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            })
    content.append({
        "type": "text",
        "text": f"\nQuestion: {query}\nPlease answer based on the provided context and images."
    })

    return HumanMessage(content=content)


def generate_answer( query: str,text_results: list,image_results: list,chat_history: list) -> str:
    """
    Full generation pipeline with memory.
    - If images present → multimodal path (HumanMessage with image blocks)
    - If text only → LCEL chain with MessagesPlaceholder
    """
    try:
        docs = results_to_documents(text_results, image_results)
        llm = get_llm()
        output_parser = get_output_parser()
        if image_results:
            message = create_multimodal_message(query, docs, chat_history)
            response = llm.invoke([message])
            return output_parser.invoke(response.content)
        else:
            context_str = format_docs(docs)
            response = get_chain().invoke({
                "context": context_str,
                "chat_history": chat_history,
                "input": query
            })
            return response

    except Exception as e:
        return f"Error generating answer: {str(e)}"


def update_chat_history(chat_history: list,query: str,answer: str) -> list:
    """
    Appends latest Q&A pair to chat history.
    Call this after every generate_answer().
    """
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))
    return chat_history