from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from model import get_embedding, get_llm


def create_vector_store(docs: list[Document]) -> FAISS:
    embedding = get_embedding()
    store = FAISS.from_documents(docs, embedding)
    print(f" Created vector store with {len(docs)} documents")
    return store


def save_vector_store(store: FAISS, path: str = "faiss_index"):
    store.save_local(path)
    print(f"Saved to {path}")


def load_vector_store(path: str = "faiss_index") -> FAISS:
    embedding = get_embedding()
    store = FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)
    print(f"Loaded from {path}")
    return store


def create_rag_chain(store: FAISS):
    llm = get_llm()
    retriever = store.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template("""
        Context from documents:
        {context}
        
        Question: {question}
        
        Answer using only the context above:""")
        | llm
        | StrOutputParser()
    )
    return rag_chain


def query_vector_store(store: FAISS, question: str) -> str:
    chain = create_rag_chain(store)
    answer = chain.invoke(question)
    return answer


def query_with_sources(store: FAISS, question: str) -> dict[str, object]:
    retriever = store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    llm = get_llm()
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""

    answer = llm.invoke(prompt)
    return {
        "answer": answer,
        "sources": [doc.metadata.get("chunk_id", "N/A") for doc in docs],
        "context": context,
    }
