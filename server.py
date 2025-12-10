"""FastAPI backend for RAG system"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from text import process_pdfs
from image import process_images
from vector_store import (
    create_vector_store,
    load_vector_store,
    query_vector_store,
    query_with_sources,
    save_vector_store,
)

app = FastAPI(title="Scoped RAG API")

# Global store
store = None


class Query(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


class QueryWithSourcesResponse(BaseModel):
    answer: str
    sources: list[str]
    context: str


def get_store():
    """Load or create vector store"""
    global store
    if store is None:
        try:
            store = load_vector_store()
        except:
            print("Creating new index...")
            pdf_chunks = process_pdfs()
            image_chunks = process_images()
            all_chunks = pdf_chunks + image_chunks
            store = create_vector_store(all_chunks)
            save_vector_store(store)
    return store


@app.on_event("startup")
async def startup():
    """Load vector store on startup"""
    get_store()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(q: Query):
    """Query the RAG system"""
    try:
        s = get_store()
        answer = query_vector_store(s, q.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-with-sources", response_model=QueryWithSourcesResponse)
def query_sources(q: Query):
    """Query with source documents"""
    try:
        s = get_store()
        result = query_with_sources(s, q.question)
        return QueryWithSourcesResponse(
            answer=result["answer"],
            sources=result["sources"],
            context=result["context"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex")
def reindex():
    """Rebuild the vector index"""
    global store
    try:
        pdf_chunks = process_pdfs()
        image_chunks = process_images()
        all_chunks = pdf_chunks + image_chunks
        store = create_vector_store(all_chunks)
        save_vector_store(store)
        return {"status": "reindexed", "chunks": len(all_chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
