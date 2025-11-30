import sys

from text import process_pdfs
from vector_store import (
    create_vector_store,
    load_vector_store,
    query_vector_store,
    save_vector_store,
)

# Get question
question = sys.argv[1] if len(sys.argv) > 1 else "summarize documents"

# Auto-create index if missing
try:
    store = load_vector_store()
except:
    print("Creating index...")
    chunks = process_pdfs()
    store = create_vector_store(chunks)
    save_vector_store(store)

# Answer
answer = query_vector_store(store, question)
print(answer)
