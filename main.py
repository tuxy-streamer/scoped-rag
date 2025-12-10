import sys

from image import process_images
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
    
    # Process PDFs
    pdf_chunks = process_pdfs()
    
    # Process Images (PNG, JPEG, JPG)
    image_chunks = process_images()
    
    # Combine all chunks
    all_chunks = pdf_chunks + image_chunks
    print(f"Total chunks: {len(all_chunks)} (PDFs: {len(pdf_chunks)}, Images: {len(image_chunks)})")
    
    store = create_vector_store(all_chunks)
    save_vector_store(store)

# Answer
answer = query_vector_store(store, question)
print(answer)
