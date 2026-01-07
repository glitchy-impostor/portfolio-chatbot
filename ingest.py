"""
Ingestion script for Krish's Resume Assistant
Run this once to create the vector index from data files.

Usage: python ingest.py
"""

import os
import pickle
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Configuration
DATA_DIR = "data"
CHUNK_SIZE = 300  # Smaller chunks for more precise retrieval
CHUNK_OVERLAP = 50  # Overlap to maintain context
EMBEDDING_MODEL = "text-embedding-3-small"

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping chunks with metadata.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk_words = words[i:i + CHUNK_SIZE]
        if len(chunk_words) < 50:  # Skip very small trailing chunks
            continue
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text,
            "source": source,
            "char_start": i,
        })
    
    return chunks

def main():
    print("Starting ingestion...")
    
    # Collect all chunks
    all_chunks = []
    
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(filepath):
            continue
            
        print(f"Processing: {filename}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = chunk_text(text, filename)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = []
    
    for i, chunk in enumerate(all_chunks):
        if i % 10 == 0:
            print(f"  Processing chunk {i+1}/{len(all_chunks)}")
        
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunk["text"]
        )
        embeddings.append(response.data[0].embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype("float32")
    
    # Create FAISS index
    print("\nCreating FAISS index...")
    dimension = len(embeddings[0])
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    
    # Save index and metadata
    print("Saving index and metadata...")
    faiss.write_index(index, "index.faiss")
    
    metadata = {
        "chunks": [{"text": c["text"], "source": c["source"]} for c in all_chunks],
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
    }
    
    with open("chunks.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print("\n✅ Ingestion complete!")
    print(f"   Index saved to: index.faiss")
    print(f"   Metadata saved to: chunks.pkl")
    print(f"   Total vectors: {index.ntotal}")

if __name__ == "__main__":
    main()
