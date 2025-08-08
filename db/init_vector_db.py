import json
import sys
from sentence_transformers import SentenceTransformer
from config import RAW_DOCS_PATH, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DIM, VECTOR_DB_COLLECTION
from db.vector_db_client import get_collection
import itertools
import time

MODEL_NAME = 'all-MiniLM-L6-v2'

model = SentenceTransformer(MODEL_NAME)
assert model.get_sentence_embedding_dimension() == VECTOR_DIM

def chunk_document(text, chunk_size=512, chunk_overlap=200):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks

if __name__ == '__main__':
    try:
        with open(RAW_DOCS_PATH, 'r') as f:
            raw_docs = json.load(f)
        print(f"[Init] Loaded {len(raw_docs)} raw documents from {RAW_DOCS_PATH}.")
    except Exception as e:
        print(f"[Init][Error] Failed to read raw_docs.json: {e}")
        sys.exit(1)

    meta_keys = ['doc_id', 'category', 'priority', 'date']
    collection = get_collection()
    all_texts, all_metadatas = [], []
    doc_id_counter = 0
    for doc in raw_docs:
        doc_id = doc.get('doc_id', f'DOC{doc_id_counter}')
        doc_id_counter += 1
        chunks = chunk_document(doc['content'], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        for idx, chunk in enumerate(chunks):
            all_texts.append(chunk)
            meta = {key: doc.get(key, 'NA') for key in meta_keys}
            meta['chunk_idx'] = idx
            all_metadatas.append(meta)
    print(f"[Init] Generated {len(all_texts)} total chunks for embedding.")

    batch_size = 48
    total = len(all_texts)
    t0 = time.time()
    for i in range(0, total, batch_size):
        batch_texts = all_texts[i:i+batch_size]
        batch_metas = all_metadatas[i:i+batch_size]
        batch_vectors = model.encode(batch_texts, show_progress_bar=False)
        # Use string ids for Chroma
        ids = [f'chunk_{i+j}' for j in range(len(batch_texts))]
        collection.add(documents=batch_texts, embeddings=batch_vectors.tolist(), metadatas=batch_metas, ids=ids)
        print(f"[Init] Inserted batch {i//batch_size+1} ({len(ids)} chunks).")
    t1 = time.time()
    print(f"[Init] Finished embedding and ingestion. {total} chunks indexed in {t1-t0:.1f} seconds.")
    count = collection.count()
    print(f"[Init] Chroma collection now contains {count} chunks.")
    if count == total:
        print("[Init] Vector DB is ready and all chunks present.")
        sys.exit(0)
    else:
        print(f"[Init][Error] Chunk count mismatch: expected {total}, found {count}.")
        sys.exit(1)
