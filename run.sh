#!/bin/bash

echo "[1/3] Starting Chroma vector database..."
docker-compose up -d
sleep 6

echo "[2/3] Initializing vector DB with chunking, embedding, and metadata..."
python3 db/init_vector_db.py
if [ $? -ne 0 ]; then
    echo "❌ Embedding pipeline failed. Check logs for errors."
    exit 1
fi

echo "[3/3] Validating vector DB status..."
python3 db/vector_db_client.py --status
if [ $? -eq 0 ]; then
    echo "✔ Vector DB ready. Documents and embeddings are queryable."
else
    echo "❌ Vector DB validation failed."
    exit 2
fi
