import chromadb
from chromadb.config import Settings
from config import VECTOR_DB_HOST, VECTOR_DB_PORT, VECTOR_DB_COLLECTION

def get_chroma_client():
    return chromadb.Client(Settings(chroma_api_impl="rest", chroma_server_host=VECTOR_DB_HOST, chroma_server_http_port=VECTOR_DB_PORT))

def get_collection(client=None):
    if client is None:
        client = get_chroma_client()
    return client.get_collection(name=VECTOR_DB_COLLECTION)

if __name__ == '__main__':
    import sys
    # Validation: print indexed document count
    try:
        collection = get_collection()
        print(f"[VectorDB] Indexed chunk count: {collection.count()}")
        sys.exit(0)
    except Exception as e:
        print(f"[VectorDB] Error: {str(e)}")
        sys.exit(1)
