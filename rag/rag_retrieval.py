from typing import List, Dict
from db.vector_db_client import get_collection
from sentence_transformers import SentenceTransformer
from config import VECTOR_DIM

MODEL_NAME = 'all-MiniLM-L6-v2'
model = SentenceTransformer(MODEL_NAME)
assert model.get_sentence_embedding_dimension() == VECTOR_DIM


def retrieve_top_k(query: str, k: int = 5) -> List[Dict]:
    """
    Retrieve the top-k semantically relevant chunks for a given query.
    Args:
        query: User's search input (string)
        k: Number of relevant chunks to return
    Returns:
        List of dicts, each containing 'chunk', 'score', and metadata keys
    """
    # IMPLEMENTATION REQUIRED
    pass
