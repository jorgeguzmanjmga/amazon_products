
from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer

from support.data import DATA_DIR, ProductRecord, load
from support.encoder import build_product_matrix, load_model, encode
from support.settings import load_settings


@dataclass(frozen=True)
class SearchIndex:
    model: SentenceTransformer
    matrix: torch.Tensor                 # (600, 384); fila i = product_ids[i]
    product_ids: list[str]
    products: dict[str, ProductRecord]
    ratings: dict[str, list[float]]      # product_id -> sus estrellas


def build_index() -> SearchIndex:
    config = load_settings()
    model = load_model(config)
    records, products = load(DATA_DIR)
    matrix, product_ids = build_product_matrix(model, config.batch_size, records, products)

    ratings: dict[str, list[float]] = {}
    for record in records:
        ratings.setdefault(record["product_id"], []).append(record["rating"])

    return SearchIndex(model, matrix, product_ids, products, ratings)

def search_products(index: SearchIndex, query: str, top_k: int = 5) -> list[dict[str, str | float | int]]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("top_k must be an integer between 1 and 20")

    query_vector = encode(index.model, [query], 1)[0]   # (384,)
    scores = index.matrix @ query_vector   # (600,); filas y consulta normalizadas -> coseno
    k = min(top_k, len(index.product_ids))
    top = torch.topk(scores, k)

    results: list[dict[str, str | float | int]] = []
    for score, position in zip(top.values.tolist(), top.indices.tolist()):
        pid = index.product_ids[position]
        results.append({
            "product_id": pid,
            "title": index.products[pid]["title"],
            "similarity": float(score),
            "review_count": len(index.ratings.get(pid, [])),
        })
    return results