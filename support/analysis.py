import math
from typing import TypedDict

import torch

from support.search import SearchIndex

STARS = (1, 2, 3, 4, 5)


class ProductAnalysis(TypedDict):
    product_id: str
    title: str | None
    review_count: int
    star_counts: dict[str, int]
    star_proportions: dict[str, float]
    mean: float | None
    median: float | None


def analyze_product(index: SearchIndex, product_id: str) -> ProductAnalysis:
    if not isinstance(product_id, str) or not product_id.strip():
        raise ValueError("product_id must not be empty")
    product_id = product_id.strip()

    stars = torch.tensor(index.ratings.get(product_id, []), dtype=torch.float64)
    review_count = int(stars.numel())

    # bincount cuenta 0..5; descartamos la posición 0 (no existen 0 estrellas)
    counts = torch.bincount(stars.long(), minlength=len(STARS) + 1)[1:]
    if review_count > 0:
        proportions = counts.double() / review_count
        mean: float | None = float(stars.mean())
        median: float | None = float(torch.quantile(stars, 0.5))  # promedia los centrales si n es par
    else:
        proportions = torch.zeros(len(STARS), dtype=torch.float64)
        mean = None
        median = None

    product = index.products.get(product_id)
    return {
        "product_id": product_id,
        "title": product["title"] if product is not None else None,
        "review_count": review_count,
        "star_counts": {str(s): int(c) for s, c in zip(STARS, counts.tolist())},
        "star_proportions": {str(s): float(p) for s, p in zip(STARS, proportions.tolist())},
        "mean": mean,
        "median": median,
    }



class MeanDifference(TypedDict):
    product_id: str
    mean_difference: float | None  # media del producto menos media del primero


class ProductComparison(TypedDict):
    reference_product_id: str
    products: list[ProductAnalysis]
    mean_differences: list[MeanDifference]


def compare_products(index: SearchIndex, product_ids: list[str]) -> ProductComparison:
    if isinstance(product_ids, str) or not isinstance(product_ids, list):
        raise ValueError("product_ids must be a list of product IDs")
    if len(product_ids) < 2:
        raise ValueError("product_ids must contain at least two product IDs")
    if any(not isinstance(pid, str) or not pid.strip() for pid in product_ids):
        raise ValueError("product_ids must not contain empty IDs")
    cleaned = [pid.strip() for pid in product_ids]
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("product_ids must be distinct")

    summaries = [analyze_product(index, pid) for pid in cleaned]

    # NaN marca una media ausente; NaN - x = NaN, así la diferencia también queda ausente
    means = torch.tensor(
        [s["mean"] if s["mean"] is not None else math.nan for s in summaries],
        dtype=torch.float64,
    )
    differences = means - means[0]

    mean_differences: list[MeanDifference] = [
        {"product_id": pid, "mean_difference": None if math.isnan(d) else float(d)}
        for pid, d in zip(cleaned, differences.tolist())
    ]
    return {
        "reference_product_id": cleaned[0],
        "products": summaries,
        "mean_differences": mean_differences,
    }
