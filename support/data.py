import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TypedDict

from support.settings import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "datasets"

def ensure_data(directory: Path) -> None:
    for filename in ("reviews.jsonl.gz", "products.jsonl.gz"):
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"Copy the provided datasets folder first: {filename}")

def rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, text in enumerate(stream, start=1):
            if text.strip():
                yield line_number, json.loads(text)

class ReviewRecord(TypedDict):
    review_id: str
    product_id: str
    text: str
    rating: float


class ProductRecord(TypedDict):
    product_id: str
    title: str
    features: list[str]
    description: list[str]
    categories: list[str]
    family: str


def load(directory: Path) -> tuple[list[ReviewRecord], dict[str, ProductRecord]]:
    products: dict[str, ProductRecord] = {}
    for line_number, product in rows(directory / "products.jsonl.gz"):
        product_id = product.get("parent_asin")
        title = product.get("title") or ""
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError(f"Invalid product ID at line {line_number}")
        if not isinstance(title, str):
            raise ValueError(f"Invalid product title at line {line_number}")
        if product_id in products:
            raise ValueError(f"Duplicate product ID at line {line_number}")
        features = product.get("features") or []
        if not isinstance(features, list) or any(not isinstance(item, str) for item in features):
            raise ValueError(f"Invalid features at line {line_number}")
        description = product.get("description") or []
        categories = product.get("categories") or []
        family = product.get("family")
        for values in (description, categories):
            if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
                raise ValueError(f"Invalid catalog text at line {line_number}")
        if not isinstance(family, str) or not family:
            raise ValueError(f"Invalid family at line {line_number}")
        products[product_id] = {
            "product_id": product_id, "title": title, "features": features,
            "description": description, "categories": categories, "family": family,
        }

    records: list[ReviewRecord] = []
    for line_number, review in rows(directory / "reviews.jsonl.gz"):
        product_id = review.get("parent_asin")
        rating = review.get("rating")
        text = review.get("text") or ""
        if not isinstance(product_id, str) or not product_id.strip():
            raise ValueError(f"Invalid product ID at review line {line_number}")
        if isinstance(rating, bool) or not isinstance(rating, (int, float)):
            raise ValueError(f"Invalid rating at review line {line_number}")
        if rating not in (1, 2, 3, 4, 5) or not isinstance(text, str):
            raise ValueError(f"Invalid review at line {line_number}")
        source_line = review.get("source_line")
        if isinstance(source_line, bool) or not isinstance(source_line, int) or source_line < 1:
            raise ValueError(f"Invalid original line at review line {line_number}")
        records.append({
            "review_id": f"Electronics:{source_line}",
            "product_id": product_id,
            "text": text,
            "rating": float(rating),
        })
    return records, products


def catalog_text(product: ProductRecord) -> str:
    unir_textos = [product["title"].strip(), " ".join(product["features"]), " ".join(product["description"])] 

    return " ".join(unir_textos).strip()




def review_texts(records: list[ReviewRecord]) -> tuple[list[str], list[str], list[str]]:
    texts: list[str] = []
    review_ids: list[str] = []
    review_product_ids: list[str] = []
    for record in records:
        if record["text"].strip() != "":
            texts.append(record["text"].strip())
            review_ids.append(record["review_id"])
            review_product_ids.append(record["product_id"])
    return texts, review_ids, review_product_ids