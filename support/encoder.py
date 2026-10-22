
import torch
from sentence_transformers import SentenceTransformer
from support.settings import ModelConfig
import torch.nn.functional as F
from support.data import ProductRecord, ReviewRecord, catalog_text, review_texts



def load_model(config: ModelConfig) -> SentenceTransformer:
    model = SentenceTransformer(
    config.model_name, device=config.device
    )
    return model


def encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> torch.Tensor:
    
    review_vectors = model.encode(
    texts,
    batch_size=batch_size,
    convert_to_tensor=True,
    normalize_embeddings=True,
    )
    return review_vectors




def build_product_matrix(
    model: SentenceTransformer,
    batch_size: int,
    records: list[ReviewRecord],
    products: dict[str, ProductRecord],
    ) -> tuple[torch.Tensor, list[str]]:
    product_ids = sorted(products)  # orden determinista; la fila i es product_ids[i]

    
    catalog_vectors = encode(model, 
                             [catalog_text(products[pid]) for pid in product_ids],
                             batch_size) 
    texts, _, review_product_ids = review_texts(records)
    review_vectors = encode(model, 
                            texts,
                            batch_size)

    positions: dict[str, list[int]] = {}
    for i, pid in enumerate(review_product_ids):
        positions.setdefault(pid, []).append(i)  # fila i de review_vectors es de pid

    rows: list[torch.Tensor] = []
    for i, pid in enumerate(product_ids):
        if pid in positions:
            review_summary = F.normalize(review_vectors[positions[pid]].mean(dim=0), dim=0)
            combined = F.normalize(0.5 * catalog_vectors[i] + 0.5 * review_summary, dim=0)
        else:
            combined = catalog_vectors[i]
        rows.append(combined)

    return torch.stack(rows), product_ids
