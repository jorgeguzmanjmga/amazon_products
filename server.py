from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from support import analysis, search
from support.analysis import ProductAnalysis, ProductComparison

# Datos, modelo y embeddings se preparan una sola vez, al iniciar el servidor
print("Preparando datos y embeddings...")
index = search.build_index()
print(f"Índice listo: {len(index.product_ids)} productos")

mcp = MCPServer("Amazon reviews")


@mcp.tool()
def search_products(query: str, top_k: int = 5) -> list[dict[str, str | float | int]]:
    """Search Amazon Electronics products by semantic similarity to an English query.

    Each product vector combines its catalog text (title, features, description)
    with the average of its review embeddings, 50 % each. Returns up to top_k
    products (1-20) in descending cosine similarity, with product_id, title,
    similarity and review_count. Similarity is not a probability or a quality score.
    """
    try:
        return search.search_products(index, query, top_k)
    except ValueError as err:
        raise ToolError(str(err)) from err


@mcp.tool()
def analyze_product(product_id: str) -> ProductAnalysis:
    """Summarize the star ratings of one product in the selected reviews.

    Returns review_count, counts and proportions for 1-5 stars, mean and median.
    An unknown product returns zero counts and null mean/median.
    """
    try:
        return analysis.analyze_product(index, product_id)
    except ValueError as err:
        raise ToolError(str(err)) from err


@mcp.tool()
def compare_products(product_ids: list[str]) -> ProductComparison:
    """Compare the ratings of two or more distinct products.

    Returns the analysis of each product and its mean difference against the
    first product. A difference is null when either mean is missing.
    """
    try:
        return analysis.compare_products(index, product_ids)
    except ValueError as err:
        raise ToolError(str(err)) from err


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8000)
