from typing import Any

from mcp import Client


async def demonstrate() -> None:
    calls: list[tuple[str, dict[str, Any]]] = [
        ("search_products", {"query": "A portable waterproof Bluetooth speaker", "top_k": 3}),
        ("analyze_product", {"product_id": "B0BCTJMN63"}),
        ("compare_products", {"product_ids": ["B0BCTJMN63", "B09MMWM2Y8"]}),
        ("analyze_product", {"product_id": "UNKNOWN_PRODUCT"}),
        ("search_products", {"query": "delivery", "top_k": 0}),
        ("search_products", {"query": "Comfortable headphones with good sound", "top_k": 2}),
        # Casos adicionales: diferencia nula, IDs repetidos, ID en blanco y recuperación
        ("compare_products", {"product_ids": ["B0BCTJMN63", "UNKNOWN_PRODUCT"]}),
        ("compare_products", {"product_ids": ["B0BCTJMN63", "B0BCTJMN63"]}),
        ("analyze_product", {"product_id": "   "}),
        ("compare_products", {"product_ids": ["B0BCTJMN63", "B0B786PFYJ", "B09WWG19YS"]}),
    ]
    async with Client("http://127.0.0.1:8000/mcp") as client:
        tools = await client.list_tools()
        print("Tools:", [tool.name for tool in tools.tools])
        for name, arguments in calls:
            print(f"\n{name}: {arguments}")
            result = await client.call_tool(name, arguments)
            print("Tool error:", result.is_error)
            print(result.model_dump_json(indent=2))
