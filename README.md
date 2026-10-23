# Búsqueda y análisis de productos de Amazon con MCP

Servidor MCP por HTTP local (Streamable HTTP) para buscar y analizar productos de
**Amazon Reviews 2023 · Electronics**, sobre una selección de **600 productos y
6 992 reseñas**. La búsqueda usa embeddings de
`sentence-transformers/all-MiniLM-L6-v2` y operaciones de PyTorch.

## Requisitos

- [uv](https://docs.astral.sh/uv/). uv instala Python 3.13 según `.python-version`.
- Conexión a internet la primera vez, para descargar el modelo a la caché local
  de Hugging Face. Los pesos no se incluyen en el proyecto.

## Ejecución

Instalar las dependencias exactas de `uv.lock`:

```bash
uv sync --locked
```

**Terminal 1 — servidor.** Al iniciar, lee los datos, carga el modelo y calcula
los embeddings (unos 75 s en CPU). No necesita ejecutar la notebook.

```bash
uv run --locked python server.py
```

Está listo cuando muestra `Índice listo: 600 productos` y escucha en
`http://127.0.0.1:8000/mcp`.

**Terminal 2 — cliente de demostración.** Descubre las tres herramientas y
ejecuta llamadas válidas e inválidas en la misma conexión.

```bash
uv run --locked python main.py
```

**Comprobación de tipos:**

```bash
uv run --locked mypy --strict server.py client.py main.py support
```

Los vectores no se guardan en disco: se recalculan en memoria cada vez que se
inicia el servidor, así que no hay nada que regenerar. El modelo, el tamaño de
lote y el dispositivo se configuran en `config.json`.

## Estructura

```text
server.py          Servidor MCP y registro de las tres herramientas
client.py          Llamadas de demostración
main.py            Ejecuta el cliente
config.json        Modelo, tamaño de lote y dispositivo
support/
├── settings.py    Carga y validación de config.json
├── data.py        Lectura de los JSONL y preparación de textos e IDs
├── encoder.py     Modelo, codificación y matriz de productos (600, 384)
├── search.py      Índice en memoria y search_products
└── analysis.py    analyze_product y compare_products
datasets/          products.jsonl.gz y reviews.jsonl.gz
```

## Herramientas

| Herramienta | Resultado |
|---|---|
| `search_products(query, top_k=5)` | Hasta `top_k` productos en orden descendente de similitud: `product_id`, `title`, `similarity`, `review_count`. |
| `analyze_product(product_id)` | `review_count`, conteos y proporciones de 1 a 5 estrellas, `mean` y `median`. |
| `compare_products(product_ids)` | El análisis de cada producto y `mean_difference` de cada uno respecto al primero. |

Comportamiento ante entradas inválidas:

- Consulta o ID en blanco, `top_k` fuera de 1–20 o no entero, menos de dos IDs o
  IDs repetidos: error claro (`is_error: true`) con el motivo.
- Producto desconocido: no es un error; devuelve conteos y proporciones en cero y
  `mean`/`median` en `null`. En una comparación, su diferencia también es `null`.
- Un error no afecta las llamadas siguientes: el índice se prepara una sola vez y
  no se modifica.

## Cómo funciona la búsqueda

1. **Catálogo:** se codifica `título + características + descripción` de cada producto.
2. **Reseñas:** se codifica cada reseña con texto por separado. Se promedian los
   vectores de cada producto y se normaliza el promedio.
3. **Producto:** `normalizar(0.5 · catálogo + 0.5 · reseñas)`. La fila `i` de la
   matriz `(600, 384)` corresponde a `product_ids[i]`.
4. **Consulta:** se codifica con el mismo modelo y la misma normalización.
   Como todos los vectores tienen longitud 1, `matriz @ consulta` da la similitud
   coseno con los 600 productos, y `torch.topk` selecciona los más cercanos.

Las estrellas no intervienen en la búsqueda: solo se usan en el análisis.

## Ejemplos de consultas

### 1. Consulta de catálogo: `"A portable waterproof Bluetooth speaker"`

| Similitud | Reseñas | Producto |
|---:|---:|---|
| 0.794 | 11 | Altec Lansing Mini H2O — Waterproof Bluetooth Speaker |
| 0.790 | 13 | EDUPLINK Waterproof Portable Bluetooth Speaker |
| 0.758 | 27 | Bose SoundLink Micro (Waterproof) |
| 0.736 | 17 | JBL Clip 3 — Waterproof Portable Bluetooth Speaker |
| 0.719 | 8 | JBL Flip 5 Waterproof Portable Wireless Bluetooth Speaker |

**Interpretación.** Los cinco resultados son altavoces Bluetooth resistentes al
agua. La consulta describe un *tipo de producto* con las mismas palabras que usan
los títulos y las características, así que la parte de catálogo del vector la
recupera con precisión.

Al comparar tres altavoces (`B0BCTJMN63`, `B0B786PFYJ`, `B09WWG19YS`), los tres
tienen **mediana 5**, así que la mediana no los distingue:

| Producto | Reseñas | Media | Diferencia vs. JBL Charge 5 |
|---|---:|---:|---:|
| JBL Charge 5 | 7 | 4.57 | 0.00 |
| Altec Lansing Mini H2O | 11 | 4.91 | +0.34 |
| Bose SoundLink Micro | 27 | 4.44 | −0.13 |

El Bose tiene 22 reseñas de 5 estrellas, pero 3 de 1 estrella (11 %) bajan su
media. El JBL Charge 5 solo tiene 7 reseñas, así que una sola reseña nueva podría
mover su media más que la del Bose: la diferencia de 0.13 no permite afirmar que
uno sea mejor que el otro.

### 2. Consulta de experiencia: `"Comfortable headphones with good sound"`

| Similitud | Reseñas | Producto |
|---:|---:|---|
| 0.729 | 30 | Sony MDR7506 Professional Large Diaphragm Headphone |
| 0.717 | 30 | OneOdio Over Ear Headphone, Wired Bass Headsets |
| 0.712 | 7 | LOBKIN Bluetooth Headphone Over Ear |
| 0.711 | 9 | ARTIX CL650 Wired Headphones for Kids & Adults |
| 0.709 | 19 | CozyPhones Sleep Headphones |

**Interpretación.** Todos los resultados son auriculares, pero de usos muy
distintos: estudio, inalámbricos, para niños y para dormir. La consulta no
describe un tipo concreto de producto, sino sensaciones (*comfortable*, *good
sound*) que los clientes mencionan en las reseñas de casi cualquier auricular.
Las similitudes están muy juntas (0.709–0.729), así que el orden entre estos
cinco es poco significativo. CozyPhones aparece por la comodidad que mencionan
sus reseñas, no porque tenga mejor sonido.

## Limitaciones y alcance de los datos

- **La similitud no es una probabilidad ni mide la calidad.** Indica cercanía
  semántica. La búsqueda siempre devuelve los `top_k` productos más cercanos,
  aunque la consulta sea poco relevante para el catálogo.
- **Las reseñas aportan temas, no opiniones a favor.** Una reseña que dice *"the
  sound is terrible"* habla del mismo tema que *"good sound"*, así que el
  embedding puede acercar un producto criticado por su sonido a una consulta que
  pide buen sonido. Para valorar un producto hay que mirar `analyze_product`.
- **El modelo está orientado al inglés** y trunca los textos de más de 256
  tokens. Las consultas deben hacerse en inglés, y las reseñas o descripciones
  muy largas solo aportan su inicio.
- **Las estadísticas describen solo las reseñas seleccionadas** (entre 5 y 30
  por producto), no todas las valoraciones de Amazon. `review_count` no mide la
  popularidad del producto, y una media basada en 7 reseñas es mucho menos
  estable que una basada en 30.
- **La selección no representa todo el catálogo Electronics:** se eligieron
  600 productos de distintas familias para explorar diversidad.
