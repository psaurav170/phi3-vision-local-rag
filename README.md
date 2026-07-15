# Multi-Modal RAG over "Attention Is All You Need"

A Retrieval-Augmented Generation (RAG) system built from scratch that answers
questions about the paper *Attention Is All You Need*. It ingests the source
PDF, embeds both text and figures into a single shared vector space with
Jina-CLIP-v1, stores them in ChromaDB, retrieves the most relevant segments for
a query, and generates grounded answers with the Phi-3-Vision model. Because
the text and image embeddings share one space, a text query can retrieve a
diagram, and retrieved figures are passed to the vision model alongside the
text.

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the notebook](#running-the-notebook)
- [Configuration](#configuration)
- [Pipeline details](#pipeline-details)
- [Example queries](#example-queries)
- [Design choices](#design-choices)
- [Troubleshooting](#troubleshooting)

## How it works

The system is a linear pipeline of five stages, each in its own notebook
section:

1. **Document ingestion**: download the PDF, extract text as overlapping
   word-based chunks, and extract embedded figures.
2. **Embedding generation**: encode text chunks and figures with Jina-CLIP-v1
   into a shared 768-dimensional space, L2-normalised for cosine similarity.
3. **Vector storage**: index all embeddings in a ChromaDB collection with
   cosine distance; figure pixels are held separately in memory.
4. **Retrieval**: embed the query, fetch the nearest neighbours, and guarantee
   at least one figure is available to the generator.
5. **Generation**: assemble retrieved text and figures into a prompt and
   generate a grounded answer with Phi-3-Vision.

## Requirements

- **Runtime:** a CUDA GPU is strongly recommended. Phi-3-Vision is a ~4B
  parameter model; on CPU it is very slow and may run out of memory. Google
  Colab with a T4 GPU runtime is sufficient.
- **Python:** 3.10 or later.
- **Models** (downloaded automatically on first run):
  - `jinaai/jina-clip-v1`
  - `microsoft/Phi-3-vision-128k-instruct`

### Libraries

The implementation uses only the permitted libraries:

- `torch`
- `chromadb`
- `numpy`
- `io`
- `fitz` (PyMuPDF)
- `requests`
- `PIL`
- `transformers`

## Installation

Install the dependencies:

```
pip install -q "transformers==4.44.2" accelerate einops timm
pip install -q chromadb pymupdf pillow requests
```

Flash-attention is intentionally not installed; both models use the eager
attention implementation.

### PyMuPDF / `fitz` note

The import name `fitz` is provided by **PyMuPDF**. An unrelated abandoned stub
package named `fitz` on PyPI squats the same import name and will cause
`AttributeError: module 'fitz' has no attribute 'open'`. If you hit this, remove
the stub and install the real package:

```
pip uninstall -y fitz frontend PyMuPDF PyMuPDFb
pip install PyMuPDF
```

Then restart the runtime before re-running, and verify:

```python
import fitz
print(fitz.__doc__)   # should show the PyMuPDF banner, not None
```

## Running the notebook

1. Open the notebook in Jupyter or Google Colab.
2. On Colab, set the runtime to GPU: **Runtime > Change runtime type > T4 GPU**.
3. Run the setup and `fitz` cells at the top.
4. Run the remaining sections in order (**Runtime > Run all** works once setup
   is done).

The first run downloads both models, which takes a few minutes. Phi-3-Vision is
saved to a local directory (`./Phi-3-vision-128k-instruct`) and reused on later
runs.

## Configuration

All tunable parameters live in the `CFG` class in Section 1:

| Parameter             | Default | Purpose                                                   |
|-----------------------|---------|-----------------------------------------------------------|
| `PDF_URL`             | paper URL | Source document to ingest                               |
| `CHUNK_SIZE`          | 220     | Words per text chunk                                       |
| `CHUNK_OVERLAP`       | 40      | Words shared between consecutive chunks                    |
| `MIN_IMG_DIM`         | 100     | Minimum width and height (px) for a figure to be kept      |
| `JINA_ID`             | jina-clip-v1 | Embedding model                                      |
| `PHI_ID`              | Phi-3-vision-128k-instruct | Generation model                        |
| `TOP_K`               | 5       | Nearest neighbours fetched per query                      |
| `MIN_IMAGES`          | 1       | Figures always surfaced to the generator                  |
| `MAX_IMAGES_TO_MODEL` | 2       | Cap on figures passed to Phi-3-Vision                     |
| `MAX_NEW_TOKENS`      | 512     | Maximum tokens generated per answer                       |

## Pipeline details

### Document ingestion (Section 2)

The PDF is downloaded into memory with a `User-Agent` header to avoid occasional
403 responses. Text is extracted per page and split into overlapping word
windows: each chunk is `CHUNK_SIZE` words with `CHUNK_OVERLAP` words shared with
the previous chunk, which preserves context across chunk boundaries. Fragments
shorter than 40 characters are skipped.

Figures are extracted from every page. Each embedded image is de-duplicated by
its PDF cross-reference (the same figure can appear on multiple pages), decoded
to RGB, and dropped if either dimension is below `MIN_IMG_DIM` (this filters out
logos and small artifacts).

### Embedding generation (Section 3)

Jina-CLIP-v1 maps text and images into one shared 768-dimensional space, which
is what allows a text query to match a figure. Text chunks are encoded with
`encode_text` and figures with `encode_image`, then every vector is
L2-normalised so that cosine similarity is a clean dot product. A sanity check
confirms that a probe query about attention retrieves a relevant text chunk.

### Vector storage (Section 4)

All embeddings are stored in a ChromaDB collection named `attention_paper`
configured for cosine distance (`hnsw:space: cosine`). Text and figures are
stored with a `type` field (`text` or `image`) and a `page` number in their
metadata. ChromaDB holds only vectors and metadata; the figure pixels are kept
in an in-memory `IMAGE_STORE` dictionary keyed by id, because the generation
step needs the original images.

### Retrieval (Section 5)

The query is embedded and used to fetch the `TOP_K` nearest neighbours. To keep
the system genuinely multimodal, if fewer than `MIN_IMAGES` figures appear in
those results, the retriever runs a second image-only query and appends the
missing figures. This guarantees the generator always has at least one figure
available.

### Generation (Section 6)

Retrieved text excerpts (with page numbers) and up to `MAX_IMAGES_TO_MODEL`
figures are assembled into a single prompt. The prompt instructs the model to
answer using only the provided context, to say so when the context is
insufficient, and to cite page numbers. Figures are referenced with Phi-3-Vision
image placeholders and passed to the processor. Generation is deterministic
(greedy, `do_sample=False`). Phi-3-Vision loads in float16 on GPU and float32 on
CPU, with eager attention.

### End-to-end pipeline (Section 7)

The `rag(query)` function ties the stages together: it retrieves context, prints
what was retrieved (ids, pages, types, distances), shows which figures were fed
to the model, and prints the generated answer.

## Example queries

Section 8 runs four representative queries:

1. **What the Transformer is** and how it differs from recurrent and
   convolutional sequence models.
2. **Multi-head attention** and why it is beneficial compared to single-head
   attention.
3. **Translation results (BLEU)** and training cost.
4. **The model architecture diagram**: a figure-centric query that exercises
   the multimodal path by feeding a retrieved figure to the vision model.

Run your own query with:

```python
rag("Your question about the paper here")
```

## Design choices

- **Shared embedding space.** Jina-CLIP-v1 places text and images in one space,
  so retrieval is genuinely cross-modal rather than two separate indexes.
- **Overlapping chunks.** Word-level overlap avoids cutting sentences and
  concepts across chunk boundaries, improving retrieval quality.
- **Guaranteed figure retrieval.** The `MIN_IMAGES` mechanism ensures the vision
  model is actually used, rather than defaulting to text-only answers.
- **Deterministic generation.** Greedy decoding makes outputs reproducible for
  evaluation.
- **Grounded prompting.** The model is constrained to the retrieved context and
  asked to cite pages, which reduces hallucination.
- **No flash-attention dependency.** Eager attention keeps setup simple and
  portable across environments.
