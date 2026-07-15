# Multimodal RAG from Scratch: Phi-3-Vision · Jina-CLIP-v1 · ChromaDB

A Retrieval-Augmented Generation system built **from scratch** that reads the paper
*"Attention Is All You Need"* (Vaswani et al., 2017), indexes **both its text and its
figures**, retrieves the most relevant segments for a query, and generates a grounded answer
with a vision-language model.

The whole thing is one Google Colab notebook, split into clearly explained sections.

---

## Files

| File | Purpose |
|------|---------|
| `rag_phi3_jina_chroma.ipynb` | The Colab notebook (run this). |
| `rag_phi3_jina_chroma.py`    | The same code as a plain Python script (percent-format / Jupytext), if you prefer a `.py`. |
| `README.md`                  | This file. |

---

## How to run

1. Open `rag_phi3_jina_chroma.ipynb` in **Google Colab**.
2. Select a **GPU** runtime: `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all`.

The notebook downloads the paper, builds the index, and answers five sample queries at the
bottom. First run takes a few minutes (model downloads). Ask your own questions with:

```python
rag("What is layer normalization used for in the Transformer?")
```

**Allowed libraries used in the core logic:** `torch`, `chromadb`, `numpy`, `io`, `fitz`,
`requests`, `PIL`, `transformers`. `einops`, `timm`, and `accelerate` are installed only
because they are runtime dependencies of the Jina-CLIP / Phi-3 remote code — they are never
imported by our own code.

**Version note:** Phi-3-Vision loads with `trust_remote_code=True` and is sensitive to the
`transformers` version; `4.44.2` works well here (`4.40.2` is a good fallback). Models are
loaded with **eager attention**, so `flash-attn` is not required.

---

## How it works (section by section)

**0. Setup** — installs packages; `torch` already ships with Colab.

**1. Config** — every tunable (chunk size/overlap, image size filter, `top_k`, token budget)
lives in one `CFG` object.

**2. Document ingestion (`fitz` + `requests`)** — the PDF is streamed into memory (no disk
write). Text is extracted page-by-page and split into **overlapping word windows** (220 words,
40-word overlap) so no sentence is lost at a boundary and every chunk keeps its **page number**
for citation. Embedded figures are decoded to `PIL` images and filtered by a minimum
dimension to drop logos/artifacts.

**3. Embeddings (Jina-CLIP-v1)** — `encode_text` and `encode_image` map text and images into
**one shared 768-d space**. We run inference under `torch.no_grad()`, in batches, and
L2-normalize so cosine similarity is consistent. *This shared space is the core trick:* a
text query can retrieve a diagram.

**4. Vector DB (ChromaDB)** — a single cosine collection holds text and image vectors.
Text chunks store their text in `documents`; figures store only metadata (`type`, `page`)
while the actual `PIL` images stay in an in-memory `IMAGE_STORE` keyed by the same id (Chroma
holds vectors, not pixels). Switch to `PersistentClient` to persist.

**5. Retrieval** — the query is embedded with the *same* text encoder and Chroma returns the
nearest neighbours. Because text-vs-text similarity usually beats text-vs-image in CLIP space,
a pure `top_k` can come back all-text; so we **guarantee at least one figure** via a
metadata-filtered query. That makes figure-centric questions actually use the figure.

**6. Generation (Phi-3-Vision)** — retrieved text and figures are assembled into a prompt with
`<|image_i|>` placeholders (count must equal the number of images). The model is told to answer
**only** from the provided context and to cite pages — this grounding is what turns retrieval
into a faithful answer instead of a hallucination.

**7. End-to-end `rag()`** — ties retrieval + generation together and prints provenance
(which pages/figures fed the answer) so every result is auditable.

**8. Examples** — five sample queries spanning text, formula, and figure questions.

---

## Design decisions & trade-offs

- **Multimodal in one space** (Jina-CLIP) rather than two separate indexes — one query, one
  search, text *and* diagrams returned together.
- **Overlapping page-level chunks** — keeps page provenance and cross-boundary context with a
  dependency-free chunker (no LangChain/NLTK needed).
- **Images in RAM, vectors in Chroma** — keeps the DB small/fast and avoids base64 bloat in
  metadata.
- **Guaranteed-figure retrieval** — a small, explicit fix for a real CLIP retrieval quirk.
- **Eager attention + fp16** — fits Phi-3-Vision (~4B) and Jina-CLIP together on a single
  free-tier T4, with no flash-attn build step.
- **Deterministic decoding + "answer only from context"** — grounded, reproducible answers.

---

## Illustrative sample outputs

> These are representative of what the notebook produces on a GPU runtime (they were **not**
> generated in a CPU-only environment). Exact wording varies with model version.

**Q:** *What is the scaled dot-product attention formula, and why is the scaling factor used?*

> Scaled dot-product attention computes attention as `softmax(QKᵀ / √d_k) · V`, where `Q`,
> `K`, `V` are the query, key, and value matrices and `d_k` is the key dimension (page 4). The
> `1/√d_k` factor counteracts the growth in dot-product magnitude for large `d_k`: without it,
> the softmax is pushed into regions with vanishingly small gradients, which slows learning
> (page 4).

**Q:** *Explain multi-head attention and why it is beneficial.*

> Multi-head attention runs several attention functions in parallel on linearly projected
> copies of the queries, keys, and values, then concatenates and projects the results (page 4).
> The paper uses `h = 8` heads. Multiple heads let the model attend to information from
> different representation subspaces at different positions simultaneously — something a single
> head, which averages such information, cannot do (page 4–5).

**Q:** *Describe the overall model architecture shown in the figure.*
*(Figure 1 is retrieved and shown to Phi-3-Vision.)*

> The figure shows an encoder–decoder Transformer. The encoder (left) is a stack of identical
> layers, each with a multi-head self-attention sub-layer and a position-wise feed-forward
> sub-layer, both wrapped in residual connections and layer normalization. The decoder (right)
> adds a third, masked multi-head attention sub-layer over the encoder output, and its
> self-attention is masked so each position only attends to earlier positions. Inputs and
> outputs are embedded and combined with positional encodings before entering the stacks; the
> decoder ends in a linear + softmax layer producing output probabilities (page 3).

**Q:** *What BLEU scores did the Transformer achieve, and how did training cost compare?*

> The big Transformer reached 28.4 BLEU on English-to-German and 41.8 BLEU on
> English-to-French, setting new state-of-the-art results while training at a fraction of the
> cost of the previous best models (page 8).

---

## Extending it

- Persist the index with `chromadb.PersistentClient(path="./chroma")`.
- Add a cross-encoder re-ranker over the top-k for sharper ordering.
- For papers whose figures are vector graphics (not embedded rasters), render whole pages to
  images as a fallback.
- Swap in sentence-aware chunking for cleaner boundaries.
