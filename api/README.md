# `api` — embed text, images, and audio

One class. Three methods. Each call returns **one L2-normalized 768-d `float32` vector**.

```python
from api import DIM, Embedder, cosine

embedder = Embedder(compute="ane")   # or Embedder(backend="mock")
vec = embedder.embed_text("a red fox")
# vec.shape == (768,)   vec.dtype == float32   abs(l2(vec) - 1) < 1e-5
embedder.close()
```

`cosine(a, b)` is a float in `[-1, 1]`. Because vectors are unit length, it is a dot product.

| Cosine | Meaning (real Core AI / ANE model, not mock) |
| --- | --- |
| `1.0` | Same input |
| `0.85–1.0` | Same thing, different wording or viewpoint |
| `0.60–0.85` | Related — UPS photo vs `a brown UPS delivery truck` 0.727, bark vs `a dog barking` 0.721 |
| ~`0` | Unrelated |

On a Mac run `python scripts/download_models.py`, then copy the `export` lines it prints (`ANEMLL_EMBEDDINGS_ARTIFACTS`, `ANEMLL_EMBEDDINGS_MODEL`, `ANEMLL_COREAI_PYTHON`). `Embedder(compute="ane")` reads those when the constructor arguments are omitted. Then `python scripts/warmup.py` once so Core AI specializes the towers for this chip and caches them — there is no per-hardware compile to ship. Validated on M4 Pro / macOS 27.0; on macOS 27.2 / M5 vision and text currently fall back to the GPU. Without Core AI, use `backend="mock"` (deterministic stand-in vectors; scores are **not** semantic).

`python -m pip install -e .` then `from api import Embedder` works anywhere. From a checkout, keep the repo root on `PYTHONPATH`.

---

## Text–text cosine

```python
from api import Embedder, cosine

embedder = Embedder(compute="ane")
q = embedder.embed_text("a red fox")                          # role="query"
d = embedder.embed_text("a red fox in the snow", role="document")
print(q.shape, cosine(q, d))
embedder.close()
```

On an M4 Pro this pair is about **0.901**; the same query vs `a delivery truck` is about **0.694** (~35 ms per sentence).

`role="query"` (or `"SearchQuery"`) vs `role="document"` (or `"Document"`) is the model's search prefix. Use query for the thing you type; document for items you index.

The Search page does this in [`demo/server.py`](../demo/server.py) (`POST /search`) and [`demo/static/search.js`](../demo/static/search.js).

---

## Image–text search

```python
from PIL import Image
from api import Embedder, cosine

embedder = Embedder(compute="ane")
photo = embedder.embed_image(Image.open("fox.jpg"))
captions = ["a red fox", "a delivery truck", "piano music"]
for text in captions:
    print(f"{cosine(photo, embedder.embed_text(text, role='document')):.3f}  {text}")
embedder.close()
```

On an M4 Pro a UPS photo scores about **0.727** vs `a brown UPS delivery truck` and **0.513** vs `a cat` (~380 ms per photo).

Same idea as Search (drop a photo, type a caption) and the Matrix page ([`POST /compare`](../demo/server.py), [`demo/static/heatmap.js`](../demo/static/heatmap.js)).

---

## Audio–text match

```python
import numpy as np
from api import Embedder, cosine

embedder = Embedder(compute="ane")
wav = np.asarray(..., dtype=np.float32)       # mono, 16 kHz
heard = embedder.embed_audio(wav, 16000)
print(cosine(heard, embedder.embed_text("a dog barking", role="document")))
embedder.close()
```

On an M4 Pro a bark clip scores about **0.721** vs `a dog barking` and **0.661** vs `a cat meowing` (~50 ms per sound).

The Heard page records a clip and searches those chunks: [`demo/static/heard.js`](../demo/static/heard.js). Clips need about 9 ms at 16 kHz (one mel frame).

---

## Camera-alert threshold

```python
from PIL import Image
from api import Embedder, cosine

embedder = Embedder(compute="ane")
frame = embedder.embed_image(Image.open("street.jpg"))
rule = embedder.embed_text("a UPS truck")
score = cosine(frame, rule)
print("fires" if score >= 0.65 else "quiet", score)
embedder.close()
```

That is a text rule. “Anything significant” on the Alert page is `1 - cosine(frame, empty_street)` — a change from the baseline photo, not a caption. `samples/camera_alert_rule.py` prints both: cosine vs the rule text, and 1 − cosine vs `--street` (or `$ANEMLL_DEMO_ALERT/frames/street.jpg`). Scoring lives in [`demo/alert_routes.py`](../demo/alert_routes.py) and [`demo/alert_score.py`](../demo/alert_score.py); the UI is [`demo/static/alert.js`](../demo/static/alert.js).

---

## Constructor

```python
Embedder(
    artifacts=None,          # dir that contains coreai/*.aimodel
    model=None,              # EmbeddingGemma 2 checkpoint (host tokenizer)
    coreai_python=None,      # python that can import coreai.runtime
    compute="ane",           # "ane" or "cpu"
    backend="coreai",        # "coreai" | "mock" | "reference"
)
```

`backend="reference"` is the full Sentence-Transformers checkpoint on CPU (parity target, not ANE).

Runnable scripts: `samples/embed_sentence.py`, `samples/image_text_search.py`, `samples/sound_matching.py`, `samples/camera_alert_rule.py`.
