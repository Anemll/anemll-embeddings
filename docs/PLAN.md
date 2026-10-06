# EmbeddingGemma 2 → Core ML / ANE conversion plan

Concrete phased plan for converting Google’s **EmbeddingGemma 2** to Core ML for the Apple Neural Engine, using ANEMLL-forge as a **read-only** source of ANE lessons—not as a drop-in converter.

**Checkpoint:** [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (prefer official; ~1.53 GB text safetensors when modality encoders are omitted).  
**Reference stack:** Sentence-Transformers `SentenceTransformer("google/embeddinggemma-2", …)`.  
**Forge reference (read-only):** `/Users/anemll/anemll-forge` — especially `docs/WORKFLOW.md`, `docs/TECHNIQUES.md`, `docs/ENVIRONMENT.md`, `docs/ANE_COMPILE_MODE_POLICY.md`, `ANE_DELTANET_NUMERICS.md`, `forge.py` `convert` → `scripts/qwen38_ane_model.py` / `qwen38_ane_chunk.py`, `tools/coreml/`.  
**Weights location:** `/Volumes/TB36/Models/anemll-embeddings` (TrueNAS SMB **TB36**). Checkpoint dir: `…/google-embeddinggemma-2`. HF cache: `…/hf-cache`. Do **not** download into this git repo, onto flash USB `/Volumes/SAN512` (slow; may hold a partial ~702M copy), or onto a full internal SSD. Do **not** download models as part of planning work.

---

## Model facts (pin these before coding)

| Item | Value |
| --- | --- |
| Total params (full multimodal) | ~740M |
| Text-only effective size | ~270M (130M transformer + 140M embedder) |
| Vision / audio towers | ~170M / ~300M (load later) |
| Layers | 24 |
| Model dim / FFN hidden | 512 / 2048 |
| Heads / KV heads | 4 heads; local KV=2, global KV=1 (GQA/MQA) |
| Local:global attention ratio | 5:1 |
| Sliding window | 1024 tokens (local layers) |
| Vocab | 262,144 |
| Context | 8,192 tokens |
| Activation | Gated FFN with **GELU** |
| Pooling | **Mean pooling** over token states (mask-aware) |
| Projection | **512 → 768** dense |
| Native output | 768-d; **MRL** truncate to 512 / 256 / 128 then **re-L2-normalize** |
| Task steering | Text instruction prefixes only (ST `prompt_name=…`) |
| Allowed dtypes (upstream) | **BF16 or FP32 only — never FP16** (FP16 → NaNs / silent trash) |
| Text-only load | `config_kwargs={"vision_config": None, "audio_config": None}` |

Sentence-Transformers graph (conceptual): Transformer → mean Pooling (`embedding_dimension=768`) → Normalize. Task prefixes must match the reference encode path or cosine parity will look “wrong” for the right reason.

---

## Phase 0 — Repo / environment (no weights yet)

1. Keep this checkout lean: code, configs, fixed prompt fixtures, docs. Weights and `.mlpackage` / `.mlmodelc` stay gitignored on the network volume.
2. Conversion venv (separate from forge’s Qwen/Core AI stack): Python 3.11, `torch`, `transformers`, `sentence-transformers`, `safetensors`, `coremltools` 9.x (public wheel first; note forge’s research env used a patched `9.1.dev1` with FP8 work—**do not assume that patch is required** for a text encoder).
3. Record `python -c '…'` / `forge.py doctor`-style version dumps when the forge checkout is available, but **do not** merge forge’s Core AI Python 3.13 / `coreai-*` pins into this project.
4. Network volume confirmed: `/Volumes/TB36/Models/anemll-embeddings`. Set `HF_HOME` / `HUGGINGFACE_HUB_CACHE` to `…/hf-cache` before any download ticket.

---

## Phase 1 — Text-only 270M → Core ML → ANE (primary)

### 1.1 Scope

- **In:** text backbone only; mean pool + 512→768 projection + optional L2 normalize; fixed prompt prefixes; MRL truncate+renorm on host or in-graph.
- **Out (this phase):** vision/audio encoders; interleaved media placeholders; forge-style KV cache, chunked decode, DFlash2, chat/serve, GPTQ/LUT quantization (defer until FP baseline is correct).
- **Success:** Core ML package runs with compute units including ANE; embedding cosine vs ST reference ≥ agreed threshold on a **fixed** prompt set; placement audit shows the hot path on ANE (not silent CPU fallback).

### 1.2 What to reuse from anemll-forge vs what must be custom

| Reuse (lessons / small tools) | Do **not** expect to reuse unmodified |
| --- | --- |
| ANE numerics: native activations can be wrong near 0; forge replaced SiLU with `0.5*x*(1+tanh(x/2))` because `x*sigmoid(x)` re-fuses to native SiLU (`ANE_DELTANET_NUMERICS.md`, `docs/TECHNIQUES.md`) — apply the **same skepticism to GELU** | `forge.py convert` / `quantize` / `chat` / `serve` — hard-wired to Qwen3.8-27B (`64` layers, `hidden_size=5120`), KV I/O, DeltaNet, LUT exports |
| Validation discipline: finite checks, **rel-L2 + norms**, not cosine alone; CPU vs ANE same graph; PyTorch same-weight parity; record OS/Xcode/coremltools hashes (`docs/WORKFLOW.md` validation section) | Chunked 4-layer LLM packages, host KV, recurrent DeltaNet state, drafter pairing |
| Compile / placement: `CPU_AND_NE` ≠ “on ANE”; capture compute plans; bonded compile mode policy is SoC-specific (`docs/ANE_COMPILE_MODE_POLICY.md`) — re-validate for Core ML (not Core AI) | Core AI Swift bridge, multifunction verify/prefill entries, V8 KV |
| Timing helper pattern in `tools/coreml/time_models.swift` (Apple sample, BSD-3) | MIL builders in `qwen38_ane_chunk.py` as a copy-paste encoder (wrong architecture) |
| Environment hygiene: separate conversion vs inference envs; pin and record (`docs/ENVIRONMENT.md`) | Forge requirements-conversion.txt Core AI pins |

**Bottom line:** embeddings need a **thin single-forward encoder path**. Forge is an LLM research stack. Steal the *engineering habits*; write a new convert/export/validate pipeline under this repo.

### 1.3 Export path (PyTorch / Sentence-Transformers → Core ML)

```
HF / ST text-only load (BF16 or FP32)
        │
        ▼
Wrap inference module:
  input_ids [1, S], attention_mask [1, S]
  → transformer last_hidden_state [1, S, 512]
  → masked mean pool → [1, 512]
  → Dense 512→768 → [1, 768]
  → (optional) L2 normalize
        │
        ▼
torch.jit.trace **or** script (prefer trace with fixed S; script if control flow needed)
  example inputs: fixed S ∈ {128, 512, 1024, 2048, …} ladder
        │
        ▼
coremltools.convert(
  …,
  inputs=[TensorType(name="input_ids", …), TensorType(name="attention_mask", …)],
  outputs=[TensorType(name="embedding", dtype=fp16|fp32)],
  minimum_deployment_target=…,  # start with a modern iOS/macOS that matches local SDK
  compute_units=CPU_AND_NE / ALL,
)
        │
        ▼
.mlpackage → smoke load → placement audit → cosine vs ST
```

**Concrete choices (phase 1 defaults):**

1. **Load:** `SentenceTransformer(..., config_kwargs={"vision_config": None, "audio_config": None}, model_kwargs={"torch_dtype": torch.bfloat16 or float32})`. Never `float16` in the PyTorch reference.
2. **Wrapper:** a small `nn.Module` that owns only the text tower + pool + projection (strip ST trainer bits). Keep task-prefix **tokenization on the host** so the Core ML graph stays `ids/mask → vector`.
3. **Trace vs script:** start with **`torch.jit.trace`** at one fixed `S` (e.g. 512). Add a small shape ladder (multifunction or separate packages) only after one shape is correct. Avoid dynamic shapes on first ANE bring-up.
4. **Pooling in-graph:** implement mask-aware mean (`sum(h * mask) / clamp(sum(mask), min=1)`) inside the module so ST and Core ML share one definition. Do not “forget” the mask (padding will poison the mean).
5. **Projection / MRL:** keep full 768 in the package; do MRL truncate + re-normalize on the host for 512/256/128 so one compiled graph serves all dims. Optionally add a second package later with in-graph truncate.
6. **Normalize:** match ST (`Normalize` module / `normalize_embeddings=True`). Document whether the `.mlpackage` emits unit vectors or raw projected vectors.
7. **coremltools path:** Torch frontend first. Escalate to hand-written MIL **only** if specific ops refuse ANE placement (forge’s pattern)—not as the default.
8. **Quantization:** FP baseline first. INT8 / LUT later, guided by forge’s “preserve ANE placement” notes (vector LUT width / axis matter). Do not GPTQ an encoder until cosine/L2 gates pass.

### 1.4 Likely ANE pitfalls

| Risk | Why it matters here | Mitigation |
| --- | --- | --- |
| **FP16 activations** | Upstream forbids FP16; ANE compute is often FP16 | Reference in BF16/FP32. Convert carefully; if Core ML lowers to FP16, gate on NaN rate + cosine **and** rel-L2. Prefer keeping sensitive ops in FP32 if the converter allows, or split embedding table / late projection. Abort any path that matches “silent NaN” behavior. |
| **GELU** | Gated FFN uses GELU; forge saw native SiLU ~1e-3 abs error near 0 | A/B native GELU vs `gelu ≈ x * Φ(x)` / tanh approximation; compare ANE vs CPU same package and vs PyTorch. Do not trust fluency of cosine alone. |
| **GQA / MQA + 5:1 local:global** | Unusual head layout vs dense MHA | Ensure RoPE / attention masks match Gemma 4 embedding variant; wrong KV repeat or head split → soft failure in cosine. |
| **Sliding window 1024** | Local layers are not full-context attention | Fixed-S traces must bake the correct windowed mask. Long prompts (up to 8K) need either chunked attention patterns that match the model or a full-context global layer schedule—copying “full causal 8K” will diverge. |
| **Mean pooling** | Sensitive to mask and dtype | Integer mask → float; protect empty sequences; compare pool-only tensors before projection when debugging. |
| **512→768 projection** | Extra matmul after pool | Include in graph for parity; check weight dtype/layout after convert. |
| **Long context (8K)** | Compile time, memory, ANE working set | Start S≤512 or 1024; grow ladder. Forge learned blocked softmax / tiling for large windows—revisit if full 8K loses placement. |
| **Embedding table 262K × 512** | Large gather | May dominate package size; watch ANE vs CPU placement of gather; consider host embedding + Core ML body only as a fallback experiment (document if used—changes I/O). |
| **Task prefixes** | Quality and parity | Host-side only; fixtures must use the same `prompt_name` / manual document title format as ST. |
| **Silent CPU fallback** | `CPU_AND_NE` can still run on CPU | Require compute-plan / timing evidence of ANE; forge explicitly warns not to promote CPU parity to ANE correctness. |

### 1.5 Validation (cosine vs Sentence-Transformers)

**Reference:** ST text-only, BF16 or FP32, fixed seed, fixed tokenizer revision, fixed `prompt_name`s.

**Fixed prompt fixture** (check into `tests/fixtures/prompts.json` or similar—small text only):

- Asymmetric: `SearchQuery` / `Document` pairs (short + medium).
- Symmetric: `SentenceSimilarity`, `Classification`, `Clustering` (one each).
- Edge: empty-ish after prefix, near-max-S padded sequence, code snippet with `CodeRetrieval`.
- MRL: same texts at 768 / 512 / 256 / 128 with post-truncate L2 renorm.

**Metrics (record all):**

1. Cosine(ST, CoreML) per prompt — primary gate (e.g. ≥ 0.999 for FP path on short prompts; tighten/loosen after first measurements).
2. Rel-L2 and embedding L2 norms (forge lesson: cosine hides magnitude bugs).
3. Finite / NaN / Inf counts (especially any FP16 path).
4. Pairwise similarity agreement: `cos(q,d)` ST vs CoreML on the same pairs (ranking-relevant).
5. CPU vs ANE on the **same** `.mlpackage`.
6. Latency p50/p95 and rough tokens/s equivalent at each S; note warm vs cold compile.

**Do not** claim ANE success from CPU-only Core ML results.

### 1.6 Suggested next engineering tickets

Small, numbered, independently mergeable:

1. **T1 — Fixture pack:** ✅ `tests/fixtures/prompts.json` + `embeddings.npy` + digests; `scripts/gen_reference_fixtures.py`.
2. **T2 — Text-only loader helper:** ✅ `src/load_text_model.py` — ST load with `vision_config`/`audio_config` None; BF16/FP32 only (refuse FP16).
3. **T3 — Inference wrapper module:** ✅ `src/embed_wrapper.py` — mask-aware mean pool + 512→768 + optional L2; `scripts/smoke_wrapper_vs_fixtures.py` cosine vs T1.
4. **T4 — Trace export:** Script `scripts/export_torchscript.py` — fixed-S trace, save `.pt` + metadata (S, dtype, git sha, model revision).
5. **T5 — coremltools convert:** `scripts/convert_coreml.py` — TorchScript → `.mlpackage`, compute units flag, I/O names documented in README.
6. **T6 — Parity harness:** `scripts/parity_cosine.py` — ST vs Core ML (CPU) cosine/rel-L2 on fixtures; JSON report.
7. **T7 — ANE smoke + placement:** Load with ANE-capable units; NaN check; placement/timing note; fail ticket if CPU fallback suspected.
8. **T8 — Shape ladder:** Repeat T4–T7 for S∈{128,512,1024} (separate packages or multifunction); document which S is default.
9. **T9 — MRL host path:** Truncate+renorm helper matching ST `truncate_dim`; parity at 128/256/512.
10. **T10 — FP16 hazard report:** Explicit experiment: what happens if convert forces FP16 compute; document NaN/cosine collapse; decide FP32 islands or other mitigation before any release claim.
11. **T11 — GELU A/B:** Native vs approximate GELU on ANE; pick recipe; note in PLAN results subsection.
12. **T12 — Network volume + HF cache docs:** README pins TB36 paths; keep caches there; no model bytes in git.

---

## Phase 2 — Multimodal (later)

Only after Phase 1 cosine/placement gates pass on text.

1. **Prefer separate packages:** vision encoder → tokens/features; audio encoder → tokens/features; shared text backbone package. Host assembles interleaved sequences (`<|image|>`, `<|video|>`, `<|audio|>` placeholders) unless a single fused graph proves necessary.
2. **Load sizes:** text+image ~440M (`audio_config=None`); text+audio ~570M (`vision_config=None`); full 740M.
3. **Token budgets:** images ~280 tokens default (configurable 70–1120); video frames ~140; audio ~25 tokens/s; shared 8K window—ANE shapes must reflect the chosen budget.
4. **Prefixes:** still text-only; media has no task prefix.
5. **Validation:** extend fixtures with tiny synthetic image/audio **paths on the network volume** (not in git); cross-modal cosine vs ST; keep text regression suite green.
6. **Reuse forge?** Still no—multimodal encoders are not Qwen DeltaNet chunks. Possibly revisit Core AI later if Core ML hits a hard ceiling; that is a separate decision.

---

## Open questions

- ~~Confirmed network volume~~ → `/Volumes/TB36/Models/anemll-embeddings` (+ `hf-cache`).
- Exact HF revision pin for reproducibility (commit hash, not only `main`).
- Whether first ship keeps **normalize inside** the `.mlpackage` or on host.
- Whether embedding **gather stays on ANE** or moves host-side for size/placement.
- Minimum macOS / deployment target for the packages we publish.
- Acceptable cosine / rel-L2 thresholds after first real measurements (numbers above are starting gates, not gospel).

---

## Explicit non-goals (near term)

- Running or modifying `forge.py convert` against EmbeddingGemma.
- Downloading the checkpoint in planning tickets.
- Shipping INT8/LUT or Core AI packages before FP Core ML parity.
- Chat, speculative decode, or OpenAI-compatible generate APIs—this project embeds, it does not serve an LM.

---

## Document history

- Stub plan: initial scaffold.
- Expanded plan (this revision): forge read-through (`WORKFLOW`, `TECHNIQUES`, `ENVIRONMENT`, `forge.py` convert → `qwen38_ane_model.build_v3`, chunk MIL path, numerics docs) + EmbeddingGemma 2 model-card architecture pinned; phased tickets T1–T12.
