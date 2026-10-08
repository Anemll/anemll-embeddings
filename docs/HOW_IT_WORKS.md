# How the Neural Engine packages are built

This is the longer version of the README “How it works” section. The goal is still the same: one embedding per photo, sound, or sentence, produced on the Apple Neural Engine (ANE) with no mid-graph GPU or CPU.

## Three towers, one host

EmbeddingGemma 2 is not one graph. It is three towers plus a small host program:

| Tower | About | Package | What it emits |
| --- | --- | --- | --- |
| Vision | ~170M | `vision_s280.aimodel` | 280 soft tokens, each 512-wide |
| Audio | ~300M | `audio_s280.aimodel` | 70 soft tokens (280 mel frames, 4× subsample) |
| Text (ids) | ~270M | `text_s128.aimodel` | one 768-d vector from token ids |
| Text (embeds) | same text tower | `text_embeds_s320.aimodel` | one 768-d vector from already-looked-up tokens |

A *soft token* is a 512-wide vector that stands in for an image patch group or a slice of audio. The host looks up word tokens itself, drops the soft tokens into the sequence wherever `<|image|>` / `<|audio|>` appear, and only then calls `text_embeds_s320`. Plain text with no media can use the smaller `text_s128` package.

Tokenization, task prefixes (`SearchQuery`, `Document`, …), and the embedding-table lookup stay on the host. That keeps the compiled graphs fixed-shape and ANE-friendly.

`scripts/export_coreai_towers.py` does `torch.export` in this repo’s venv, then converts `.pt2` → `.aimodel` under `ANEMLL_COREAI_PYTHON`. It does **not** call `forge.py convert`. The FLOAT32 Core ML text tree is left alone.

## Why fp16 — and why the original model forbids it

The Neural Engine wants 16-bit (fp16) compute. Upstream EmbeddingGemma 2 tells you never to run the *PyTorch* model in fp16: it NaNs or goes silently wrong. This repo still refuses fp16 in `src/load_text_model.py`.

The shipped Core AI graphs are a different story. Vision, audio, and `text_embeds` are converted with `cast16` (an fp16 graph). Two rewrites make that safe:

1. **Robust norms.** Activations can reach ~900. Squaring that overflows fp16. RMSNorm and LayerNorm run on `x / max|x|` first (the forge `rms_robust` idea). Without this, cast16 zeroed the vision tower.
2. **Safe mask fill.** The original audio invalid-logit is `-1e9`, which is `-inf` in fp16 and then NaN after softmax. The graph uses `-1e4` instead. Softmax is soft-capped to ±50, so `-1e4` already zeroes a masked key.

The host still passes a 0/1 keep-mask. Do not force that mask to all ones: fp16 noise on pad rows can leak into the real frames.

`text_embeds` keeps a float32 output edge (an edge cast, not a flipped graph). Text *ids* stay int32 — the 262,144-word vocabulary does not fit in int16.

## Why attention and masks were rewritten

A first export that “looks like Hugging Face” compiles, then the ANE refuses whole families of ops and MPSGraph runs them on the GPU. The leftover “unnamed GPU island” from earlier notes was that: the three towers were FP32, the ANE refuses every f32 op, and almost the whole tower ran on the GPU.

What had to change (see `src/vision_export_patches.py`, `src/audio_export_patches.py`, `src/trace_patches.py`):

- **No fused SDPA** when Q, K, and V shapes differ. Vision attention is Q/K/V as `[S, D]`, then matmul + softmax. Softmax is tiled as 6 head groups × 2 query blocks (18 query blocks matched poorly on the ANE).
- **No 1-bit / bool masks.** The ANE cannot reshape `i1`. Masks are float additive biases, 4-D.
- **No `aten.unfold`.** Audio chunked windows use `index_select` / a baked one-hot instead.
- **No int64 gathers.** Position tables and one-hots are float matmuls. Vision one-hot uses `relu` (a `clamp` feeding that matmul was wrong on the ANE).
- **No `torch.cat` / `split` RoPE.** Vision rotary position is one constant rotate-half matmul.
- **Layout swaps are constants.** Computed permutations in-graph were refused; they are baked.
- **Audio conv-stem layout** is permutes, not a mid-graph gather. Rel-pos sinusoids and keys are precomputed per layer.

`scripts/dump_mpsgraph.py` reads the compiled `mps` graph, including each op’s device and `ane_validation_message`. That is how “fully on the ANE” is checked: `mps.fullyPlacedOnANE`, `mps.noGPUActivity`, one ANE region, no GPU or CPU regions.

## What “fully on the ANE” means here

On an M4 Pro, macOS 27.0, the shipped packages report:

| Package | Placement | Cosine vs FP32 CPU | p50 |
| --- | --- | --- | --- |
| `vision_s280` | fully ANE, 1 region | 0.999954 (row min 0.99979) | 337 ms |
| `audio_s280` | fully ANE, 1 region | 0.999927 (25 valid rows, row min 0.99944) | 10.8 ms |
| `text_embeds_s320` | fully ANE, 1 region | 0.999963 | 34.8 ms |

Inputs: aurora image, `tone_a4` audio, caption + image soft tokens (269 tokens). Reference is the patched PyTorch eager model in FP32 on CPU.

End-to-end host cosine against the Sentence-Transformers fixtures is a separate number (the gap predates the fp16 ANE work and sits in the host pipeline):

| Case | before (FP32, GPU) | fp16, CPU | fp16, ANE |
| --- | --- | --- | --- |
| image | 0.9859 | 0.9861 | 0.9860 |
| caption | 0.9466 | 0.9465 | 0.9462 |
| audio | 0.8708 | 0.8700 | 0.8669 |
| mix | 0.8987 | 0.8970 | 0.8983 |
| text (`text_s128`) | 0.9951 | 0.9951 | 0.99996 |

Vision attention is most of each ~21 ms layer (~15 ms) and is at its floor for exact fp16 with the 12 tilings that still match. A 140-soft-token budget ran 108 ms fully on the ANE (cosine 0.99994 to its own FP32 reference) but changes the embedding (cosine 0.958 to s280), so it is not shipped.

## Older Core ML text path

`scripts/export_torchscript.py` + `scripts/convert_coreml.py` still build a fixed-S text `.mlpackage`. That path is not the multimodal product. On the M4 Pro, FP16 Core ML *does* place on the ANE (2307 ANE / 17 CPU preferred ops). Those 17 leftovers are a **begin-of-graph** island (pad/window mask + embedding gather), not a mid-graph CPU island. Mid-graph CPU would be a fail. The FP16 *CPU* path was the weak one (min cosine 0.920 vs the T1 fixtures). Timings for S=512 (warmup 2, iters 5): CPU_ONLY p50 65.0 ms / CPU_AND_NE 27.4 ms / CPU_AND_GPU 38.6 ms / ALL 28.4 ms.

## Limitations that come from the hardware

- M4 Pro / macOS 27.0 is the validated setup.
- On macOS 27.2 (M5, newer ANE) the Core AI ANE pre-check currently rejects the vision and text packages (`invalid MLIR-MPS program`) and they fall back to the GPU. Audio still runs on the ANE.
- Audio shorter than one mel frame (about 9 ms at 16 kHz) cannot produce a soft token.
