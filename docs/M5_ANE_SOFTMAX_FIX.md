# M5 / macOS 27.2: unfused attention softmax puts text and vision on the ANE

Status: verified on Apple M5 (macOS 27.2), M4 Pro and M3 Ultra (macOS 27.0).
All three towers are fully on the ANE on all three Macs. Text latency is
unchanged on 27.0; vision is 22-34% slower there. See
[Verification on macOS 27.0](#verification-on-macos-270). Option 1 below was
chosen: the M5 export of both towers is published for every Mac at Hugging Face
revision `18e1b7e85cdf0c58d924c5d270c7a4be1a40159a` (`ANE_REVISION`).

## Summary

- On macOS 27.2 the published `text_embeds_s320` and `vision_s280` towers fall
  back to the GPU; `audio_s280` stays on the ANE.
- Cause: a `softmax` between the two attention matmuls
  (`q @ K^T -> softmax -> @ V`). MPSGraph fuses that chain into an
  `mps_spi.sdpa` op, and the 27.2 ANE check rejects the graph, so Core AI
  places the whole tower on the GPU. The check runs when Core AI first loads
  (specializes) the tower on that Mac, not at export, and prints no error:
  the only console output is `Failed to import MPS module`.
- Fix: spell the softmax out as `amax`/`sub`/`exp`/`sum`/`reciprocal`/`mul`
  (`softmax_unfused` in `model/trace_patches.py`), used by the text and vision
  attention.
- Result on M5 / 27.2: re-exported `text_embeds_s320` and `vision_s280` are
  fullyOnANE (one ANE region, no GPU regions). With the published
  `audio_s280`, `scripts/warmup.py --require-ane` exits 0.
- The ANE is slower than the GPU on the M5 for both towers (see Results).

## Root cause

Placement does not depend on shape, mask or softmax dtype. Evidence, all on the
M5, each package cold-compiled with a fresh `CFFIXED_USER_HOME`:

1. Layer bisect on the real model, truncated and exported with the repo's own
   patches:
   - 0 layers (PLE, masks, pool, projection, L2): fullyOnANE.
   - 1 layer: GPU.
   - One layer, each sub-block alone: MLP, PLE gate and input norm are
     fullyOnANE. Attention is GPU.
2. Attention stages (one layer, attention only):
   - `q @ K^T` then `@ V` with no softmax: fullyOnANE.
   - Adding the softmax: GPU.
3. Mini graphs:
   - A standalone softmax, `q @ K^T -> softmax` alone, and `softmax -> @ V`
     alone are each fullyOnANE.
   - `q @ K^T -> softmax -> @ V` is GPU for every size tried, from
     `[2,64,32,64]` up to the real `[1,2,640,320]x256`, with or without mask,
     with an f32 or f16 softmax.
   - `exp -> sum -> divide` between the matmuls: fullyOnANE.
   - `exp -> sum -> reciprocal -> mul`: fullyOnANE.
   - Row-max subtract then `divide`: GPU again (recognised as a softmax).
   - Row-max subtract then `reciprocal * mul`: fullyOnANE.
4. Op diff against the passing `audio_s280` (segmented AICode dumps via
   `MPSGRAPH_DUMP_SEGMENTED_AICODE_MODULE_TO_FILE`): the text-only ops
   (`gelu_v1` tanh, `split_v1`, `bitwise_and`/`not`, `sqrt`) were each exported
   alone, together with the sliding mask, full mask, masked mean and L2 head,
   and all are fullyOnANE. Audio's softmax is a tiny `[8,6,12,24]` and its
   attention has no plain `matmul -> softmax -> matmul` chain.
5. Vision uses the same pattern (`matmul -> torch.softmax -> matmul`).

Caveats:

- The `sdpa` fusion is inferred from the behaviour above. The fallback happens
  before any dump, so there is no dump of the rejected `sdpa` op, and nothing
  here proves the fused `sdpa` is the cause - only that removing the softmax
  pattern fixes placement.
- No rejection message was printed in these runs (earlier notes quoted
  `invalid MLIR-MPS program`; that string never appeared). The only console
  output for rejected towers was `Failed to import MPS module`, at first load.
- The published text package compiles to 4 nested GPU regions
  (`GPU_region_0_nested_0..3`); the new one to a single ANE region.

The one-off bisect scripts are not part of the repo.

## The fix

`model/trace_patches.py`, `softmax_unfused(scores)`:

```python
e = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
return e * torch.reciprocal(e.sum(dim=-1, keepdim=True))
```

- The row-max subtract is required for accuracy. A version without it was
  fullyOnANE but gave cosine only 0.88-0.92: valid attention scores reach
  12.8-20.7 across layers, and fp16 `exp` overflows above about 11.
- `reciprocal * mul`, not `divide`: the divide form is re-fused into a softmax
  (evidence 3).
- Text: the eager attention in `model/trace_patches.py` calls it.
  Vision: `_vision_attn_forward` in `model/vision_export_patches.py` calls it.
- No change to package I/O, dtypes or shapes; the signatures in
  `hf/towers.yaml` are unchanged.
- `tests/test_softmax_unfused.py` checks shape, numerics against
  `torch.softmax` (fp32 and fp16, including scores around 20) and that the
  traced graph has no `softmax` or `div` op.
- `model/parity_text_embeds_ab.py` is a text-only A/B parity tool for
  `text_embeds_s320` packages.

## Results on M5 (macOS 27.2)

| | Published `text_embeds_s320` | New `text_embeds_s320` |
|---|---|---|
| Placement (warmup, cache manifest, mpsgraph dump) | GPU (4 nested GPU regions) | fullyOnANE (1 ANE region, 0 GPU regions) |
| Warm p50 / p90, ANE-preferred (`time_coreai_text_embeds.py`, warmup 10, 100 iters) | 24.82 / 25.16 ms (runs on GPU) | 33.07 / 33.48 ms (ANE) |
| Same tool, GPU-preferred | 25.4 ms | 26.0 ms |
| First compile / cold load, fresh home | about 5 s | 23.5-24.4 s |

- The 23.5-24.4 s figure is the first ANE compile of this graph. Later loads
  with a fresh Core AI home took about 3 s (the ANE compiler appears to cache
  outside the Core AI home; not confirmed).
- Text is about 8 ms slower on the ANE than on the GPU on the M5, in line with
  the 34.8 ms documented for M4 Pro (27.0).
- Vision (`vision_s280`, dummy input, ANE-preferred, 40 iterations): published
  (GPU) p50 103 ms, new (ANE) 429 ms. Cold compile about 55 s. The M4 Pro
  figure in the repo is 337 ms.

Cosine, 12 fixture prompts (`tests/fixtures/prompts.json`) through the host
path (`api.embedder.CoreAIBackend`), tool `model/parity_text_embeds_ab.py`:

| Comparison | min | mean |
|---|---|---|
| new vs FP32 CPU reference (full checkpoint) | 0.999934 | 0.999972 |
| new vs published package (GPU) | 0.999933 | 0.999972 |
| new vs ST fixture `embeddings.npy` (bf16 on MPS) | 0.999882 | 0.999938 |
| published vs FP32 CPU | 0.999998 | 0.999999 |
| published vs ST fixture | 0.999934 | 0.999963 |
| ST fixture vs FP32 CPU | 0.999935 | 0.999964 |

- The 0.9999 target is met against FP32 and against the published package.
- Against the bf16 fixture the minimum is 0.999882 (prompt `cls_one`); the
  fixture itself is only 0.999935 from FP32 on that prompt, so most of the gap
  is fixture bf16 noise.
- The published package is near-perfect against FP32 because it ran on the
  GPU on the M5. The new package runs in fp16 on the ANE, with the error level
  the repo reports for the ANE on M4 Pro (0.999963).
- Vision: on 3 synthetic images, new (ANE) vs published (GPU) cosine is
  0.999960, 0.999982 and 0.999978. The multimodal fixtures were not run.

`main.mlirb` sha256 of the M5 test packages: text
`8181d927facbcee1817a1bcb5efbd9a502b5468d5c131c0839aa624b7e1b8e1a`, vision
`5ebb5342a5c78ab2790c82358e3ad90092ea3bbdde6a9dc85a47d4ecc8d37c0c`. Exports
are not reproducible by hash (a re-export of the unchanged published code gives
a different `main.hash`), so only behaviour is comparable.

## Verification on macOS 27.0

Done on M4 Pro (macOS 27.0, build 26A5425a) and M3 Ultra (macOS 27.0, build
26A428), each from a fresh checkout of this branch with a fresh
`CFFIXED_USER_HOME`. Tested: the M5-exported packages above (`main.mlirb`
sha256 `8181d927...` text, `5ebb5342...` vision) on both Macs, and a native
re-export on the M4 Pro (sha256 `22c08bd2...` text, `d9fd3432...` vision;
same embeddings as the M5 export to cosine 1.000000, same timings within 4%).
Audio is the published `audio_s280` on every run.

Placement: `scripts/warmup.py --require-ane` exits 0 for both the published
and the new packages on both Macs; no `GPU_region` in the Core AI cache.

Warm p50 on the ANE (dummy inputs, 100 iterations; vision 30):

| Mac | Package set | text | vision | audio |
|---|---|---|---|---|
| M4 Pro | published | 34.2 ms | 339 ms | 11.2 ms |
| M4 Pro | new (M5 export) | 34.8 ms | 429 ms | 11.2 ms |
| M4 Pro | new (native export) | 33.5 ms | 414 ms | 11.2 ms |
| M3 Ultra | published | 35.2 ms | 349 ms | 11.7 ms |
| M3 Ultra | new (M5 export) | 35.2 ms | 465 ms | 11.6 ms |
| M5 (27.2) | published (text, vision on GPU) | 24.0-27.2 ms | 99 ms | 13.4 ms |
| M5 (27.2) | new | 32.6-33.0 ms | 426 ms | 13.8 ms |

On real photos the vision forward is 414 ms (M4 Pro), 466 ms (M3 Ultra) and
422 ms (M5) with the new tower, against 339 / 349 / 99 ms with the published
one.

Cosine of the final embedding (host path) vs the same host path with the
patched PyTorch towers in FP32 on CPU, minimum over 12 text fixtures, 8 real
photos (macOS user pictures) and 4 real sounds (speech and system sounds):

| Mac | Package set | text | photos | sounds |
|---|---|---|---|---|
| M4 Pro | published | 0.999943 | 0.999903 | 0.998908 |
| M4 Pro | new | 0.999934 | 0.999918 | 0.998951 |
| M3 Ultra | published | 0.999943 | 0.999918 | 0.998909 |
| M3 Ultra | new | 0.999934 | 0.999891 | 0.998951 |
| M5 | new | 0.999934 | 0.999918 | 0.998951 |

New vs published on 27.0: text 0.999990, photos 0.999955 or better, sounds
0.999915 or better. The sound minimum is the same with both sets (the audio
package did not change); the gap to FP32 is in the end-to-end audio path, not
in this fix. On the M3 Ultra the photo minimum is 0.999891, just under 0.9999
(published: 0.999918).

Result: text is a free win (same speed and accuracy on 27.0, fully on the ANE
on 27.2). Vision is 22-34% slower on 27.0 with the unfused softmax, and on the
M5 the ANE is about 4x slower than the GPU for vision. Options:

1. Publish both new towers (fully on the ANE everywhere, slower vision).
2. Publish the new text tower only and keep the published vision tower (fast
   on 27.0; on 27.2 vision stays on the GPU, which is faster there anyway).
3. Per-OS vision packages (fused softmax for 27.0, unfused for 27.2+), picked
   by `download_models.py` from the macOS version.
4. Try to recover vision speed with the unfused softmax (for example without
   the row max if vision scores stay in fp16 range, or different tiling).

## Background: why macOS 27.0 was expected to be safe, and how it was checked

Why it should be safe:

- No new op types. The segmented AICode op histogram differs only in counts:
  `softmax_v1` goes from 24 to 0, and `reduce_max`, `sub`, `reduce_sum`, `exp`
  and `mul` already appear in the published graph. `reciprocal` lowers to the
  existing `divide`.
- Core AI's own `softmax_v1` body is `exp -> reduce_sum -> divide`, which the
  new graph spells out (plus `amax`/`sub`).
- Package I/O and signatures are unchanged.

Real risk: on 27.0 the old graph probably ran attention through the fused ANE
`sdpa`; the new graph does not. So even with the same op set:

- Latency could move either way (published: 34.8 ms text, 337 ms vision on
  M4 Pro).
- fp16 numerics could differ (the unfused softmax accumulates in fp16 on the
  ANE).

Steps used for the verification above (M4 Pro and M3 Ultra, macOS 27.0):

1. Check out this branch. Re-export with the commands below into a fresh
   artifacts dir, and use a fresh `CFFIXED_USER_HOME`.
2. `python scripts/warmup.py --require-ane`: expect exit 0 with text and
   vision fullyOnANE.
3. Confirm placement in the cache manifest or with `model/dump_mpsgraph.py`:
   no `GPU_region`.
4. `python model/parity_text_embeds_ab.py --artifacts new=<dir>
   --artifacts published=<dir> --host ~/.anemll-embeddings/ane/host
   --model-full <full ckpt>`: require min cosine >= 0.9999 against FP32 and
   compare with the published 0.999963.
5. `python model/time_coreai_text_embeds.py --artifacts <dir> --warmup 10
   --iters 100`: compare p50 with 34.8 ms (text); vision with 337 ms.
6. Vision: also compare against the published package on real images.

## Reproduction

From the repo root, with the Core AI export venv in `ANEMLL_COREAI_PYTHON`
and the host venv as `python`:

```sh
export ANEMLL_COREAI_PYTHON=<path to the Core AI export venv python>
python scripts/download_export_assets.py --dest <ckpt dir>      # ~1.5 GB
export ANEMLL_EMBEDDINGS_MODEL=<ckpt dir>/embeddinggemma-2-full
python model/export_coreai_towers.py --tower vision --tower text_embeds --artifacts <art dir>
ln -s ~/.anemll-embeddings/ane/audio_s280/audio_s280.aimodel <art dir>/coreai/
ANEMLL_EMBEDDINGS_ARTIFACTS=<art dir> python scripts/warmup.py --coreai-home <fresh home> --require-ane
```

Export takes about 30 s for text. The first ANE compile takes about 25 s for
text and 56 s for vision on the M5.

## Open questions

- Vision is 22-34% slower on 27.0 with the unfused softmax (option 1 was
  shipped). Recovering that speed (option 4) is open.
- The ANE is slower than the GPU on the M5 (text 33 vs 25 ms, vision 429 vs
  103 ms). Decide whether 27.2 should stay ANE-pinned or follow the hardware.
- Tuning of the unfused attention (tiling, fusing steps) was not attempted.
- Why the 27.2 pre-check rejects `mps_spi.sdpa`, and whether a later macOS
  reverts it, is unknown.
- `audio_s280` is unchanged; it was already fullyOnANE.
