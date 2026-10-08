# Local demo

This demo shows EmbeddingGemma 2 running on your Mac’s Neural Engine. It turns photos, sounds, and text into embeddings — lists of numbers that capture meaning — so things that mean the same thing land close together, even across types. A photo of a fox, a bark, and the words “a red fox” can match each other. Everything stays on your Mac.

The pages are plain HTML. There is no front-end build step.

## Start the server

You need the public Core AI packages (`vision_s280`, `audio_s280`, `text_embeds_s320`) plus the Google host checkpoint. `python scripts/download_models.py` fetches them (about 2.8 GB; sizes, flags, and example output: [scripts/README.md](../scripts/README.md)) and creates the `artifacts/coreai/<name>.aimodel` symlinks `api.Embedder` expects. No Hugging Face login. Put the printed `export` lines in `~/.zshrc` or a file you `source`.

```sh
python scripts/download_models.py
# copy the export lines it prints, then:
python scripts/warmup.py
export ANEMLL_DEMO_DATA=$HOME/.anemll-embeddings/demo
export ANEMLL_DEMO_CORPUS=$HOME/.anemll-embeddings/corpus
export ANEMLL_DEMO_ALERT=$HOME/.anemll-embeddings/alert

python -m pip install -r demo/requirements.txt
python -m demo.server --backend coreai --host 0.0.0.0 --port 8766
```

First `warmup.py` load compiles each tower for this Mac and caches it. There is no separate per-hardware compile to ship. Validated on M4 Pro / macOS 27.0; on macOS 27.2 / M5 vision and text currently fall back to the GPU.

`python -m demo` and `python -m demo.server` are the same command. Then open **http://127.0.0.1:8766**. The server embeds through the public [`api.Embedder`](../api/README.md) (`embed_text` / `embed_image` / `embed_audio` + `cosine`). This package only serves the pages.

| Flag / env | Default | Meaning |
| --- | --- | --- |
| `--backend` / `ANEMLL_DEMO_BACKEND` | `mock` | **`coreai`** = real Neural Engine packages. `reference` = full checkpoint on CPU. `mock` = fake vectors so the UI runs without Core AI. |
| `--host` / `ANEMLL_DEMO_HOST` | `0.0.0.0` | Bind address. Open the UI at `127.0.0.1` if you want the microphone. |
| `--port` / `ANEMLL_DEMO_PORT` | **8766** | 8765 is often taken by AnemllAgentHost. |
| `--compute` / `ANEMLL_DEMO_COMPUTE` | `ane` | Neural Engine. Use `cpu` to force the CPU path. The demo does **not** read shell `ANEMLL_COREAI_COMPUTE`. |
| `--data-dir` / `ANEMLL_DEMO_DATA` | `~/.anemll-embeddings/demo` | Index and saved media. Must not sit inside the artifacts directory. |
| `--alert-dir` / `ANEMLL_DEMO_ALERT` | `~/.anemll-embeddings/alert` | Camera-alert frames, sounds, and Sparky’s reference photo. Must not sit inside the artifacts directory. |

`GET http://127.0.0.1:8766/health` reports the backend, the three towers, warmup time, and placement (`fullyOnANE` when the runtime says so). Each page shows that as a badge: `on ANE · N ms`.

Without a Mac / Neural Engine:

```sh
python -m pip install -r demo/requirements.txt
python -m demo.server --backend mock --port 8766
```

`mock` is only for the UI and API. Matches are hash-like stand-ins, not real cross-modal search.

## Load the sample corpus

Optional. Downloads a small CC0 / CC BY set (fox, rain, piano, and similar) via Openverse. Licenses go in `manifest.json`. Nothing is committed.

```sh
python samples/fetch_corpus.py --dest "$ANEMLL_DEMO_CORPUS"
python samples/seed_index.py --base-url http://127.0.0.1:8766 --corpus "$ANEMLL_DEMO_CORPUS"
```

`--limit-images N` and `--limit-audio N` shrink the download. `--limit N` on `seed_index.py` indexes only the first N items.

Audio conversion uses `ffmpeg` if present, otherwise macOS `afconvert` plus a trim to 8 seconds. If neither tool exists, images still download and audio topics are skipped.

Keep `$ANEMLL_DEMO_DATA`, `$ANEMLL_DEMO_CORPUS`, and `$ANEMLL_DEMO_ALERT` **outside** `$ANEMLL_EMBEDDINGS_ARTIFACTS`. The server refuses those paths if they sit inside the package directory.

You can also skip the corpus and click **Load samples** on the Search page (colored squares, two captions, two tones).

## Search — `/`

Drop photos, audio, or text on the **left** to add them to your library, then type a query like “a red fox” or drop an image or sound on the **right** to see the closest matches with similarity bars.

The whole left panel is the drop target, not only the dashed box. A failed add shows an error on that panel and is not stored.

**Try:**

1. Click **Load samples**, then search `northern lights`. The matching caption should come back near the top. The badge is the time to embed that query.
2. After seeding the corpus, search `a red fox`, then `rain` or `piano`. Toggle **Images** / **Audio** / **Text** and change **Top**.
3. Drop one of your own photos on the left, then drop another photo (or type a short description) on the right. Use **Record query** to search with a few seconds of sound.

To add a clip from disk, drop it here (Search), not on Heard.

## Heatmap — `/heatmap`

The nav label is **Matrix**. The server re-embeds the items you tick and fills a grid: every item versus every other. A higher score (brighter cell) means more alike. You need at least two library items first. The API will compare at most 12.

**Try:**

1. On Search, click **Load samples**. Open `/heatmap`. Leave the first items ticked and click **Build matrix**.
2. Look at two color squares versus each other, then a square versus the “northern lights” caption, then a square versus a tone. Same-type pairs should score higher than unrelated pairs.
3. After the corpus is loaded, include the red fox photo and search-like captions in the same grid.

## Search what I heard — `/heard`

Record a few seconds, then type what you heard. The page indexes short mic chunks with timestamps and searches **only those chunks** (this browser session’s audio). It does not search the main library’s photos or captions, and it has no file-upload control — drop clips on Search instead.

The chunk length is 5–10 seconds (default 8). On the Core AI package, audio longer than the 280-frame window is sliced and averaged.

**Try:**

1. Open **http://127.0.0.1:8766/heard** (localhost, not a LAN IP). Click **Record**, make a sound for a few seconds, click **Stop**. A row appears on the timeline with playable audio.
2. Type `a dog bark`, `piano`, or whatever you just made into **What did I hear?** Hits highlight on the timeline.
3. Record a second chunk, then search again. Only this session is searched; a reload starts a new session.

The browser will only give the microphone to a **secure context**: `http://127.0.0.1` or HTTPS. `http://<your-mac-ip>:8766` can show the pages and do text/file search, but recording will fail.

## Camera alert — `/alert`

Four alerts are already set: **Anything significant**, **UPS truck**, **Sparky**, and **Dog barking**. They are read-only. Click a camera frame or a sound and watch which ones fire. Editing rules, thresholds, and pet photos is under **Advanced / customize**.

**Anything significant** is not a text search. It is how different a frame is from the empty-street photo (`1 − cosine`). The line sits halfway between that photo (score 0) and the smallest real change. UPS, Sparky, and Dog barking ship with *threshold lines* (about 0.65, 0.75, and 0.68) — halfway between the hit and the closest miss, not the pair scores in the root README (UPS photo 0.727 vs its caption, bark 0.721 vs “a dog barking”). The page measures this set again in the background so one click is already on the right side of the line.

The frames and clips are not in git. Fetch them first (CC0 / CC BY / CC BY-SA, licenses in `manifest.json`):

```sh
python samples/fetch_alert.py --dest "${ANEMLL_DEMO_ALERT:-$HOME/.anemll-embeddings/alert}"
```

Sparky’s two photos are the same black cat (Nikolai Bulykin, Medeo, Almaty). Matching is visual similarity to that reference, not identity verification. Sounds also show a meter against “a cat meowing”; that meter is not a fifth alert.

**Try:** open `/alert` and click the empty street (nothing), the UPS truck (Significant + UPS), the FedEx truck and the person (Significant only), the ginger cat (Significant, Unknown cat), Sparky’s test photo (Significant + Sparky), the bark (Dog barking), and the meow (the cat-meowing meter, not Dog barking). Each meter shows the margin past the line.

## Limits

- Validated on an M4 Pro, macOS 27.0, with `--backend coreai`. On macOS 27.2 (M5) the vision and text packages currently fall back to the GPU; audio still runs on the Neural Engine.
- Audio must produce at least one mel frame (about 9 ms at 16 kHz). Shorter clips return `audio too short (min N ms)`.
- Browser uploads that are not WAV (webm/ogg from some mics) need `ffmpeg` on the server.
- Images are capped at 20 MiB, audio at 30 MiB.
- Weights are not in git. Check the EmbeddingGemma 2 terms before you download.

## API (if you are wiring a client)

| Method | Path | Role |
| --- | --- | --- |
| `GET` | `/health` | Backend, placement, towers, index size |
| `POST` | `/embed` | Embed only (JSON `{"text": "..."}` or one multipart file) |
| `POST` | `/index` | Embed and store |
| `POST` | `/search` | Embed a query and return top-k cosine matches (`k` 1–50, server default 8; the Search page default is 6) |
| `POST` | `/compare` | Re-embed up to 12 ids and return a cosine matrix |
| `GET` | `/items` | List (`?modality=`, `?session=`) |
| `DELETE` | `/items/{id}` | Remove |
| `GET` | `/media/{id}` | Stored JPEG or WAV |
| `GET` | `/alert/catalog` | Preset rules, frames, sounds, and whether the alert media is on disk |
| `POST` | `/alert/score` | Cosine of selected frames or sounds against the alert rules |

## Tests

```sh
python tests/test_api_embedder.py
python tests/test_download_models.py
python tests/test_demo_api.py
python tests/test_demo_coreai_masks.py
```
