const libraryPanel = document.getElementById("library-panel");
const searchPanel = document.getElementById("search-panel");
const library = document.getElementById("library");
const results = document.getElementById("results");
const preview = document.getElementById("query-preview");
const errorBox = document.getElementById("error");
const libraryError = document.getElementById("library-error");
const libraryStatus = document.getElementById("library-status");
const badge = document.getElementById("latency-badge");
const kInput = document.getElementById("k");

let mic = null;
let recording = false;
let lastQuery = null;
let queryUrl = null;

function setError(err) {
  errorBox.textContent = err ? String(err.message || err) : "";
}

function setLibraryError(err) {
  libraryError.textContent = err ? String(err.message || err) : "";
}

function currentFilter() {
  const pressed = document.querySelector(".filters button[aria-pressed='true']");
  return (pressed && pressed.dataset.filter) || "";
}

function currentK() {
  const value = Number(kInput.value);
  if (!Number.isFinite(value)) return 6;
  return Math.max(1, Math.min(50, Math.round(value)));
}

function filesFromEvent(event) {
  const transfer = event.dataTransfer;
  if (!transfer) return [];
  const found = [];
  if (transfer.files && transfer.files.length) {
    for (const file of transfer.files) found.push(file);
  }
  if (!found.length && transfer.items) {
    for (const item of transfer.items) {
      if (item.kind === "file") {
        const file = item.getAsFile();
        if (file) found.push(file);
      }
    }
  }
  return found;
}

function isTextFile(file) {
  return file.type.startsWith("text/") || /\.(txt|text|md)$/i.test(file.name);
}

function isAudioFile(file) {
  return file.type.startsWith("audio/") || /\.(wav|wave|mp3|m4a|ogg|oga|webm|flac|aac|mp4)$/i.test(file.name);
}

function modalityIcon(modality) {
  if (modality === "image") {
    return `<svg class="mod-icon" viewBox="0 0 16 16" aria-hidden="true"><rect x="1.5" y="2.5" width="13" height="11" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4"/><circle cx="5.5" cy="6.2" r="1.2" fill="currentColor"/><path d="M2.5 11.5 L6 8 L8.5 10.2 L11 7.5 L13.5 11.5" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>`;
  }
  if (modality === "audio") {
    return `<svg class="mod-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M1 8 H3 L5 4 L7 12 L9 6 L11 10 L13 8 H15" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>`;
  }
  return `<svg class="mod-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 4 H13 M3 8 H13 M3 12 H9" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>`;
}

function similarityPercent(score) {
  const value = Math.max(0, Math.min(1, Number(score) || 0));
  return Math.round(value * 100);
}

function renderLibrary(items) {
  library.innerHTML = "";
  if (!items.length) {
    library.innerHTML = "<p class='hint'>Nothing indexed yet. Drop a photo, a sound, or a caption here.</p>";
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "card";
    const audio = item.modality === "audio" && item.media_url
      ? `<audio controls src="${escapeHtml(item.media_url)}"></audio>`
      : "";
    card.innerHTML = `
      ${mediaBlock(item)}
      <div class="meta">
        ${modalityIcon(item.modality)} ${chip(item.modality)}
        <strong></strong>
        <p></p>
        ${audio}
        <button type="button">Remove</button>
      </div>`;
    card.querySelector("strong").textContent = labelOf(item);
    card.querySelector("p").textContent = item.credit || "";
    card.querySelector("button").addEventListener("click", async () => {
      setLibraryError("");
      try {
        await api(`/items/${item.id}`, { method: "DELETE" });
        await refresh();
      } catch (err) {
        setLibraryError(err);
      }
    });
    library.appendChild(card);
  });
}

function renderQuery(query) {
  if (queryUrl) {
    URL.revokeObjectURL(queryUrl);
    queryUrl = null;
  }
  preview.innerHTML = "";
  if (!query) {
    preview.innerHTML = "<p class='hint'>No query yet.</p>";
    return;
  }
  const card = document.createElement("div");
  card.className = "query-preview-card";
  if (query.type === "text") {
    const text = document.createElement("p");
    text.className = "query-text";
    text.textContent = query.text;
    card.appendChild(text);
  } else if (isAudioFile(query.file)) {
    queryUrl = URL.createObjectURL(query.file);
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.src = queryUrl;
    const name = document.createElement("strong");
    name.textContent = query.file.name || "Recording";
    card.append(audio, name);
  } else {
    queryUrl = URL.createObjectURL(query.file);
    const image = document.createElement("img");
    image.className = "query-thumb";
    image.alt = "";
    image.src = queryUrl;
    const name = document.createElement("strong");
    name.textContent = query.file.name || "Image";
    card.append(image, name);
  }
  preview.appendChild(card);
}

function renderHits(hits) {
  results.innerHTML = "";
  if (!hits.length) {
    results.innerHTML = "<p class='hint'>No library items match this search.</p>";
    return;
  }
  hits.forEach((hit) => {
    const row = document.createElement("article");
    row.className = "hit";
    const percent = similarityPercent(hit.score);
    const audio = hit.modality === "audio" && hit.media_url
      ? `<audio controls src="${escapeHtml(hit.media_url)}"></audio>`
      : "";
    row.innerHTML = `
      ${mediaBlock(hit)}
      <div class="meta">
        ${modalityIcon(hit.modality)} ${chip(hit.modality)}
        <strong></strong>
        <p class="score-label"></p>
        <div class="score"><span></span></div>
        ${audio}
      </div>`;
    row.querySelector("strong").textContent = labelOf(hit);
    row.querySelector(".score-label").textContent = `${percent}% similar`;
    row.querySelector(".score span").style.width = `${percent}%`;
    results.appendChild(row);
  });
}

async function refresh() {
  const data = await api("/items");
  renderLibrary(data.items || []);
}

async function indexForm(form) {
  await api("/index", { method: "POST", body: form });
  await refresh();
}

async function indexText(text, label) {
  const body = new FormData();
  body.set("text", text);
  if (label) body.set("label", label);
  await indexForm(body);
}

async function indexFile(file) {
  if (isTextFile(file)) {
    const text = (await file.text()).trim();
    if (!text) throw new Error(`${file.name || "That file"} has no caption text.`);
    await indexText(text, file.name);
    return;
  }
  const body = new FormData();
  body.set("file", file, file.name || "upload");
  body.set("label", file.name || "upload");
  await indexForm(body);
}

async function addFiles(files) {
  setLibraryError("");
  if (!files.length) {
    libraryStatus.textContent = "";
    setLibraryError("That drop had no file. Drop a photo, a sound, or a .txt caption.");
    return;
  }
  try {
    for (let i = 0; i < files.length; i += 1) {
      const file = files[i];
      libraryStatus.textContent = `Adding ${i + 1} of ${files.length}: ${file.name || "file"}`;
      await indexFile(file);
    }
    libraryStatus.textContent = files.length === 1
      ? `Added ${files[0].name || "item"}`
      : `Added ${files.length} items`;
  } catch (err) {
    libraryStatus.textContent = "";
    setLibraryError(err);
  }
}

async function runSearch() {
  if (!lastQuery) return;
  setError("");
  renderQuery(lastQuery);
  const body = new FormData();
  body.set("k", String(currentK()));
  const filter = currentFilter();
  if (filter) body.set("filter_modality", filter);
  if (lastQuery.type === "text") {
    body.set("text", lastQuery.text);
  } else {
    body.set("file", lastQuery.file, lastQuery.file.name || "query");
  }
  const data = await api("/search", { method: "POST", body });
  showBadge(badge, data.query);
  renderHits(data.results || []);
}

function bindPanelDrop(panel, handler) {
  panel.addEventListener("dragenter", (event) => {
    event.preventDefault();
    panel.classList.add("hot");
  });
  panel.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    panel.classList.add("hot");
  });
  panel.addEventListener("dragleave", (event) => {
    if (event.relatedTarget && panel.contains(event.relatedTarget)) return;
    panel.classList.remove("hot");
  });
  panel.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    panel.classList.remove("hot");
    handler(filesFromEvent(event));
  });
}

window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

bindPanelDrop(libraryPanel, (files) => {
  addFiles(files);
});
bindPanelDrop(searchPanel, (files) => {
  if (!files.length) {
    setError("Drop an image, a sound, or a .txt caption to search.");
    return;
  }
  const file = files[0];
  if (isTextFile(file)) {
    file.text().then((raw) => {
      const text = raw.trim();
      if (!text) {
        setError("That text file is empty.");
        return;
      }
      document.getElementById("query").value = text;
      lastQuery = { type: "text", text };
      runSearch().catch(setError);
    }).catch(setError);
    return;
  }
  lastQuery = { type: "file", file };
  runSearch().catch(setError);
});

document.getElementById("caption-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setLibraryError("");
  const input = document.getElementById("caption");
  const text = input.value.trim();
  if (!text) {
    setLibraryError("Type a caption to add.");
    return;
  }
  libraryStatus.textContent = "Adding caption…";
  try {
    await indexText(text, text);
    input.value = "";
    libraryStatus.textContent = "Added caption";
  } catch (err) {
    libraryStatus.textContent = "";
    setLibraryError(err);
  }
});

document.getElementById("query-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = document.getElementById("query").value.trim();
  if (!text) {
    setError("Type a query, or drop an image or sound.");
    return;
  }
  lastQuery = { type: "text", text };
  runSearch().catch(setError);
});

document.getElementById("query-file").addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  if (isTextFile(file)) {
    file.text().then((raw) => {
      const text = raw.trim();
      document.getElementById("query").value = text;
      lastQuery = { type: "text", text };
      return runSearch();
    }).catch(setError);
    return;
  }
  lastQuery = { type: "file", file };
  runSearch().catch(setError);
});

document.querySelectorAll(".filters button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".filters button").forEach((other) => {
      other.setAttribute("aria-pressed", other === button ? "true" : "false");
    });
    if (lastQuery) runSearch().catch(setError);
  });
});

kInput.addEventListener("change", () => {
  kInput.value = String(currentK());
  if (lastQuery) runSearch().catch(setError);
});

document.getElementById("mic").addEventListener("click", async () => {
  const button = document.getElementById("mic");
  setError("");
  try {
    if (!recording) {
      mic = new MicCapture();
      await mic.start();
      recording = true;
      button.textContent = "Stop and search";
      return;
    }
    button.disabled = true;
    const { samples, rate } = await mic.stop();
    recording = false;
    button.textContent = "Record query";
    const file = new File([encodeWav(samples, rate)], "query.wav", { type: "audio/wav" });
    lastQuery = { type: "file", file };
    await runSearch();
  } catch (err) {
    recording = false;
    button.textContent = "Record query";
    setError(err);
  } finally {
    button.disabled = false;
  }
});

document.getElementById("refresh").addEventListener("click", () => {
  refresh().catch(setLibraryError);
});

document.getElementById("samples").addEventListener("click", async () => {
  setLibraryError("");
  const button = document.getElementById("samples");
  button.disabled = true;
  try {
    const colors = [
      ["Red square", "#c44536"],
      ["Blue square", "#2f5d9f"],
      ["Green square", "#2f7d4a"],
      ["Sand square", "#d7b07a"],
    ];
    let step = 0;
    const total = colors.length + 4;
    for (const [label, color] of colors) {
      step += 1;
      libraryStatus.textContent = `Adding ${step} of ${total}: ${label}`;
      const canvas = document.createElement("canvas");
      canvas.width = 64;
      canvas.height = 64;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = color;
      ctx.fillRect(0, 0, 64, 64);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      const body = new FormData();
      body.set("file", new File([blob], `${label}.png`, { type: "image/png" }));
      body.set("label", label);
      await indexForm(body);
    }
    step += 1;
    libraryStatus.textContent = `Adding ${step} of ${total}: northern lights`;
    await indexText("the northern lights are a green glow in the sky", "northern lights");
    step += 1;
    libraryStatus.textContent = `Adding ${step} of ${total}: payroll ledger`;
    await indexText("a payroll ledger of quarterly accounts", "payroll ledger");
    const rate = 16000;
    for (const [freq, label] of [[440, "tone A4"], [880, "tone A5"]]) {
      step += 1;
      libraryStatus.textContent = `Adding ${step} of ${total}: ${label}`;
      const samples = new Float32Array(rate);
      for (let i = 0; i < samples.length; i += 1) {
        samples[i] = 0.2 * Math.sin((2 * Math.PI * freq * i) / rate);
      }
      const body = new FormData();
      body.set("file", new File([encodeWav(samples, rate)], `${label}.wav`, { type: "audio/wav" }));
      body.set("label", label);
      await indexForm(body);
    }
    libraryStatus.textContent = "Added samples";
  } catch (err) {
    libraryStatus.textContent = "";
    setLibraryError(err);
  } finally {
    button.disabled = false;
  }
});

loadHealth(document.getElementById("banner")).catch(setError);
refresh().catch(setLibraryError);
