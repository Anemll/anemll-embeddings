const drop = document.getElementById("drop");
const library = document.getElementById("library");
const results = document.getElementById("results");
const errorBox = document.getElementById("error");
const badge = document.getElementById("latency-badge");
let mic = null;
let recording = false;

function setError(err) {
  errorBox.textContent = err ? String(err.message || err) : "";
}

function renderLibrary(items) {
  library.innerHTML = "";
  if (!items.length) {
    library.innerHTML = "<p class='hint'>Nothing indexed yet.</p>";
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "card";
    const audio = item.modality === "audio" && item.media_url
      ? `<audio controls src="${item.media_url}"></audio>`
      : "";
    card.innerHTML = `
      ${mediaBlock(item)}
      <div class="meta">
        ${chip(item.modality)}
        <strong>${labelOf(item)}</strong>
        <p>${item.credit || ""}</p>
        ${audio}
        <button type="button" data-id="${item.id}">Remove</button>
      </div>`;
    card.querySelector("button").addEventListener("click", async () => {
      await api(`/items/${item.id}`, { method: "DELETE" });
      await refresh();
    });
    library.appendChild(card);
  });
}

function renderHits(hits) {
  results.innerHTML = "";
  if (!hits.length) {
    results.innerHTML = "<p class='hint'>No indexed items yet.</p>";
    return;
  }
  hits.forEach((hit) => {
    const row = document.createElement("article");
    row.className = "hit";
    const width = Math.max(0, Math.min(100, Math.round((hit.score + 1) * 50)));
    const audio = hit.modality === "audio" && hit.media_url
      ? `<audio controls src="${hit.media_url}"></audio>`
      : "";
    row.innerHTML = `
      ${mediaBlock(hit)}
      <div class="meta">
        ${chip(hit.modality)}
        <strong>${labelOf(hit)}</strong>
        <p>cosine ${Number(hit.score).toFixed(3)}</p>
        <div class="score"><span style="width:${width}%"></span></div>
        ${audio}
      </div>`;
    results.appendChild(row);
  });
}

async function refresh() {
  const data = await api("/items");
  renderLibrary(data.items || []);
}

async function indexForm(form) {
  const data = await api("/index", { method: "POST", body: form });
  showBadge(badge, data);
  await refresh();
}

async function indexText(text, label) {
  const body = new FormData();
  body.set("text", text);
  if (label) body.set("label", label);
  await indexForm(body);
}

async function indexFile(file) {
  if (file.type.startsWith("text/") || file.name.endsWith(".txt")) {
    const text = await file.text();
    await indexText(text.trim(), file.name);
    return;
  }
  const body = new FormData();
  body.set("file", file);
  body.set("label", file.name);
  await indexForm(body);
}

drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("hot");
});
drop.addEventListener("dragleave", () => drop.classList.remove("hot"));
drop.addEventListener("drop", async (event) => {
  event.preventDefault();
  drop.classList.remove("hot");
  setError("");
  try {
    for (const file of event.dataTransfer.files) await indexFile(file);
  } catch (err) {
    setError(err);
  }
});

document.getElementById("caption-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setError("");
  const input = document.getElementById("caption");
  try {
    await indexText(input.value.trim(), input.value.trim());
    input.value = "";
  } catch (err) {
    setError(err);
  }
});

document.getElementById("query-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setError("");
  const text = document.getElementById("query").value.trim();
  if (!text) return;
  try {
    const data = await api("/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, k: 8 }),
    });
    showBadge(badge, data.query);
    renderHits(data.results || []);
  } catch (err) {
    setError(err);
  }
});

document.getElementById("query-image").addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  setError("");
  try {
    const body = new FormData();
    body.set("file", file);
    body.set("k", "8");
    const data = await api("/search", { method: "POST", body });
    showBadge(badge, data.query);
    renderHits(data.results || []);
  } catch (err) {
    setError(err);
  }
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
    const blob = encodeWav(samples, rate);
    const body = new FormData();
    body.set("file", new File([blob], "query.wav", { type: "audio/wav" }));
    body.set("k", "8");
    const data = await api("/search", { method: "POST", body });
    showBadge(badge, data.query);
    renderHits(data.results || []);
  } catch (err) {
    recording = false;
    button.textContent = "Record query";
    setError(err);
  } finally {
    button.disabled = false;
  }
});

document.getElementById("refresh").addEventListener("click", () => refresh().catch(setError));

document.getElementById("samples").addEventListener("click", async () => {
  setError("");
  const button = document.getElementById("samples");
  button.disabled = true;
  try {
    const colors = [
      ["Red square", "#c44536"],
      ["Blue square", "#2f5d9f"],
      ["Green square", "#2f7d4a"],
      ["Sand square", "#d7b07a"],
    ];
    for (const [label, color] of colors) {
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
    await indexText("the northern lights are a green glow in the sky", "northern lights");
    await indexText("a payroll ledger of quarterly accounts", "payroll ledger");
    const rate = 16000;
    for (const [freq, label] of [[440, "tone A4"], [880, "tone A5"]]) {
      const samples = new Float32Array(rate);
      for (let i = 0; i < samples.length; i += 1) {
        samples[i] = 0.2 * Math.sin((2 * Math.PI * freq * i) / rate);
      }
      const body = new FormData();
      body.set("file", new File([encodeWav(samples, rate)], `${label}.wav`, { type: "audio/wav" }));
      body.set("label", label);
      await indexForm(body);
    }
  } catch (err) {
    setError(err);
  } finally {
    button.disabled = false;
  }
});

loadHealth(document.getElementById("banner")).catch(setError);
refresh().catch(setError);
