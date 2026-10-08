const timeline = document.getElementById("timeline");
const results = document.getElementById("results");
const errorBox = document.getElementById("error");
const recordBadge = document.getElementById("record-badge");
const searchBadge = document.getElementById("latency-badge");
const session = `heard-${Math.random().toString(16).slice(2, 10)}`;
let mic = null;
let timer = null;
let cursor = 0;
let flushing = Promise.resolve();
const hits = new Set();

function setError(err) {
  errorBox.textContent = err ? String(err.message || err) : "";
}

function renderTimeline(items) {
  timeline.innerHTML = "";
  const rows = (items || []).slice().sort((a, b) => Number(a.t_start) - Number(b.t_start));
  if (!rows.length) {
    timeline.innerHTML = "<p class='hint'>No chunks yet.</p>";
    return;
  }
  rows.forEach((item) => {
    const row = document.createElement("div");
    row.className = "chunk" + (hits.has(item.id) ? " hit-mark" : "");
    row.dataset.id = item.id;
    const start = Number(item.t_start || 0).toFixed(1);
    const end = Number(item.t_end || 0).toFixed(1);
    row.innerHTML = `
      <div>${start}s – ${end}s</div>
      <audio controls src="${item.media_url}"></audio>`;
    timeline.appendChild(row);
  });
}

async function refresh() {
  const data = await api(`/items?session=${encodeURIComponent(session)}`);
  renderTimeline(data.items || []);
}

async function indexChunk(samples, rate, t0, t1) {
  const body = new FormData();
  body.set("file", new File([encodeWav(samples, rate)], `chunk-${t0.toFixed(1)}.wav`, { type: "audio/wav" }));
  body.set("label", `${t0.toFixed(1)}s`);
  body.set("session", session);
  body.set("t_start", String(t0));
  body.set("t_end", String(t1));
  const data = await api("/index", { method: "POST", body });
  showBadge(recordBadge, data);
  await refresh();
}

function enqueueFlush(finalPass) {
  const run = flushing.then(() => flush(finalPass));
  flushing = run.catch(() => {});
  return run;
}

async function flush(finalPass) {
  if (!mic) return;
  const stopped = finalPass ? await mic.stop() : null;
  const samples = stopped ? stopped.samples : mic.takeSamples();
  const rate = stopped ? stopped.rate : mic.rate;
  if (samples.length < 16) return;
  // A stop that lands just after a chunk boundary leaves a short tail.
  // Keep that audio only when it is the first chunk in the session.
  if (cursor > 0 && samples.length < rate) return;
  const seconds = samples.length / rate;
  const t0 = cursor;
  const t1 = cursor + seconds;
  cursor = t1;
  await indexChunk(samples, rate, t0, t1);
}

document.getElementById("record").addEventListener("click", async () => {
  const button = document.getElementById("record");
  setError("");
  try {
    if (!mic) {
      mic = new MicCapture();
      await mic.start();
      cursor = 0;
      const seconds = Math.min(10, Math.max(5, Number(document.getElementById("seconds").value) || 8));
      timer = window.setInterval(() => {
        enqueueFlush(false).catch(setError);
      }, seconds * 1000);
      button.textContent = "Stop";
      return;
    }
    button.disabled = true;
    window.clearInterval(timer);
    timer = null;
    await enqueueFlush(true);
    mic = null;
    button.textContent = "Record";
  } catch (err) {
    setError(err);
    mic = null;
    button.textContent = "Record";
  } finally {
    button.disabled = false;
  }
});

document.getElementById("find-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setError("");
  const text = document.getElementById("find").value.trim();
  if (!text) return;
  try {
    const data = await api("/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        text,
        k: 8,
        session,
        filter_modality: "audio",
      }),
    });
    showBadge(searchBadge, data.query);
    hits.clear();
    results.innerHTML = "";
    (data.results || []).forEach((hit) => {
      hits.add(hit.id);
      const row = document.createElement("article");
      row.className = "hit";
      row.innerHTML = `
        <div class="placeholder">${Number(hit.t_start || 0).toFixed(1)}s</div>
        <div class="meta">
          ${chip("audio")}
          <strong>${labelOf(hit)}</strong>
          <p>cosine ${Number(hit.score).toFixed(3)}</p>
          ${hit.media_url ? `<audio controls src="${hit.media_url}"></audio>` : ""}
        </div>`;
      results.appendChild(row);
    });
    if (!results.children.length) {
      results.innerHTML = "<p class='hint'>No chunks in this session yet.</p>";
    }
    await refresh();
  } catch (err) {
    setError(err);
  }
});

loadHealth(document.getElementById("banner")).catch(setError);
refresh().catch(setError);
