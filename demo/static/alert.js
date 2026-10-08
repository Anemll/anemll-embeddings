/* Camera alert: preset rules, click a frame or a sound, see what fires. */

const STORAGE_KEY = "anemll-alert-rules-v2";
const statusNode = document.getElementById("alert-status");
const errorNode = document.getElementById("alert-error");
const badge = document.getElementById("latency-badge");
const rulesNode = document.getElementById("rules");
const framesNode = document.getElementById("frames");
const soundsNode = document.getElementById("sounds");
const advancedBody = document.getElementById("advanced-body");
const fetchHint = document.getElementById("fetch-hint");

const state = {
  catalog: null,
  rules: [],
  scores: {},
  compareThreshold: null,
  compareLocked: false,
  compareFitted: false,
  calibrating: null,
  busy: false,
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[ch]));
}

function cameraIcon() {
  return `<svg class="kind-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 8h3l2-2h6l2 2h3v11H4V8z" fill="none" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="13.2" r="3" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>`;
}

function speakerIcon() {
  return `<svg class="kind-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9h3.2L12 5.5v13L7.2 15H4V9z" fill="currentColor"/><path d="M15 9.2a4 4 0 0 1 0 5.6" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;
}

function setStatus(text) {
  statusNode.textContent = text || "";
}

function setError(text) {
  errorNode.textContent = text || "";
}

function cloneRules(rules) {
  return rules.map((rule) => ({
    ...rule,
    locked: false,
    fitted: false,
  }));
}

function loadRules(defaults) {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (saved && Array.isArray(saved.rules) && saved.rules.length) {
      return saved.rules;
    }
  } catch (err) {
    /* ignore a bad local copy and use the presets */
  }
  return cloneRules(defaults);
}

function saveRules() {
  const payload = {
    rules: state.rules,
    compareThreshold: state.compareThreshold,
    compareLocked: state.compareLocked,
    compareFitted: state.compareFitted,
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function thresholdLabel(rule) {
  const number = formatScore(rule.threshold);
  if (rule.locked) return `Threshold ${number} · custom`;
  if (rule.type === "change" && rule.fitted) return `Threshold ${number} · halfway to the closest change`;
  if (rule.fitted) return `Threshold ${number} · midway between a hit and the closest miss`;
  if (rule.type === "change") return `Threshold ${number} · different from the usual empty street`;
  return `Threshold ${number} · set from the M4 samples`;
}

function formatScore(score) {
  const value = Number(score);
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (value !== 0 && abs < 0.0005) return value.toExponential(1);
  const digits = abs !== 0 && abs < 0.05 ? 4 : 3;
  return value.toFixed(digits);
}

function barWidth(score, threshold) {
  if (score == null || Number.isNaN(Number(score))) return 0;
  const value = Number(score);
  const line = Number(threshold);
  // The tick stays at the middle. ±0.08 around the line fills the bar, so a
  // thin gap (UPS vs FedEx, bark vs meow) is wide enough to see.
  const half = 0.08;
  if (!Number.isFinite(line)) return Math.max(0, Math.min(100, Math.max(value, 0) * 100));
  const pos = 0.5 + (value - line) / (2 * half);
  return Math.max(0, Math.min(100, pos * 100));
}

function formatMargin(score, threshold) {
  const delta = Number(score) - Number(threshold);
  if (!Number.isFinite(delta)) return "";
  const text = Math.abs(delta).toFixed(3);
  return delta >= 0 ? `+${text}` : `-${text}`;
}

function ruleById(id) {
  return state.rules.find((rule) => rule.id === id);
}

function ruleDecided(rule) {
  if (!rule) return false;
  if (rule.locked || rule.fitted) return true;
  return rule.pending !== true;
}

function applyThresholds() {
  Object.values(state.scores).forEach((item) => {
    (item.rules || []).forEach((entry) => {
      const rule = ruleById(entry.id);
      if (!rule || entry.score == null) return;
      entry.threshold = rule.threshold;
      entry.high = entry.score >= rule.threshold;
      entry.chip = rule.chip || rule.name;
      entry.color = rule.color;
    });
    item.fired = (item.rules || []).filter((entry) => entry.high && ruleDecided(ruleById(entry.id))).map((entry) => entry.chip);
    const photoRules = (item.rules || []).filter((entry) => {
      const rule = ruleById(entry.id);
      return rule && rule.type === "photo";
    });
    const photoDecided = photoRules.some((entry) => ruleDecided(ruleById(entry.id)));
    const photoHit = photoRules.some((entry) => entry.high && ruleDecided(ruleById(entry.id)));
    item.unknown_cat = item.kind === "cat" && photoDecided && !photoHit;
    (item.comparisons || []).forEach((entry) => {
      if (state.compareThreshold == null || entry.score == null) return;
      entry.threshold = state.compareThreshold;
      entry.high = entry.score >= state.compareThreshold;
    });
  });
}

function meterMarkup(entry, label, decided) {
  const verdict = entry.score == null || !decided ? "" : (entry.high ? "HIGH" : "LOW");
  const tone = entry.score == null || !decided ? "" : (entry.high ? "high" : "low");
  const width = barWidth(entry.score, entry.threshold);
  const number = entry.score == null ? "—" : formatScore(entry.score);
  const margin = entry.score == null || !decided ? "" : formatMargin(entry.score, entry.threshold);
  const marginClass = entry.high ? "up" : "down";
  const title = margin
    ? `Score ${number}. Threshold ${formatScore(entry.threshold)}. Margin ${margin}.`
    : "Score = how similar in meaning";
  const hint = entry.hint ? `<p class="change-hint">${esc(entry.hint)}</p>` : "";
  const tick = decided ? `<i class="threshold-tick" title="threshold"></i>` : "";
  return `<div class="rule-meter ${tone}" title="${esc(title)}">
    <div class="meter-top">
      <span>${esc(label)}</span>
      <span class="verdict">${verdict}</span>
      <span class="num">${number}${margin ? ` <span class="margin ${marginClass}">${margin}</span>` : ""}</span>
    </div>
    <div class="score" title="${esc(title)}"><span data-width="${width}"></span>${tick}</div>
    ${hint}
  </div>`;
}

function animateMeters(root) {
  root.querySelectorAll(".score span").forEach((bar) => {
    const width = bar.dataset.width || "0";
    bar.style.width = "0%";
    requestAnimationFrame(() => {
      bar.style.width = `${width}%`;
    });
  });
}

function chipsMarkup(item) {
  const parts = [];
  (item.fired || []).forEach((chip) => {
    const entry = (item.rules || []).find((row) => row.chip === chip && row.high);
    const color = (entry && entry.color) || "#e39a45";
    parts.push(`<span class="alert-chip" style="--rule:${esc(color)}">${esc(chip)}</span>`);
  });
  if (item.unknown_cat) {
    parts.push(`<span class="alert-chip unknown">Unknown cat</span>`);
  }
  if (!parts.length) {
    const pending = (item.rules || []).some((entry) => !ruleDecided(ruleById(entry.id)));
    if (pending) return `<p class="quiet-line">Score everything to set the line.</p>`;
    return `<p class="quiet-line">No alert fires</p>`;
  }
  return `<div class="chip-row">${parts.join("")}</div>`;
}

function referenceThumbs(rule) {
  const ref = (state.catalog.references || []).find((row) => row.rule_id === rule.id);
  if (!ref || !ref.available) {
    return `<p class="hint">No reference photo yet.</p>`;
  }
  const count = ref.count || 1;
  const caption = ref.caption || "Reference photo";
  const thumbs = [];
  for (let index = 0; index < count; index += 1) {
    const src = `${ref.media_url}${ref.media_url.includes("?") ? "&" : "?"}index=${index}`;
    thumbs.push(`<figure class="ref-figure"><img src="${esc(src)}" alt="${esc(caption)}"><figcaption>${esc(caption)}</figcaption></figure>`);
  }
  return `<div class="ref-row">${thumbs.join("")}</div>`;
}

function ruleKind(rule) {
  if (rule.type === "photo") return "Photo";
  if (rule.type === "change") return "Change";
  return "Text";
}

function renderRules() {
  rulesNode.innerHTML = state.rules.map((rule) => {
    const query = rule.type === "photo"
      ? referenceThumbs(rule)
      : rule.type === "change"
        ? `<p class="rule-query">${esc(rule.label || "different from the usual empty street")}</p>`
        : `<p class="rule-query">“${esc(rule.text || "")}”</p>`;
    const honesty = rule.type === "photo"
      ? `<p class="hint">Visual similarity to the reference photo, not identity verification. Another black cat may also match.</p>`
      : "";
    return `<article class="rule-card" style="--rule:${esc(rule.color)}">
      <div class="rule-kicker"><span class="type-badge">${ruleKind(rule)}</span></div>
      <h3>${esc(rule.name)}</h3>
      ${query}
      ${honesty}
      <p class="threshold-line" data-threshold-label="${esc(rule.id)}">${esc(thresholdLabel(rule))}</p>
    </article>`;
  }).join("");
}

function syncThresholdLabels() {
  state.rules.forEach((rule) => {
    const node = rulesNode.querySelector(`[data-threshold-label="${rule.id}"]`);
    if (node) node.textContent = thresholdLabel(rule);
    const input = advancedBody.querySelector(`[data-threshold="${rule.id}"]`);
    if (input && document.activeElement !== input) input.value = String(rule.threshold);
    const readout = advancedBody.querySelector(`[data-threshold-readout="${rule.id}"]`);
    if (readout) readout.textContent = Number(rule.threshold).toFixed(3);
  });
}

function tileMedia(item) {
  if (!item.available || !item.media_url) {
    return `<div class="frame-missing">Not downloaded. Drop a ${item.modality === "audio" ? "sound" : "photo"} here.</div>`;
  }
  if (item.modality === "audio") {
    return `<div class="sound-face">${speakerIcon()}<span class="play-mark" aria-hidden="true">▶</span><audio preload="none" src="${esc(item.media_url)}"></audio></div>`;
  }
  return `<img src="${esc(item.media_url)}" alt="${esc(item.caption)}">`;
}

function paintTile(tile, item) {
  const result = state.scores[item.id];
  tile.classList.toggle("scored", Boolean(result));
  tile.classList.toggle("fires", Boolean(result && result.fired && result.fired.length));
  tile.classList.toggle("quiet", Boolean(result && (!result.fired || !result.fired.length)));
  const scoreRoot = tile.querySelector(".tile-scores");
  if (!result) {
    scoreRoot.innerHTML = `<p class="quiet-line">Click to score</p>`;
    return;
  }
  const meters = (result.rules || []).map((entry) => meterMarkup(entry, entry.chip || entry.name, ruleDecided(ruleById(entry.id)))).join("");
  const compare = (result.comparisons || []).map((entry) => meterMarkup(entry, entry.label, true)).join("");
  const compareBlock = compare ? `<div class="compare-block"><p>Compared with “a cat meowing”</p>${compare}</div>` : "";
  scoreRoot.innerHTML = `${chipsMarkup(result)}${meters}${compareBlock}`;
  animateMeters(scoreRoot);
}

function renderTiles() {
  const build = (item) => {
    const icon = item.modality === "audio" ? speakerIcon() : cameraIcon();
    const credit = item.credit ? `<p class="credit">${esc(item.credit)}</p>` : "";
    return `<article class="cam-tile" data-id="${esc(item.id)}" tabindex="0" role="button" title="Score = how similar in meaning" aria-label="${esc(item.caption)}">
      <div class="tile-media">${tileMedia(item)}<span class="tile-badge">${icon}</span></div>
      <h3>${esc(item.caption)}</h3>
      ${credit}
      <div class="tile-scores"><p class="quiet-line">Click to score</p></div>
    </article>`;
  };
  framesNode.innerHTML = state.catalog.frames.map(build).join("");
  soundsNode.innerHTML = state.catalog.sounds.map(build).join("");
  [...framesNode.children, ...soundsNode.children].forEach((tile) => {
    const item = findItem(tile.dataset.id);
    tile.addEventListener("click", () => activate(item));
    tile.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate(item);
      }
    });
    bindDrop(tile, (file) => replaceItem(item, file));
    paintTile(tile, item);
  });
}

function findItem(id) {
  return [...state.catalog.frames, ...state.catalog.sounds].find((item) => item.id === id);
}

function paintAll() {
  document.querySelectorAll(".cam-tile").forEach((tile) => {
    const item = findItem(tile.dataset.id);
    if (item) paintTile(tile, item);
  });
  syncThresholdLabels();
}

async function activate(item) {
  if (!item || state.busy) return;
  if (!item.available) {
    setError(`“${item.caption}” is not downloaded. Run ${state.catalog.fetch}, or drop your own file on the tile.`);
    return;
  }
  const audio = document.querySelector(`.cam-tile[data-id="${item.id}"] audio`);
  if (audio) {
    audio.currentTime = 0;
    audio.play().catch(() => {});
  }
  await score([item.id], `Scoring ${item.caption}…`);
}

async function score(ids, label) {
  if (state.busy) return;
  state.busy = true;
  setError("");
  setStatus(label || "Scoring…");
  document.getElementById("score-all").disabled = true;
  try {
    if (state.calibrating) await state.calibrating;
    const data = await api("/alert/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_ids: ids,
        rules: state.rules.map(wireRule),
        include_compare: true,
      }),
    });
    showBadge(badge, data);
    absorb(data);
    setStatus(ids && ids.length === 1 ? `Scored ${findItem(ids[0]).caption}` : "Scored everything");
  } catch (err) {
    setError(err.message || String(err));
    setStatus("");
  } finally {
    state.busy = false;
    document.getElementById("score-all").disabled = false;
  }
}

function wireRule(rule) {
  return {
    id: rule.id,
    type: rule.type,
    name: rule.name,
    chip: rule.chip || rule.name,
    text: rule.text,
    label: rule.label || null,
    baseline_id: rule.baseline_id || null,
    scope: rule.scope || "all",
    threshold: rule.threshold,
    positive_ids: rule.positive_ids || [],
    color: rule.color,
  };
}

function takeThresholds(data) {
  Object.entries(data.suggested_thresholds || {}).forEach(([id, value]) => {
    const rule = ruleById(id);
    if (!rule || rule.locked || value == null) return;
    rule.threshold = Number(value);
    rule.fitted = true;
    rule.pending = false;
  });
  if (data.compare_threshold != null && !state.compareLocked) {
    state.compareThreshold = Number(data.compare_threshold);
    state.compareFitted = true;
  }
}

function absorb(data) {
  takeThresholds(data);
  (data.items || []).forEach((item) => {
    state.scores[item.id] = item;
  });
  applyThresholds();
  saveRules();
  renderRules();
  paintAll();
}

async function calibrate() {
  const ids = [...state.catalog.frames, ...state.catalog.sounds]
    .filter((item) => item.available)
    .map((item) => item.id);
  if (ids.length < 2) return;
  setStatus("Setting alert lines from the samples…");
  try {
    const data = await api("/alert/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        item_ids: ids,
        rules: state.rules.map(wireRule),
        include_compare: true,
      }),
    });
    takeThresholds(data);
    applyThresholds();
    saveRules();
    renderRules();
    syncThresholdLabels();
    paintAll();
  } catch (err) {
    /* The shipped M4 lines still score a click if measuring this set fails. */
  } finally {
    if (!state.busy) setStatus("");
  }
}

function bindDrop(node, handler) {
  node.addEventListener("dragenter", (event) => {
    event.preventDefault();
    node.classList.add("hot");
  });
  node.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    node.classList.add("hot");
  });
  node.addEventListener("dragleave", (event) => {
    if (event.relatedTarget && node.contains(event.relatedTarget)) return;
    node.classList.remove("hot");
  });
  node.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    node.classList.remove("hot");
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) handler(file);
  });
}

async function replaceItem(item, file) {
  const image = item.modality === "image";
  const ok = image ? file.type.startsWith("image/") : (file.type.startsWith("audio/") || /\.(wav|mp3|m4a|ogg|flac)$/i.test(file.name));
  if (!ok) {
    setError(image ? "Drop an image on a camera frame." : "Drop an audio file on a sound.");
    return;
  }
  setError("");
  setStatus(`Replacing ${item.caption}…`);
  const body = new FormData();
  body.set("file", file, file.name);
  const path = image ? `/alert/frames/${item.id}` : `/alert/sounds/${item.id}`;
  try {
    await api(path, { method: "POST", body });
    item.available = true;
    item.media_url = `/alert/media/${item.id}?t=${Date.now()}`;
    delete state.scores[item.id];
    const tile = document.querySelector(`.cam-tile[data-id="${item.id}"]`);
    if (tile) {
      tile.querySelector(".tile-media").innerHTML = `${tileMedia(item)}<span class="tile-badge">${image ? cameraIcon() : speakerIcon()}</span>`;
    }
    await score([item.id], `Scoring ${item.caption}…`);
  } catch (err) {
    setError(err.message || String(err));
  }
}

function renderAdvanced() {
  const cards = state.rules.map((rule) => {
    const textField = rule.type === "photo"
      ? `<label>Reference photos (1–3)
          <input data-photos="${esc(rule.id)}" type="file" accept="image/*" multiple>
        </label>${referenceThumbs(rule)}`
      : rule.type === "change"
        ? `<p class="hint">Fires when a frame differs from the empty-street photo. The score is 1 − cosine against that photo, not a text match.</p>`
        : `<label>Description
          <input data-text="${esc(rule.id)}" type="text" value="${esc(rule.text || "")}">
        </label>`;
    const baseline = rule.type === "text"
      ? `<label class="check"><input data-baseline="${esc(rule.id)}" type="checkbox" ${rule.baseline_id ? "checked" : ""}> Compare against the empty street</label>`
      : "";
    return `<fieldset class="adv-rule" style="--rule:${esc(rule.color)}">
      <legend><span class="type-badge">${ruleKind(rule)}</span> ${esc(rule.name)}</legend>
      <label>Name <input data-name="${esc(rule.id)}" type="text" value="${esc(rule.name)}"></label>
      ${textField}
      <label>Scope
        <select data-scope="${esc(rule.id)}">
          ${["image", "audio", "all"].map((scope) => `<option value="${scope}" ${rule.scope === scope ? "selected" : ""}>${scope}</option>`).join("")}
        </select>
      </label>
      ${baseline}
      <label>Threshold <span data-threshold-readout="${esc(rule.id)}">${Number(rule.threshold).toFixed(3)}</span>
        <input data-threshold="${esc(rule.id)}" type="range" min="-0.2" max="1" step="0.005" value="${Number(rule.threshold)}">
      </label>
      <button type="button" data-remove="${esc(rule.id)}">Remove</button>
    </fieldset>`;
  }).join("");
  advancedBody.innerHTML = `
    <p class="hint">Editing lives here. The list above stays read-only until you change a rule. Drop a photo on a camera tile, or a sound on a clip, to replace that sample. A photo rule stores the average of 1–3 reference shots.</p>
    <div class="adv-grid">${cards}</div>
    <div class="row">
      <button type="button" id="add-text">Add text alert</button>
      <button type="button" id="add-photo">Add photo alert</button>
      <button type="button" id="reset-alerts">Reset to the four presets</button>
    </div>
    <p class="hint" id="photo-hint"></p>`;
  advancedBody.querySelectorAll("[data-name]").forEach((input) => {
    input.addEventListener("input", () => {
      const rule = ruleById(input.dataset.name);
      rule.name = input.value.trim() || rule.name;
      rule.chip = rule.name;
      saveRules();
      renderRules();
      applyThresholds();
      paintAll();
    });
  });
  advancedBody.querySelectorAll("[data-text]").forEach((input) => {
    input.addEventListener("input", () => {
      const rule = ruleById(input.dataset.text);
      rule.text = input.value;
      rule.fitted = false;
      rule.locked = false;
      rule.pending = true;
      saveRules();
      renderRules();
      scheduleRescore();
    });
  });
  advancedBody.querySelectorAll("[data-scope]").forEach((input) => {
    input.addEventListener("change", () => {
      ruleById(input.dataset.scope).scope = input.value;
      saveRules();
      scheduleRescore();
    });
  });
  advancedBody.querySelectorAll("[data-baseline]").forEach((input) => {
    input.addEventListener("change", () => {
      const rule = ruleById(input.dataset.baseline);
      rule.baseline_id = input.checked ? "street" : null;
      rule.fitted = false;
      saveRules();
      renderRules();
      scheduleRescore();
    });
  });
  advancedBody.querySelectorAll("[data-threshold]").forEach((input) => {
    input.addEventListener("input", () => {
      const rule = ruleById(input.dataset.threshold);
      rule.threshold = Number(input.value);
      rule.locked = true;
      rule.fitted = false;
      saveRules();
      applyThresholds();
      renderRules();
      paintAll();
    });
  });
  advancedBody.querySelectorAll("[data-remove]").forEach((button) => {
    button.addEventListener("click", () => {
      state.rules = state.rules.filter((rule) => rule.id !== button.dataset.remove);
      if (!state.rules.length) state.rules = cloneRules(state.catalog.rules);
      saveRules();
      renderRules();
      renderAdvanced();
      applyThresholds();
      paintAll();
    });
  });
  advancedBody.querySelectorAll("[data-photos]").forEach((input) => {
    input.addEventListener("change", async () => {
      const files = [...input.files].slice(0, 3);
      if (!files.length) return;
      const body = new FormData();
      files.forEach((file) => body.append("file", file, file.name));
      setStatus("Saving reference photos…");
      try {
        await api(`/alert/references/${input.dataset.photos}`, { method: "POST", body });
        const ref = (state.catalog.references || []).find((row) => row.rule_id === input.dataset.photos);
        if (ref) {
          ref.available = true;
          ref.count = files.length;
          ref.media_url = `/alert/reference/${input.dataset.photos}`;
        }
        const rule = ruleById(input.dataset.photos);
        if (rule) {
          rule.fitted = false;
          rule.locked = false;
        }
        saveRules();
        renderRules();
        renderAdvanced();
        await rescoreKnown();
      } catch (err) {
        setError(err.message || String(err));
      }
    });
  });
  document.getElementById("add-text").addEventListener("click", () => {
    const id = `custom-${Math.random().toString(16).slice(2, 8)}`;
    state.rules.push({
      id,
      type: "text",
      name: "New alert",
      chip: "New alert",
      text: "a bicycle leaning by the door",
      baseline_id: null,
      scope: "all",
      threshold: 0.65,
      color: "#c4b4e0",
      locked: false,
      fitted: false,
      pending: true,
    });
    saveRules();
    renderRules();
    renderAdvanced();
  });
  document.getElementById("add-photo").addEventListener("click", () => {
    document.getElementById("photo-hint").textContent = "Choose 1–3 photos, then they become a new named alert.";
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = "image/*";
    picker.multiple = true;
    picker.addEventListener("change", async () => {
      const files = [...picker.files].slice(0, 3);
      if (!files.length) return;
      const id = `pet-${Math.random().toString(16).slice(2, 8)}`;
      const body = new FormData();
      files.forEach((file) => body.append("file", file, file.name));
      try {
        await api(`/alert/references/${id}`, { method: "POST", body });
        state.rules.push({
          id,
          type: "photo",
          name: "My cat",
          chip: "My cat",
          text: null,
          baseline_id: null,
          scope: "image",
          threshold: 0.75,
          color: "#d7c16e",
          locked: false,
          fitted: false,
          pending: true,
        });
        state.catalog.references.push({
          id: `${id}-ref`,
          rule_id: id,
          caption: "Reference photo",
          available: true,
          count: files.length,
          media_url: `/alert/reference/${id}`,
        });
        saveRules();
        renderRules();
        renderAdvanced();
        await rescoreKnown();
      } catch (err) {
        setError(err.message || String(err));
      }
    });
    picker.click();
  });
  document.getElementById("reset-alerts").addEventListener("click", resetAll);
}

let rescoreTimer = 0;

function scheduleRescore() {
  window.clearTimeout(rescoreTimer);
  rescoreTimer = window.setTimeout(() => {
    rescoreKnown();
  }, 400);
}

async function rescoreKnown() {
  const ids = Object.keys(state.scores).filter((id) => findItem(id) && findItem(id).available);
  if (!ids.length) return;
  await score(ids, "Updating scores…");
}

async function resetAll() {
  localStorage.removeItem(STORAGE_KEY);
  state.scores = {};
  state.compareLocked = false;
  state.compareFitted = false;
  try {
    await api("/alert/overrides", { method: "DELETE" });
  } catch (err) {
    setError(err.message || String(err));
  }
  state.rules = cloneRules(state.catalog.rules);
  state.compareThreshold = state.catalog.compare.threshold;
  saveRules();
  await reloadCatalog();
}

async function reloadCatalog() {
  state.catalog = await api("/alert/catalog");
  fetchHint.hidden = Boolean(state.catalog.ready);
  if (!state.catalog.ready) {
    fetchHint.textContent = `Some samples are not on disk yet. From the repo: ${state.catalog.fetch}`;
  }
  renderRules();
  renderTiles();
  renderAdvanced();
}

async function boot() {
  try {
    await loadHealth(document.getElementById("banner"));
    state.catalog = await api("/alert/catalog");
    state.rules = loadRules(state.catalog.rules);
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
      if (saved && saved.compareThreshold != null) state.compareThreshold = saved.compareThreshold;
      if (saved && saved.compareLocked) state.compareLocked = true;
      if (saved && saved.compareFitted) state.compareFitted = true;
    } catch (err) {
      /* presets already loaded */
    }
    if (state.compareThreshold == null) state.compareThreshold = state.catalog.compare.threshold;
    fetchHint.hidden = Boolean(state.catalog.ready);
    if (!state.catalog.ready) {
      fetchHint.textContent = `Some samples are not on disk yet. From the repo: ${state.catalog.fetch}`;
    }
    renderRules();
    renderTiles();
    renderAdvanced();
    state.calibrating = calibrate().finally(() => {
      state.calibrating = null;
    });
  } catch (err) {
    setError(err.message || String(err));
  }
}

window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

document.getElementById("score-all").addEventListener("click", async () => {
  const ids = [...state.catalog.frames, ...state.catalog.sounds].filter((item) => item.available).map((item) => item.id);
  if (!ids.length) {
    setError(`Nothing to score yet. ${state.catalog.fetch}`);
    return;
  }
  await score(ids, "Scoring every frame and sound…");
});

boot();
