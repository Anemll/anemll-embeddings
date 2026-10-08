/* Camera alert: pick one frame and (optionally) one sound, see which alerts fire. */

const STORAGE_KEY = "anemll-alert-rules-v2";
const statusNode = document.getElementById("alert-status");
const errorNode = document.getElementById("alert-error");
const badge = document.getElementById("latency-badge");
const infoNode = document.getElementById("info");
const infoBody = document.getElementById("info-body");
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
  frameId: null,
  soundId: null,
};

/* ---------- small helpers ---------- */

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

function shortCaption(caption) {
  return String(caption || "").replace(/^Front door cam:\s*/i, "").replace(/^Sound:\s*/i, "");
}

function formatScore(score) {
  const value = Number(score);
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (value !== 0 && abs < 0.0005) return value.toExponential(1);
  const digits = abs !== 0 && abs < 0.05 ? 4 : 3;
  return value.toFixed(digits);
}

function formatMargin(score, threshold) {
  const delta = Number(score) - Number(threshold);
  if (!Number.isFinite(delta)) return "";
  const text = Math.abs(delta).toFixed(3);
  return delta >= 0 ? `+${text}` : `-${text}`;
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

/* ---------- rules and thresholds (calibration logic unchanged) ---------- */

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

/* ---------- catalog lookups ---------- */

function allItems() {
  return [...state.catalog.frames, ...state.catalog.sounds];
}

function findItem(id) {
  return allItems().find((item) => item.id === id);
}

function selectedResults() {
  return {
    frame: state.frameId ? state.scores[state.frameId] || null : null,
    sound: state.soundId ? state.scores[state.soundId] || null : null,
  };
}

/* Which selected item a rule reads from: image rules the frame, audio rules the sound. */
function sourceFor(rule) {
  const scope = rule.scope || "all";
  if (scope === "image") return state.frameId ? [state.frameId] : [];
  if (scope === "audio") return state.soundId ? [state.soundId] : [];
  return [state.frameId, state.soundId].filter(Boolean);
}

function entryFor(rule) {
  for (const id of sourceFor(rule)) {
    const result = state.scores[id];
    if (!result) return { pending: true, id };
    const entry = (result.rules || []).find((row) => row.id === rule.id);
    if (entry && entry.score != null) return { entry, id };
  }
  return null;
}

function referenceFor(ruleId) {
  return (state.catalog.references || []).find((row) => row.rule_id === ruleId);
}

function referenceSrc(ref, index) {
  return `${ref.media_url}${ref.media_url.includes("?") ? "&" : "?"}index=${index}`;
}

/* ---------- info panel ---------- */

function firedChips(result) {
  const parts = (result.fired || []).map((chip) => {
    const entry = (result.rules || []).find((row) => row.chip === chip && row.high);
    const color = (entry && entry.color) || "#e39a45";
    return `<span class="big-chip" style="--rule:${esc(color)}">${esc(chip)}</span>`;
  });
  if (result.unknown_cat) parts.push(`<span class="big-chip unknown">Unknown cat</span>`);
  return parts;
}

function infoPart(kind, id, result) {
  const icon = kind === "frame" ? cameraIcon() : speakerIcon();
  const label = kind === "frame" ? "Frame" : "Sound";
  if (!id) {
    const empty = kind === "frame" ? "No frame selected" : "No sound playing";
    return `<div class="info-part"><span class="info-icon">${icon}</span><p class="info-empty">${empty}</p></div>`;
  }
  const item = findItem(id);
  const caption = `${label}: ${shortCaption(item ? item.caption : id)}`;
  if (!result) {
    return `<div class="info-part"><span class="info-icon">${icon}</span><p class="info-caption">${esc(caption)}</p><p class="info-pending">Scoring…</p></div>`;
  }
  const chips = firedChips(result);
  const pending = (result.rules || []).some((entry) => !ruleDecided(ruleById(entry.id)));
  const verdict = chips.length
    ? `<div class="info-chips">${chips.join("")}</div>`
    : `<p class="info-none">${pending ? "Setting the alert lines…" : "No alert fires"}</p>`;
  const hints = (result.rules || []).filter((entry) => entry.hint).map((entry) => entry.hint);
  const compare = (result.comparisons || []).filter((entry) => entry.score != null).map((entry) =>
    `Not an alert: “${esc(entry.label)}” ${formatScore(entry.score)} (${entry.high ? "above" : "below"} ${formatScore(entry.threshold)})`);
  const extra = [...hints.map(esc), ...compare];
  const hintLine = extra.length ? `<p class="info-hint">${extra.join(" · ")}</p>` : "";
  return `<div class="info-part"><span class="info-icon">${icon}</span><p class="info-caption">${esc(caption)}</p>${verdict}${hintLine}</div>`;
}

function renderInfo() {
  const { frame, sound } = selectedResults();
  const count = [frame, sound].filter(Boolean).reduce((sum, result) => sum + firedChips(result).length, 0);
  let headline;
  if (!state.frameId && !state.soundId) {
    headline = `<p class="info-headline quiet">Pick a frame or a sound</p>`;
  } else if ((state.frameId && !frame) || (state.soundId && !sound)) {
    headline = `<p class="info-headline quiet">Scoring…</p>`;
  } else if (count) {
    headline = `<p class="info-headline">${count} alert${count === 1 ? "" : "s"} fired</p>`;
  } else {
    headline = `<p class="info-headline quiet">All quiet</p>`;
  }
  infoNode.classList.toggle("fired", count > 0);
  infoBody.innerHTML = `${headline}${infoPart("frame", state.frameId, frame)}${infoPart("sound", state.soundId, sound)}`;
}

/* ---------- compact alert rows ---------- */

function renderRules() {
  rulesNode.innerHTML = state.rules.map((rule) => {
    const ref = rule.type === "photo" ? referenceFor(rule.id) : null;
    const thumb = ref && ref.available
      ? `<img class="arow-thumb" src="${esc(referenceSrc(ref, 0))}" alt="${esc(ref.caption || "Reference photo")}" title="${esc(ref.caption || "Reference photo")}">`
      : "";
    const found = entryFor(rule);
    const decided = ruleDecided(rule);
    let stateText = "—";
    let meter = "";
    let fired = false;
    const scope = rule.scope === "audio" ? "a sound" : rule.scope === "image" ? "a frame" : "a frame or a sound";
    if (!found) {
      meter = `<span class="arow-scope">Pick ${scope}</span>`;
    } else if (found.pending) {
      stateText = "…";
      meter = `<span class="arow-scope">Scoring…</span>`;
    } else {
      const { entry } = found;
      fired = Boolean(entry.high && decided);
      stateText = !decided ? "SETTING" : (fired ? "FIRED" : "NOT FIRED");
      const margin = decided ? formatMargin(entry.score, entry.threshold) : "";
      const marginHtml = margin ? ` <span class="${entry.high ? "up" : "down"}">${margin}</span>` : "";
      const title = `Score ${formatScore(entry.score)} · threshold ${formatScore(entry.threshold)}${margin ? ` · margin ${margin}` : ""}`;
      meter = `<div class="mini-bar" title="${esc(title)}"><span data-width="${barWidth(entry.score, entry.threshold)}"></span>${decided ? "<i></i>" : ""}</div>
        <span class="arow-num">${formatScore(entry.score)}${marginHtml}</span>`;
    }
    return `<div class="arow${fired ? " fired" : ""}" style="--rule:${esc(rule.color)}" title="${esc(thresholdLabel(rule))}">
      <span class="arow-dot"></span>
      <span class="arow-name">${thumb}<span>${esc(rule.name)}</span></span>
      <span class="arow-state">${stateText}</span>
      <div class="arow-meter">${meter}</div>
    </div>`;
  }).join("");
  rulesNode.querySelectorAll(".mini-bar span").forEach((bar) => {
    bar.style.width = "0%";
    requestAnimationFrame(() => {
      bar.style.width = `${bar.dataset.width || 0}%`;
    });
  });
}

/* ---------- camera frames and sounds ---------- */

function firedDots(result) {
  if (!result) return "";
  const dots = (result.fired || []).map((chip) => {
    const entry = (result.rules || []).find((row) => row.chip === chip && row.high);
    return `<i style="--rule:${esc((entry && entry.color) || "#e39a45")}" title="${esc(chip)}"></i>`;
  });
  if (result.unknown_cat) dots.push(`<i class="unknown" title="Unknown cat"></i>`);
  return dots.length ? `<span class="dots">${dots.join("")}</span>` : "";
}

function frameMedia(item) {
  if (!item.available || !item.media_url) {
    return `<div class="frame-missing">Not downloaded. Drop a photo here.</div>`;
  }
  return `<img src="${esc(item.media_url)}" alt="${esc(item.caption)}" draggable="false">`;
}

function renderFrames() {
  framesNode.innerHTML = state.catalog.frames.map((item) => `
    <button type="button" class="frame" role="radio" data-id="${esc(item.id)}" aria-checked="false" title="${esc(item.caption)}${item.credit ? ` · ${esc(item.credit)}` : ""}">
      <div class="frame-media">${frameMedia(item)}</div>
      <div class="frame-foot"><span class="frame-cap">${esc(shortCaption(item.caption))}</span><span class="frame-dots"></span></div>
    </button>`).join("");
  framesNode.querySelectorAll(".frame").forEach((node) => {
    const item = findItem(node.dataset.id);
    node.addEventListener("click", () => selectFrame(item));
    bindDrop(node, (file) => replaceItem(item, file));
  });
}

function renderSounds() {
  soundsNode.innerHTML = state.catalog.sounds.map((item) => `
    <button type="button" class="sound" data-id="${esc(item.id)}" aria-pressed="false" title="${esc(item.caption)}">
      <span class="sound-icon">${speakerIcon()}</span>
      <span class="sound-cap">${esc(shortCaption(item.caption))}</span>
      <span class="frame-dots"></span>
      <span class="sound-state">OFF</span>
      ${item.available && item.media_url ? `<audio preload="none" src="${esc(item.media_url)}"></audio>` : ""}
    </button>`).join("");
  soundsNode.querySelectorAll(".sound").forEach((node) => {
    const item = findItem(node.dataset.id);
    node.addEventListener("click", () => toggleSound(item));
    bindDrop(node, (file) => replaceItem(item, file));
  });
}

function paintFeed() {
  framesNode.querySelectorAll(".frame").forEach((node) => {
    const id = node.dataset.id;
    node.setAttribute("aria-checked", id === state.frameId ? "true" : "false");
    node.querySelector(".frame-dots").innerHTML = firedDots(state.scores[id]);
  });
  soundsNode.querySelectorAll(".sound").forEach((node) => {
    const id = node.dataset.id;
    const on = id === state.soundId;
    node.setAttribute("aria-pressed", on ? "true" : "false");
    node.querySelector(".sound-state").textContent = on ? "ON" : "OFF";
    node.querySelector(".frame-dots").innerHTML = firedDots(state.scores[id]);
  });
}

function paintAll() {
  renderInfo();
  renderRules();
  paintFeed();
  syncAdvanced();
}

function stopSounds(exceptId) {
  soundsNode.querySelectorAll(".sound").forEach((node) => {
    if (node.dataset.id === exceptId) return;
    const audio = node.querySelector("audio");
    if (audio) audio.pause();
  });
}

function selectFrame(item) {
  if (!item) return;
  if (!item.available) {
    setError(`“${item.caption}” is not downloaded. Run ${state.catalog.fetch}, or drop your own photo on it.`);
    return;
  }
  setError("");
  state.frameId = item.id;
  paintAll();
  ensureScored();
}

function toggleSound(item) {
  if (!item) return;
  if (state.soundId === item.id) {
    state.soundId = null;
    stopSounds(null);
    paintAll();
    return;
  }
  if (!item.available) {
    setError(`“${item.caption}” is not downloaded. Run ${state.catalog.fetch}, or drop your own sound on it.`);
    return;
  }
  setError("");
  state.soundId = item.id;
  stopSounds(item.id);
  const audio = soundsNode.querySelector(`.sound[data-id="${item.id}"] audio`);
  if (audio) {
    audio.currentTime = 0;
    audio.play().catch(() => {});
  }
  paintAll();
  ensureScored();
}

/* Score whatever is selected and not scored yet; clicks during a request queue up. */
async function ensureScored() {
  if (state.busy) return;
  const ids = [state.frameId, state.soundId].filter((id) => id && !state.scores[id]);
  if (!ids.length) return;
  const names = ids.map((id) => shortCaption(findItem(id).caption)).join(" + ");
  await score(ids, `Scoring ${names}…`);
  ensureScored();
}

/* ---------- scoring ---------- */

async function score(ids, label) {
  if (state.busy) return;
  state.busy = true;
  setError("");
  setStatus(label || "Scoring…");
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
    showLatency(data);
    absorb(data);
    setStatus("");
  } catch (err) {
    setError(err.message || String(err));
    setStatus("");
  } finally {
    state.busy = false;
  }
}

/* Cached embeddings come back as 0 ms; say so instead of claiming 0 ms. */
function showLatency(data) {
  showBadge(badge, data);
  if (data && Number(data.fresh_embeds) === 0) {
    badge.textContent = latencyBadge(data).replace(/ · \d+ ms$/, " · cached");
  }
}

function absorb(data) {
  takeThresholds(data);
  (data.items || []).forEach((item) => {
    state.scores[item.id] = item;
  });
  applyThresholds();
  saveRules();
  paintAll();
}

async function calibrate() {
  const ids = allItems().filter((item) => item.available).map((item) => item.id);
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
    paintAll();
  } catch (err) {
    /* The shipped M4 lines still score a click if measuring this set fails. */
  } finally {
    if (!state.busy) setStatus("");
  }
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

/* ---------- drop to replace a sample ---------- */

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
    if (image) {
      renderFrames();
      selectFrame(item);
    } else {
      renderSounds();
      state.soundId = null;
      toggleSound(item);
    }
  } catch (err) {
    setError(err.message || String(err));
  }
}

/* ---------- advanced: rule text, thresholds, reference photos ---------- */

function ruleKind(rule) {
  if (rule.type === "photo") return "Photo";
  if (rule.type === "change") return "Change";
  return "Text";
}

function syncAdvanced() {
  state.rules.forEach((rule) => {
    const label = advancedBody.querySelector(`[data-threshold-label="${rule.id}"]`);
    if (label) label.textContent = thresholdLabel(rule);
    const input = advancedBody.querySelector(`[data-threshold="${rule.id}"]`);
    if (input && document.activeElement !== input) input.value = String(rule.threshold);
    const readout = advancedBody.querySelector(`[data-threshold-readout="${rule.id}"]`);
    if (readout) readout.textContent = Number(rule.threshold).toFixed(3);
  });
}

function referenceThumbs(rule) {
  const ref = referenceFor(rule.id);
  if (!ref || !ref.available) return `<p class="adv-honest">No reference photo yet.</p>`;
  const thumbs = [];
  for (let index = 0; index < (ref.count || 1); index += 1) {
    thumbs.push(`<img src="${esc(referenceSrc(ref, index))}" alt="${esc(ref.caption || "Reference photo")}">`);
  }
  const credit = ref.credit ? `<p class="adv-honest">${esc(ref.credit)}</p>` : "";
  return `<div class="adv-refs">${thumbs.join("")}</div>${credit}`;
}

function renderAdvanced() {
  const cards = state.rules.map((rule) => {
    const query = rule.type === "photo"
      ? `${referenceThumbs(rule)}
         <p class="adv-honest">Visual similarity to the reference photo, not identity verification. Another black cat may also match.</p>
         <label>Reference photos (1–3)<input data-photos="${esc(rule.id)}" type="file" accept="image/*" multiple></label>`
      : rule.type === "change"
        ? `<p class="adv-query">${esc(rule.label || "different from the usual empty street")}</p>
           <p class="adv-honest">Fires when a frame differs from the empty-street photo. The score is 1 − cosine against that photo, not a text match.</p>`
        : `<label>Description<input data-text="${esc(rule.id)}" type="text" value="${esc(rule.text || "")}"></label>`;
    const baseline = rule.type === "text"
      ? `<label class="adv-check"><input data-baseline="${esc(rule.id)}" type="checkbox" ${rule.baseline_id ? "checked" : ""}> Compare against the empty street</label>`
      : "";
    return `<fieldset class="adv-rule" style="--rule:${esc(rule.color)}">
      <legend>${ruleKind(rule)} · ${esc(rule.name)}</legend>
      <label>Name<input data-name="${esc(rule.id)}" type="text" value="${esc(rule.name)}"></label>
      ${query}
      <label>Scope
        <select data-scope="${esc(rule.id)}">
          ${["image", "audio", "all"].map((scope) => `<option value="${scope}" ${rule.scope === scope ? "selected" : ""}>${scope}</option>`).join("")}
        </select>
      </label>
      ${baseline}
      <label>Threshold <span data-threshold-readout="${esc(rule.id)}">${Number(rule.threshold).toFixed(3)}</span>
        <input data-threshold="${esc(rule.id)}" type="range" min="-0.2" max="1" step="0.005" value="${Number(rule.threshold)}">
      </label>
      <p class="adv-threshold" data-threshold-label="${esc(rule.id)}">${esc(thresholdLabel(rule))}</p>
      <button type="button" data-remove="${esc(rule.id)}">Remove</button>
    </fieldset>`;
  }).join("");
  advancedBody.innerHTML = `
    <div class="adv-rules">${cards}</div>
    <div class="adv-actions">
      <button type="button" id="score-all">Score every sample</button>
      <button type="button" id="add-text">Add text alert</button>
      <button type="button" id="add-photo">Add photo alert</button>
      <button type="button" id="reset-alerts">Reset to the four presets</button>
    </div>
    <p class="adv-note" id="photo-hint">Drop a photo on a frame, or a sound on a clip, to replace that sample. Cosine on L2-normalized vectors; the badge is the embed time for that click.</p>`;

  advancedBody.querySelectorAll("[data-name]").forEach((input) => {
    input.addEventListener("input", () => {
      const rule = ruleById(input.dataset.name);
      rule.name = input.value.trim() || rule.name;
      rule.chip = rule.name;
      saveRules();
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
      paintAll();
      scheduleRescore();
    });
  });
  advancedBody.querySelectorAll("[data-scope]").forEach((input) => {
    input.addEventListener("change", () => {
      ruleById(input.dataset.scope).scope = input.value;
      saveRules();
      paintAll();
      scheduleRescore();
    });
  });
  advancedBody.querySelectorAll("[data-baseline]").forEach((input) => {
    input.addEventListener("change", () => {
      const rule = ruleById(input.dataset.baseline);
      rule.baseline_id = input.checked ? "street" : null;
      rule.fitted = false;
      saveRules();
      paintAll();
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
      paintAll();
    });
  });
  advancedBody.querySelectorAll("[data-remove]").forEach((button) => {
    button.addEventListener("click", () => {
      state.rules = state.rules.filter((rule) => rule.id !== button.dataset.remove);
      if (!state.rules.length) state.rules = cloneRules(state.catalog.rules);
      saveRules();
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
        const ref = referenceFor(input.dataset.photos);
        if (ref) {
          ref.available = true;
          ref.count = files.length;
          ref.media_url = `/alert/reference/${input.dataset.photos}?t=${Date.now()}`;
        }
        const rule = ruleById(input.dataset.photos);
        if (rule) {
          rule.fitted = false;
          rule.locked = false;
        }
        saveRules();
        renderAdvanced();
        paintAll();
        await rescoreKnown();
      } catch (err) {
        setError(err.message || String(err));
      }
    });
  });
  document.getElementById("score-all").addEventListener("click", async () => {
    const ids = allItems().filter((item) => item.available).map((item) => item.id);
    if (!ids.length) {
      setError(`Nothing to score yet. ${state.catalog.fetch}`);
      return;
    }
    await score(ids, "Scoring every frame and sound…");
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
    renderAdvanced();
    paintAll();
  });
  document.getElementById("add-photo").addEventListener("click", () => {
    document.getElementById("photo-hint").textContent = "Choose 1–3 photos; they become a new named alert.";
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
        renderAdvanced();
        paintAll();
        await rescoreKnown();
      } catch (err) {
        setError(err.message || String(err));
      }
    });
    picker.click();
  });
  document.getElementById("reset-alerts").addEventListener("click", resetAll);
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
  ensureScored();
}

async function reloadCatalog() {
  state.catalog = await api("/alert/catalog");
  showFetchHint();
  renderFrames();
  renderSounds();
  renderAdvanced();
  paintAll();
}

function showFetchHint() {
  fetchHint.hidden = Boolean(state.catalog.ready);
  if (!state.catalog.ready) {
    fetchHint.textContent = `Some samples are not on disk yet. From the repo: ${state.catalog.fetch}`;
  }
}

/* ---------- boot ---------- */

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
    showFetchHint();
    renderFrames();
    renderSounds();
    renderAdvanced();
    paintAll();
    state.calibrating = calibrate().finally(() => {
      state.calibrating = null;
    });
  } catch (err) {
    setError(err.message || String(err));
  }
}

window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

boot();
