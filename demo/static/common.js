/* Shared helpers for the showcase pages. No build step. */

function latencyBadge(info) {
  const ms = Math.round(Number(info && info.latency_ms) || 0);
  const placement = (info && info.placement) || "";
  const where = placement === "fullyOnANE" ? "ANE" : (placement || (info && info.backend) || "cpu");
  return `on ${where} · ${ms} ms`;
}

function showBadge(node, info) {
  if (!node) return;
  node.hidden = false;
  node.textContent = latencyBadge(info);
  node.dataset.placement = (info && info.placement) || "";
}

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail || data.error || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function encodeWav(samples, sampleRate) {
  const n = samples.length;
  const buffer = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  write(0, "RIFF");
  view.setUint32(4, 36 + n * 2, true);
  write(8, "WAVE");
  write(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, "data");
  view.setUint32(40, n * 2, true);
  let offset = 44;
  for (let i = 0; i < n; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function concatFloat32(chunks) {
  const n = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Float32Array(n);
  let offset = 0;
  chunks.forEach((chunk) => {
    out.set(chunk, offset);
    offset += chunk.length;
  });
  return out;
}

class MicCapture {
  constructor() {
    this.chunks = [];
    this.rate = 48000;
    this._ctx = null;
    this._stream = null;
  }

  async start() {
    this.chunks = [];
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this._ctx = new AudioContext();
    this.rate = this._ctx.sampleRate;
    const source = this._ctx.createMediaStreamSource(this._stream);
    const workletUrl = "/static/recorder-worklet.js";
    try {
      await this._ctx.audioWorklet.addModule(workletUrl);
      const node = new AudioWorkletNode(this._ctx, "capture");
      node.port.onmessage = (event) => {
        this.chunks.push(new Float32Array(event.data));
      };
      source.connect(node);
      this._node = node;
    } catch (err) {
      const processor = this._ctx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        this.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      const sink = this._ctx.createGain();
      sink.gain.value = 0;
      source.connect(processor);
      processor.connect(sink);
      sink.connect(this._ctx.destination);
      this._node = processor;
    }
  }

  takeSamples() {
    const samples = concatFloat32(this.chunks);
    this.chunks = [];
    return samples;
  }

  async stop() {
    const samples = this.takeSamples();
    if (this._node) this._node.disconnect();
    if (this._stream) this._stream.getTracks().forEach((track) => track.stop());
    if (this._ctx) await this._ctx.close();
    this._node = null;
    this._stream = null;
    this._ctx = null;
    return { samples, rate: this.rate };
  }
}

/* Escape text for an HTML string. Prefer textContent / DOM nodes for
   anything a user typed; this is for the few fixed templates below. */
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[ch]));
}

/* Build an element with text content only (never parsed as HTML). */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function chipNode(modality) {
  return el("span", `chip ${String(modality || "").replace(/[^a-z]/gi, "")}`, modality);
}

function chip(modality) {
  const safe = escapeHtml(modality);
  return `<span class="chip ${safe}">${safe}</span>`;
}

function mediaBlock(item) {
  if (item.modality === "image" && item.media_url) {
    return `<img class="thumb" alt="" src="${escapeHtml(item.media_url)}">`;
  }
  if (item.modality === "audio" && item.media_url) {
    return `<div class="placeholder">audio</div>`;
  }
  return `<div class="placeholder">${escapeHtml(item.modality)}</div>`;
}

function labelOf(item) {
  return item.label || item.text || item.modality;
}

async function loadHealth(banner) {
  const health = await api("/health");
  if (!banner) return health;
  const names = (health.towers || []).map((tower) => tower.name).join(", ");
  if (health.backend === "coreai") {
    banner.textContent = `Core AI backend · ${health.placement || "placement pending"} · towers ${names}`;
  } else {
    banner.textContent = `Running the ${health.backend} backend (${health.placement || "cpu"}). ` +
      "Vectors are deterministic stand-ins so these pages work without the Neural Engine. " +
      "On a Mac, start with --backend coreai (ANE is the default compute).";
  }
  return health;
}
