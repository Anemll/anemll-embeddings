const checks = document.getElementById("checks");
const matrix = document.getElementById("matrix");
const errorBox = document.getElementById("error");
const badge = document.getElementById("latency-badge");

function setError(err) {
  errorBox.textContent = err ? String(err.message || err) : "";
}

function heatColor(value) {
  const t = Math.max(0, Math.min(1, (Number(value) + 0.05) / 1.05));
  const ink = [28, 25, 21];
  const copper = [227, 154, 69];
  const rgb = ink.map((channel, i) => Math.round(channel + (copper[i] - channel) * t));
  return `rgb(${rgb.join(",")})`;
}

function renderMatrix(payload) {
  const items = payload.items || [];
  const scores = payload.matrix || [];
  const table = document.createElement("table");
  table.className = "heat";
  const head = document.createElement("tr");
  head.appendChild(document.createElement("th"));
  items.forEach((item) => {
    const cell = document.createElement("th");
    cell.textContent = labelOf(item);
    head.appendChild(cell);
  });
  table.appendChild(head);
  items.forEach((item, row) => {
    const tr = document.createElement("tr");
    const label = document.createElement("th");
    label.textContent = labelOf(item);
    tr.appendChild(label);
    items.forEach((_, col) => {
      const td = document.createElement("td");
      const value = scores[row][col];
      td.textContent = Number(value).toFixed(2);
      td.style.background = heatColor(value);
      td.style.color = value > 0.62 ? "#1a140d" : "#f4efe6";
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  matrix.innerHTML = "";
  matrix.appendChild(table);
}

async function loadChecks() {
  const data = await api("/items");
  checks.innerHTML = "";
  (data.items || []).forEach((item, index) => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" value="${item.id}" ${index < 6 ? "checked" : ""}> ${chip(item.modality)} ${labelOf(item)}`;
    checks.appendChild(label);
  });
  if (!checks.children.length) {
    checks.innerHTML = "<p class='hint'>Index a few items on the Search page first.</p>";
  }
}

document.getElementById("build").addEventListener("click", async () => {
  setError("");
  const ids = [...checks.querySelectorAll("input:checked")].map((node) => node.value);
  if (ids.length < 2) {
    setError(new Error("pick at least two items"));
    return;
  }
  try {
    const data = await api("/compare", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ ids }),
    });
    showBadge(badge, data);
    renderMatrix(data);
  } catch (err) {
    setError(err);
  }
});

loadHealth(document.getElementById("banner")).catch(setError);
loadChecks().catch(setError);
