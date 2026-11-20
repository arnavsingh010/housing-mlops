# ruff: noqa: E501
"""A tiny single-page UI served by the API at `/` (no build step, no extra hosting)."""

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Housing price predictor</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#1c2330; --muted:#5d6778; --accent:#2563eb; --border:#d9dee7; --ok:#0f7b45; --err:#b42318; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f141b; --card:#18202b; --text:#e8edf4; --muted:#93a0b4; --accent:#5b8def; --border:#2b3647; --ok:#4ade80; --err:#f87171; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--text); }
  main { max-width: 720px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  p.sub { margin: 0 0 20px; color: var(--muted); }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap:14px; }
  label { display:block; font-size:.85rem; font-weight:600; margin-bottom:4px; }
  label small { display:block; font-weight:400; color:var(--muted); }
  input { width:100%; padding:9px 10px; border:1px solid var(--border); border-radius:8px; background:var(--bg); color:var(--text); font-size:1rem; }
  input:focus { outline:2px solid var(--accent); outline-offset:1px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
  button { padding:10px 18px; border-radius:8px; border:1px solid var(--border); background:var(--card); color:var(--text); font-size:1rem; cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  button:disabled { opacity:.6; cursor:wait; }
  #result { margin-top:18px; padding:16px; border-radius:10px; border:1px solid var(--border); display:none; }
  #result .price { font-size:2rem; font-weight:700; color:var(--ok); }
  #result.error .msg { color:var(--err); }
  footer { margin-top:18px; font-size:.85rem; color:var(--muted); }
  a { color:var(--accent); }
</style>
</head>
<body>
<main>
  <h1>Housing price predictor</h1>
  <p class="sub">Enter details of a California census block and get a predicted median house value.</p>

  <div class="card">
    <div class="grid" id="fields"></div>
    <div class="row">
      <button class="primary" id="go">Predict price</button>
      <button id="example">Fill example</button>
    </div>
    <div id="result" role="status" aria-live="polite"></div>
  </div>

  <footer>
    <a href="/docs">API docs</a> &middot; <a href="/model-info">Model info</a> &middot; <a href="/health">Health</a>
  </footer>
</main>

<script>
const FIELDS = [
  ["MedInc", "Median income", "tens of thousands of $ (e.g. 8.3 = $83k)", 8.3252, "0.01"],
  ["HouseAge", "House age", "median age in years", 41, "1"],
  ["AveRooms", "Avg rooms", "rooms per household", 6.98, "0.01"],
  ["AveBedrms", "Avg bedrooms", "bedrooms per household", 1.02, "0.01"],
  ["Population", "Population", "people in the block", 322, "1"],
  ["AveOccup", "Avg occupants", "people per household", 2.56, "0.01"],
  ["Latitude", "Latitude", "California: 32.5 to 42", 37.88, "0.01"],
  ["Longitude", "Longitude", "California: -124.4 to -114.3", -122.23, "0.01"],
];

const fieldsEl = document.getElementById("fields");
for (const [key, label, hint, value, step] of FIELDS) {
  const wrap = document.createElement("div");
  wrap.innerHTML = '<label for="' + key + '">' + label + '<small>' + hint + '</small></label>' +
    '<input id="' + key + '" type="number" step="' + step + '" value="' + value + '" required>';
  fieldsEl.appendChild(wrap);
}

const resultEl = document.getElementById("result");
const goBtn = document.getElementById("go");

function show(html, isError) {
  resultEl.style.display = "block";
  resultEl.className = isError ? "error" : "";
  resultEl.innerHTML = html;
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

document.getElementById("example").addEventListener("click", () => {
  for (const [key, , , value] of FIELDS) document.getElementById(key).value = value;
  resultEl.style.display = "none";
});

goBtn.addEventListener("click", async () => {
  const body = {};
  for (const [key] of FIELDS) {
    const v = document.getElementById(key).value;
    if (v === "") { show('<div class="msg">Please fill in every field (' + key + ' is empty).</div>', true); return; }
    body[key] = Number(v);
  }
  goBtn.disabled = true;
  goBtn.textContent = "Predicting...";
  show("<div>Working... the first request can take up to a minute if the server was asleep.</div>", false);
  try {
    const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map(d => d.loc[d.loc.length - 1] + ": " + d.msg).join("<br>")
        : data.detail;
      show('<div class="msg">' + (Array.isArray(data.detail) ? detail : esc(String(detail))) + "</div>", true);
    } else {
      const usd = Math.round(data.predicted_value_usd).toLocaleString("en-US");
      show('<div>Predicted median house value</div><div class="price">$' + usd + "</div>", false);
    }
  } catch (err) {
    show('<div class="msg">Could not reach the server: ' + esc(String(err)) + "</div>", true);
  } finally {
    goBtn.disabled = false;
    goBtn.textContent = "Predict price";
  }
});
</script>
</body>
</html>
"""
