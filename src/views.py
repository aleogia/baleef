from config import LANG_NAMES, TARGET_LANG

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>baleef</title>
  <style>
    :root {
      --bg: #000000;
      --text: #ffffff;
      --font-size: 32px;
      --font-family: 'Segoe UI', system-ui, sans-serif;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-family);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }

    .side {
      flex: 1;
      display: flex;
      flex-direction: column;
      padding: 28px 36px;
      gap: 16px;
      min-width: 0;
    }

    /* ── Top bar ── */
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }
    .side-label {
      font-size: 10px;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: #383838;
    }
    .lang-fixed {
      font-size: 14px;
      color: #555;
      padding: 7px 12px;
    }
    .lang-select {
      background: #111;
      color: #aaa;
      border: 1px solid #252525;
      border-radius: 8px;
      padding: 7px 32px 7px 12px;
      font-size: 14px;
      cursor: pointer;
      outline: none;
      appearance: none;
      -webkit-appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23555' d='M5 6L0 0h10z'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-position: right 10px center;
      transition: border-color 0.2s;
    }
    .lang-select:hover { border-color: #444; color: #fff; }
    .dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #1e1e1e;
      transition: background 0.3s;
      flex-shrink: 0;
    }
    .dot.active { background: #22c55e; }

    /* ── Feed: 4 phrases, newest at bottom ── */
    .feed {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      gap: 0;
      overflow: hidden;
    }
    .feed-item {
      padding: 14px 0;
      border-top: 1px solid #141414;
      animation: slideUp 0.3s ease;
    }
    .feed-item:first-child { border-top: none; }
    /* Newest item always full opacity, older items progressively dimmed */
    .feed-item { opacity: 0.15; }
    .feed-item:nth-last-child(4) { opacity: 0.2; }
    .feed-item:nth-last-child(3) { opacity: 0.4; }
    .feed-item:nth-last-child(2) { opacity: 0.7; }
    .feed-item:nth-last-child(1) { opacity: 1; }
    .feed-item .f-translated {
      font-size: var(--font-size);
      font-weight: 300;
      line-height: 1.35;
      word-break: break-word;
    }
    .feed-item .f-original {
      font-size: 13px;
      color: #555;
      font-style: italic;
      margin-top: 4px;
    }
    .feed-item .f-meta {
      font-size: 11px;
      color: #282828;
      margin-top: 3px;
    }
    @keyframes slideUp {
      from { opacity: 0; transform: translateY(12px); }
      to   { transform: translateY(0); }
    }
    #page-controls {
      position: fixed; bottom: 14px; right: 14px;
      display: flex; gap: 6px; opacity: 0.08; transition: opacity 0.25s; z-index: 99;
    }
    #page-controls:hover { opacity: 1; }
    .page-btn {
      background: #111; color: #777; border: 1px solid #2a2a2a;
      border-radius: 5px; padding: 5px 11px; font-size: 11px;
      letter-spacing: 0.5px; cursor: pointer;
    }
    .page-btn:hover { color: #ddd; border-color: #444; }
    .page-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="side">
    <div class="topbar">
      <div class="side-label">baleef</div>
      <select class="lang-select" onchange="setLang('A', this.value)">
        <option value="fra_Latn">French</option>
        <option value="eng_Latn" selected>English</option>
        <option value="spa_Latn">Spanish</option>
        <option value="deu_Latn">German</option>
        <option value="arb_Arab">Arabic</option>
        <option value="zho_Hans">Chinese</option>
        <option value="jpn_Jpan">Japanese</option>
        <option value="por_Latn">Portuguese</option>
        <option value="rus_Cyrl">Russian</option>
        <option value="ita_Latn">Italian</option>
        <option value="hin_Deva">Hindi</option>
      </select>
      <div class="dot" id="dotA"></div>
    </div>
    <div class="feed" id="feedA"></div>
  </div>

  <script>
    let MAX = 4;
    const SPEAKER_COLORS = ['#60a5fa','#f472b6','#4ade80','#fb923c','#c084fc','#fbbf24','#2dd4bf','#f87171'];

    function applyConfig(cfg) {
      const r = document.documentElement.style;
      if (cfg.bg_color   !== undefined) r.setProperty('--bg',          cfg.bg_color);
      if (cfg.text_color !== undefined) r.setProperty('--text',        cfg.text_color);
      if (cfg.font_size  !== undefined) r.setProperty('--font-size',   cfg.font_size + 'px');
      if (cfg.font_family!== undefined) r.setProperty('--font-family', cfg.font_family);
      if (cfg.max_phrases !== undefined) {
        MAX = cfg.max_phrases;
        ['A','B'].forEach(s => {
          const f = document.getElementById('feed' + s);
          if (f) while (f.children.length > MAX) f.removeChild(f.firstChild);
        });
      }
      if (cfg.custom_fonts) cfg.custom_fonts.forEach(fn => {
        if (document.querySelector('[data-font="' + fn + '"]')) return;
        const nm = fn.replace(/\\.[^.]+$/, '').replace(/_/g, ' ');
        const st = document.createElement('style');
        st.setAttribute('data-font', fn);
        st.textContent = "@font-face{font-family:'" + nm + "';src:url('/fonts/" + fn + "')}";
        document.head.appendChild(st);
      });
    }

    function connectConfig() {
      const ws = new WebSocket('ws://' + location.host + '/ws/config');
      ws.onmessage = e => applyConfig(JSON.parse(e.data));
      ws.onclose = () => setTimeout(connectConfig, 2000);
    }

    function setLang(side, code) {
      fetch('/lang/' + side + '?code=' + code, { method: 'POST' });
    }

    function esc(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function addItem(side, d) {
      const feed = document.getElementById('feed' + side);

      const item = document.createElement('div');
      item.className = 'feed-item';
      const color = SPEAKER_COLORS[(d.speaker_id || 0) % SPEAKER_COLORS.length];
      item.innerHTML =
        '<div class="f-translated" style="color:' + color + '">' + esc(d.translated) + '</div>' +
        '<div class="f-original">' + esc(d.original) + '</div>' +
        '<div class="f-meta">' + esc(d.src_lang) + ' → ' + esc(d.tgt_lang) + ' · ' + d.ms + 'ms</div>';

      feed.appendChild(item);

      while (feed.children.length > MAX) feed.removeChild(feed.firstChild);
    }

    function connect(side) {
      const dot = document.getElementById('dot' + side);
      const ws  = new WebSocket('ws://' + location.host + '/ws/' + side);

      ws.onopen = () => {
        document.getElementById('feed' + side).innerHTML = '';
      };
      ws.onmessage = e => {
        const d = JSON.parse(e.data);
        addItem(side, d);
        dot.classList.add('active');
        setTimeout(() => dot.classList.remove('active'), 600);
      };
      ws.onclose = () => setTimeout(() => connect(side), 1500);
    }

    connect('A');
    connectConfig();

    function hardRefresh() {
      location.replace(location.pathname + '?_=' + Date.now());
    }

    async function restartServer() {
      if (!confirm('Redémarrer le serveur ?')) return;
      const btn = document.getElementById('restart-btn');
      btn.textContent = '…'; btn.disabled = true;
      try { await fetch('/restart', { method: 'POST' }); } catch(e) {}
      const poll = setInterval(() => {
        fetch('/').then(r => { if (r.ok) { clearInterval(poll); location.reload(); } }).catch(() => {});
      }, 1000);
    }
  </script>
  <div id="page-controls">
    <button class="page-btn" onclick="hardRefresh()">Refresh</button>
    <button class="page-btn" id="restart-btn" onclick="restartServer()">Restart</button>
  </div>
</body>
</html>
"""


ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>baleef — Admin</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0f0f0f; color: #e0e0e0; font-family: 'Segoe UI', monospace, sans-serif; font-size: 13px; }
    header {
      position: sticky; top: 0; background: #0f0f0f;
      border-bottom: 1px solid #222; padding: 14px 24px;
      display: flex; align-items: center; gap: 16px; z-index: 10;
    }
    header h1 { font-size: 14px; font-weight: 600; letter-spacing: 2px; color: #fff; text-transform: uppercase; }
    .badge { font-size: 11px; padding: 2px 8px; border-radius: 4px; background: #1a1a1a; color: #555; }
    .badge.live { background: #052e16; color: #22c55e; }
    .controls { margin-left: auto; display: flex; gap: 8px; }
    button {
      background: #1a1a1a; color: #888; border: 1px solid #2a2a2a;
      padding: 5px 12px; border-radius: 6px; font-size: 12px; cursor: pointer;
    }
    button:hover { background: #252525; color: #ccc; }
    #log { padding: 12px 0; }
    .row {
      display: grid;
      grid-template-columns: 52px 26px 90px 90px 52px 48px 1fr;
      gap: 0 12px;
      align-items: baseline;
      padding: 7px 24px;
      border-bottom: 1px solid #161616;
      transition: background 0.15s;
    }
    .row:hover { background: #161616; }
    .row.translation { }
    .row.connection { color: #444; }
    .row.lang-change { color: #555; }
    .ts { color: #333; font-variant-numeric: tabular-nums; }
    .side-a { color: #60a5fa; font-weight: 600; }
    .side-b { color: #f472b6; font-weight: 600; }
    .src { color: #888; }
    .tgt { color: #888; }
    .ms { color: #555; font-variant-numeric: tabular-nums; text-align: right; }
    .cache { color: #22c55e; font-size: 11px; }
    .original { color: #555; font-style: italic; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .translated { color: #e0e0e0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .full-span { grid-column: 3 / -1; }
    #stats {
      position: sticky; bottom: 0; background: #0a0a0a;
      border-top: 1px solid #1e1e1e; padding: 8px 24px;
      display: flex; gap: 24px; font-size: 12px; color: #444;
    }
    #stats span { color: #666; }
    #stats b { color: #888; }
    #upload-bar {
      background: #0d0d0d; border-bottom: 1px solid #1e1e1e;
      padding: 10px 24px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    }
    #upload-bar label { font-size: 11px; color: #555; text-transform: uppercase; letter-spacing: 1px; }
    #upload-side {
      background: #1a1a1a; color: #aaa; border: 1px solid #2a2a2a;
      padding: 4px 8px; border-radius: 5px; font-size: 12px; cursor: pointer;
    }
    #upload-file {
      font-size: 12px; color: #777;
      background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 5px;
      padding: 4px 8px; cursor: pointer; flex: 1; min-width: 0;
    }
    #upload-file::file-selector-button {
      background: #252525; color: #888; border: none; border-radius: 4px;
      padding: 3px 8px; font-size: 11px; cursor: pointer; margin-right: 8px;
    }
    #upload-btn {
      background: #1a3a2a; color: #22c55e; border: 1px solid #1e5c3a;
      padding: 5px 14px; border-radius: 6px; font-size: 12px; cursor: pointer; white-space: nowrap;
    }
    #upload-btn:hover { background: #1e4a33; }
    #upload-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    #upload-status { font-size: 12px; color: #555; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #upload-status.ok { color: #22c55e; }
    #upload-status.err { color: #f87171; }

    /* ── Display settings ── */
    #display-bar {
      background: #080808; border-bottom: 1px solid #1a1a1a;
      padding: 8px 24px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
    }
    .bar-label { font-size: 10px; color: #3a3a3a; text-transform: uppercase; letter-spacing: 1px; white-space: nowrap; }
    .ctrl-grp { display: flex; align-items: center; gap: 5px; }
    .ctrl-grp > label { font-size: 11px; color: #4a4a4a; white-space: nowrap; }
    .ctrl-sep { width: 1px; height: 18px; background: #1e1e1e; margin: 0 2px; }
    input[type="color"] {
      width: 28px; height: 22px; border: 1px solid #2a2a2a; border-radius: 3px;
      cursor: pointer; background: none; padding: 1px 2px;
    }
    input[type="range"]#cfg-size { width: 72px; accent-color: #22c55e; cursor: pointer; }
    #cfg-size-val { font-size: 11px; color: #555; min-width: 30px; }
    input[type="number"]#cfg-phrases {
      width: 44px; background: #1a1a1a; color: #aaa; border: 1px solid #2a2a2a;
      border-radius: 4px; padding: 3px 5px; font-size: 12px; text-align: center;
    }
    #cfg-font {
      background: #1a1a1a; color: #aaa; border: 1px solid #2a2a2a;
      border-radius: 4px; padding: 3px 8px; font-size: 12px; cursor: pointer;
    }
    #font-upload-btn {
      background: #1a1a1a; color: #666; border: 1px solid #2a2a2a;
      padding: 3px 10px; border-radius: 4px; font-size: 11px; cursor: pointer;
    }
    #font-upload-btn:hover { color: #aaa; }
    #font-status { font-size: 11px; color: #22c55e; }

    /* ── Glossary ── */
    #glossary-bar {
      background: #070707; border-bottom: 1px solid #161616;
      padding: 8px 24px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
    }
    #gls-src, #gls-tgt {
      background: #1a1a1a; color: #aaa; border: 1px solid #2a2a2a;
      border-radius: 4px; padding: 3px 8px; font-size: 12px; cursor: pointer;
    }
    #gls-import-btn {
      background: #1a1a1a; color: #666; border: 1px solid #2a2a2a;
      padding: 4px 12px; border-radius: 5px; font-size: 12px; cursor: pointer;
    }
    #gls-import-btn:hover { color: #aaa; }
    #gls-status { font-size: 12px; }
    #gls-status.ok { color: #22c55e; }
    #gls-status.err { color: #f87171; }
    .gls-hint { font-size: 10px; color: #2a2a2a; font-style: italic; }
  </style>
</head>
<body>
  <header>
    <h1>baleef</h1>
    <div class="badge" id="status">connecting…</div>
    <div class="controls">
      <button onclick="clearLog()">Effacer</button>
      <button onclick="togglePause()" id="pauseBtn">Pause</button>
      <button onclick="hardRefresh()">Refresh</button>
      <button id="admin-restart-btn" onclick="restartServer()">Restart</button>
    </div>
  </header>
  <div id="display-bar">
    <span class="bar-label">Affichage</span>
    <div class="ctrl-grp">
      <label>Fond</label>
      <input type="color" id="cfg-bg" value="#000000" oninput="pushConfig('bg_color', this.value)">
    </div>
    <div class="ctrl-grp">
      <label>Texte</label>
      <input type="color" id="cfg-text" value="#ffffff" oninput="pushConfig('text_color', this.value)">
    </div>
    <div class="ctrl-sep"></div>
    <div class="ctrl-grp">
      <label>Taille</label>
      <input type="range" id="cfg-size" min="10" max="120" value="32"
             oninput="document.getElementById('cfg-size-val').textContent=this.value+'px'; pushConfig('font_size', +this.value)">
      <span id="cfg-size-val">32px</span>
    </div>
    <div class="ctrl-grp">
      <label>Phrases</label>
      <input type="number" id="cfg-phrases" min="1" max="12" value="4"
             onchange="pushConfig('max_phrases', +this.value)">
    </div>
    <div class="ctrl-sep"></div>
    <div class="ctrl-grp">
      <label>Police</label>
      <select id="cfg-font" onchange="pushConfig('font_family', this.value)">
        <option value="'Segoe UI', system-ui, sans-serif">Segoe UI</option>
        <option value="system-ui, sans-serif">System UI</option>
        <option value="Georgia, serif">Georgia</option>
        <option value="'Courier New', monospace">Courier New</option>
        <option value="Arial, sans-serif">Arial</option>
        <option value="'Times New Roman', serif">Times New Roman</option>
      </select>
      <input type="file" id="font-file" accept=".ttf,.otf,.woff,.woff2" style="display:none" onchange="uploadFont(this)">
      <button id="font-upload-btn" onclick="document.getElementById('font-file').click()">+ Police</button>
      <span id="font-status"></span>
    </div>
  </div>

  <div id="glossary-bar">
    <span class="bar-label">Glossaire</span>
    <div class="ctrl-grp">
      <label>Source</label>
      <select id="gls-src">
        <option value="fr">Français</option>
        <option value="en">English</option>
        <option value="es">Español</option>
        <option value="de">Deutsch</option>
        <option value="ar">العربية</option>
        <option value="zh">中文</option>
        <option value="ja">日本語</option>
        <option value="pt">Português</option>
        <option value="ru">Русский</option>
        <option value="it">Italiano</option>
        <option value="hi">हिन्दी</option>
      </select>
    </div>
    <div class="ctrl-grp">
      <label>Traduction vers</label>
      <select id="gls-tgt">
        <option value="eng_Latn">English</option>
        <option value="fra_Latn">Français</option>
        <option value="spa_Latn">Español</option>
        <option value="deu_Latn">Deutsch</option>
        <option value="arb_Arab">العربية</option>
        <option value="zho_Hans">中文</option>
        <option value="jpn_Jpan">日本語</option>
        <option value="por_Latn">Português</option>
        <option value="rus_Cyrl">Русский</option>
        <option value="ita_Latn">Italiano</option>
        <option value="hin_Deva">हिन्दी</option>
      </select>
    </div>
    <input type="file" id="gls-file" accept=".csv,.tsv,.txt" style="display:none" onchange="importGlossary(this)">
    <button id="gls-import-btn" onclick="document.getElementById('gls-file').click()">Importer CSV / TSV</button>
    <span id="gls-status"></span>
    <span class="gls-hint">2 col : source, traduction — 4 col : source, src_lang, tgt_lang, traduction — # = commentaire</span>
  </div>

  <div id="upload-bar">
    <label>Test audio</label>
    <select id="upload-side">
      <option value="A">Side A (source : français)</option>
      <option value="B">Side B (source : auto)</option>
    </select>
    <input type="file" id="upload-file" accept=".wav,.flac,.ogg,.opus">
    <button id="upload-btn" onclick="uploadAudio()">Envoyer</button>
    <span id="upload-status"></span>
  </div>
  <div id="log"></div>
  <div id="stats">
    <span>Traductions : <b id="st-total">0</b></span>
    <span>Cache hits : <b id="st-cache">0</b></span>
    <span>Side A : <b id="st-a">0</b></span>
    <span>Side B : <b id="st-b">0</b></span>
    <span>Latence moy. : <b id="st-avg">—</b></span>
  </div>
  <script>
    let paused = false, total = 0, cacheHits = 0, sideA = 0, sideB = 0, totalMs = 0;

    function togglePause() {
      paused = !paused;
      document.getElementById('pauseBtn').textContent = paused ? 'Reprendre' : 'Pause';
    }
    function clearLog() {
      document.getElementById('log').innerHTML = '';
      total = cacheHits = sideA = sideB = totalMs = 0;
      updateStats();
    }
    function esc(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
    function updateStats() {
      document.getElementById('st-total').textContent = total;
      document.getElementById('st-cache').textContent = cacheHits + (total ? ' (' + Math.round(cacheHits/total*100) + '%)' : '');
      document.getElementById('st-a').textContent = sideA;
      document.getElementById('st-b').textContent = sideB;
      document.getElementById('st-avg').textContent = total ? Math.round(totalMs/total) + ' ms' : '—';
    }
    function addRow(e) {
      if (paused) return;
      const log = document.getElementById('log');
      const row = document.createElement('div');

      if (e.kind === 'translation') {
        total++; totalMs += e.ms;
        if (e.cached) cacheHits++;
        if (e.side === 'A') sideA++; else sideB++;
        updateStats();
        row.className = 'row translation';
        row.innerHTML =
          '<span class="ts">' + esc(e.ts) + '</span>' +
          '<span class="side-' + e.side.toLowerCase() + '">' + esc(e.side) + '</span>' +
          '<span class="src">' + esc(e.src_lang) + '</span>' +
          '<span class="tgt">' + esc(e.tgt_lang) + '</span>' +
          '<span class="ms">' + e.ms + ' ms</span>' +
          '<span class="cache">' + (e.cached ? '⚡cache' : '') + '</span>' +
          '<span class="translated">' + esc(e.translated) + ' <span class="original">← ' + esc(e.original) + '</span></span>';
      } else {
        row.className = 'row ' + (e.kind || 'info');
        row.innerHTML =
          '<span class="ts">' + esc(e.ts) + '</span>' +
          '<span></span>' +
          '<span class="full-span" style="color:#333">' + esc(JSON.stringify(e)) + '</span>';
      }

      log.appendChild(row);
      log.lastElementChild.scrollIntoView({ behavior: 'smooth', block: 'end' });

      // Keep max 500 rows in DOM
      while (log.children.length > 500) log.removeChild(log.firstChild);
    }

    async function importGlossary(input) {
      const file = input.files[0];
      if (!file) return;
      const status = document.getElementById('gls-status');
      const src = document.getElementById('gls-src').value;
      const tgt = document.getElementById('gls-tgt').value;
      status.textContent = 'Import…'; status.className = '';
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch('/upload/glossary?src_lang=' + src + '&tgt_lang=' + tgt, { method: 'POST', body: fd });
      const d = await r.json();
      if (d.added !== undefined) {
        let msg = '✓ ' + d.added + ' entrée' + (d.added > 1 ? 's' : '') + ' ajoutée' + (d.added > 1 ? 's' : '');
        if (d.skipped) msg += ' · ' + d.skipped + ' ignorée' + (d.skipped > 1 ? 's' : '');
        if (d.sample && d.sample.length)
          msg += ' — ex : "' + esc(d.sample[0].source) + '" → "' + esc(d.sample[0].translation) + '"';
        status.textContent = msg; status.className = 'ok';
      } else {
        status.textContent = d.detail || 'Erreur'; status.className = 'err';
      }
      input.value = '';
    }

    async function pushConfig(key, value) {
      await fetch('/config/display', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [key]: value })
      });
    }

    async function uploadFont(input) {
      const file = input.files[0];
      if (!file) return;
      const status = document.getElementById('font-status');
      status.textContent = '…';
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch('/upload/font', { method: 'POST', body: fd });
      const d = await r.json();
      const sel = document.getElementById('cfg-font');
      const opt = document.createElement('option');
      const ff = "'" + d.font_name + "', sans-serif";
      opt.value = ff; opt.textContent = d.font_name + ' ✓'; opt.selected = true;
      sel.appendChild(opt);
      pushConfig('font_family', ff);
      status.textContent = d.font_name + ' chargée';
      setTimeout(() => status.textContent = '', 3000);
      input.value = '';
    }

    // Init display controls from server
    fetch('/config/display').then(r => r.json()).then(cfg => {
      document.getElementById('cfg-bg').value = cfg.bg_color;
      document.getElementById('cfg-text').value = cfg.text_color;
      document.getElementById('cfg-size').value = cfg.font_size;
      document.getElementById('cfg-size-val').textContent = cfg.font_size + 'px';
      document.getElementById('cfg-phrases').value = cfg.max_phrases;
      const sel = document.getElementById('cfg-font');
      (cfg.custom_fonts || []).forEach(fn => {
        const nm = fn.replace(/\\.[^.]+$/, '').replace(/_/g, ' ');
        const opt = document.createElement('option');
        opt.value = "'" + nm + "', sans-serif"; opt.textContent = nm + ' ✓';
        sel.appendChild(opt);
      });
      for (const opt of sel.options) {
        if (opt.value === cfg.font_family) { opt.selected = true; break; }
      }
    });

    async function uploadAudio() {
      const side = document.getElementById('upload-side').value;
      const fileInput = document.getElementById('upload-file');
      const status = document.getElementById('upload-status');
      const btn = document.getElementById('upload-btn');
      if (!fileInput.files.length) { status.textContent = 'Choisir un fichier'; status.className = 'err'; return; }
      btn.disabled = true;
      status.textContent = 'Traitement…';
      status.className = '';
      const fd = new FormData();
      fd.append('file', fileInput.files[0]);
      try {
        const r = await fetch('/upload/' + side, { method: 'POST', body: fd });
        const d = await r.json();
        if (!r.ok || d.error) {
          status.textContent = d.detail || d.error || 'Erreur';
          status.className = 'err';
        } else {
          status.textContent = '✓ "' + d.original + '" → "' + d.translated + '" (' + d.ms + ' ms)';
          status.className = 'ok';
        }
      } catch(e) {
        status.textContent = 'Erreur réseau : ' + e.message;
        status.className = 'err';
      } finally {
        btn.disabled = false;
      }
    }

    function hardRefresh() {
      location.replace(location.pathname + '?_=' + Date.now());
    }

    async function restartServer() {
      if (!confirm('Redémarrer le serveur ?')) return;
      const btn = document.getElementById('admin-restart-btn');
      btn.textContent = '…'; btn.disabled = true;
      try { await fetch('/restart', { method: 'POST' }); } catch(e) {}
      const poll = setInterval(() => {
        fetch('/').then(r => { if (r.ok) { clearInterval(poll); location.reload(); } }).catch(() => {});
      }, 1000);
    }

    function connect() {
      const ws = new WebSocket('ws://' + location.host + '/ws/admin');
      const status = document.getElementById('status');
      ws.onopen = () => { status.textContent = 'live'; status.className = 'badge live'; };
      ws.onmessage = e => addRow(JSON.parse(e.data));
      ws.onclose = () => {
        status.textContent = 'reconnecting…'; status.className = 'badge';
        setTimeout(connect, 1500);
      };
    }
    connect();
  </script>
</body>
</html>"""


def make_single_side_html(side: str) -> str:
    lang_ctrl = (
        f"""<select class="lang-select" onchange="setLang('{side}', this.value)">
        {"".join(f'<option value="{c}"{" selected" if c == TARGET_LANG[side] else ""}>{n}</option>' for c, n in LANG_NAMES.items())}
      </select>"""
        if side == "A"
        else '<span class="lang-fixed">Français</span>'
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>baleef — Side {side}</title>
  <style>
    :root {{
      --bg: #000000;
      --text: #ffffff;
      --font-size: 32px;
      --font-family: 'Segoe UI', system-ui, sans-serif;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg); color: var(--text);
      font-family: var(--font-family);
      display: flex; flex-direction: column;
      height: 100vh; overflow: hidden;
      padding: 28px 36px; gap: 16px;
    }}
    .topbar {{
      display: flex; align-items: center;
      justify-content: space-between; flex-shrink: 0;
    }}
    .side-label {{
      font-size: 10px; letter-spacing: 3px;
      text-transform: uppercase; color: #383838;
    }}
    .lang-select {{
      background: #111; color: #aaa; border: 1px solid #252525;
      border-radius: 8px; padding: 7px 32px 7px 12px; font-size: 14px;
      cursor: pointer; outline: none; appearance: none;
      background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%23555' d='M5 6L0 0h10z'/%3E%3C/svg%3E");
      background-repeat: no-repeat; background-position: right 10px center;
    }}
    .lang-fixed {{ font-size: 14px; color: #555; padding: 7px 12px; }}
    .dot {{
      width: 8px; height: 8px; border-radius: 50%;
      background: #1e1e1e; transition: background 0.3s; flex-shrink: 0;
    }}
    .dot.active {{ background: #22c55e; }}
    .feed {{
      flex: 1; display: flex; flex-direction: column;
      justify-content: flex-end; overflow: hidden;
    }}
    .feed-item {{
      padding: 14px 0; border-top: 1px solid #141414;
      animation: slideUp 0.3s ease;
    }}
    .feed-item:first-child {{ border-top: none; }}
    .feed-item {{ opacity: 0.15; }}
    .feed-item:nth-last-child(4) {{ opacity: 0.2; }}
    .feed-item:nth-last-child(3) {{ opacity: 0.4; }}
    .feed-item:nth-last-child(2) {{ opacity: 0.7; }}
    .feed-item:nth-last-child(1) {{ opacity: 1; }}
    .feed-item .f-translated {{
      font-size: var(--font-size); font-weight: 300;
      line-height: 1.35; word-break: break-word;
    }}
    .feed-item .f-original {{ font-size: 13px; color: #555; font-style: italic; margin-top: 4px; }}
    .feed-item .f-meta {{ font-size: 11px; color: #282828; margin-top: 3px; }}
    @keyframes slideUp {{
      from {{ opacity: 0; transform: translateY(12px); }}
      to   {{ transform: translateY(0); }}
    }}
    #page-controls {{
      position: fixed; bottom: 14px; right: 14px;
      display: flex; gap: 6px; opacity: 0.08; transition: opacity 0.25s; z-index: 99;
    }}
    #page-controls:hover {{ opacity: 1; }}
    .page-btn {{
      background: #111; color: #777; border: 1px solid #2a2a2a;
      border-radius: 5px; padding: 5px 11px; font-size: 11px;
      letter-spacing: 0.5px; cursor: pointer;
    }}
    .page-btn:hover {{ color: #ddd; border-color: #444; }}
    .page-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="side-label">Side {side}</div>
    {lang_ctrl}
    <div class="dot" id="dot{side}"></div>
  </div>
  <div class="feed" id="feed{side}"></div>
  <script>
    let MAX = 4;
    const SPEAKER_COLORS = ['#60a5fa','#f472b6','#4ade80','#fb923c','#c084fc','#fbbf24','#2dd4bf','#f87171'];

    function applyConfig(cfg) {{
      const r = document.documentElement.style;
      if (cfg.bg_color    !== undefined) r.setProperty('--bg',          cfg.bg_color);
      if (cfg.text_color  !== undefined) r.setProperty('--text',        cfg.text_color);
      if (cfg.font_size   !== undefined) r.setProperty('--font-size',   cfg.font_size + 'px');
      if (cfg.font_family !== undefined) r.setProperty('--font-family', cfg.font_family);
      if (cfg.max_phrases !== undefined) {{
        MAX = cfg.max_phrases;
        const f = document.getElementById('feed{side}');
        if (f) while (f.children.length > MAX) f.removeChild(f.firstChild);
      }}
      if (cfg.custom_fonts) cfg.custom_fonts.forEach(fn => {{
        if (document.querySelector('[data-font="' + fn + '"]')) return;
        const nm = fn.replace(/\\.[^.]+$/, '').replace(/_/g, ' ');
        const st = document.createElement('style');
        st.setAttribute('data-font', fn);
        st.textContent = "@font-face{{font-family:'" + nm + "';src:url('/fonts/" + fn + "')}}";
        document.head.appendChild(st);
      }});
    }}

    function connectConfig() {{
      const ws = new WebSocket('ws://' + location.host + '/ws/config');
      ws.onmessage = e => applyConfig(JSON.parse(e.data));
      ws.onclose = () => setTimeout(connectConfig, 2000);
    }}

    function setLang(side, code) {{
      fetch('/lang/' + side + '?code=' + code, {{ method: 'POST' }});
    }}
    function esc(s) {{
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }}
    function addItem(d) {{
      const feed = document.getElementById('feed{side}');
      const item = document.createElement('div');
      item.className = 'feed-item';
      const color = SPEAKER_COLORS[(d.speaker_id || 0) % SPEAKER_COLORS.length];
      item.innerHTML =
        '<div class="f-translated" style="color:' + color + '">' + esc(d.translated) + '</div>' +
        '<div class="f-original">' + esc(d.original) + '</div>' +
        '<div class="f-meta">' + esc(d.src_lang) + ' → ' + esc(d.tgt_lang) + ' · ' + d.ms + 'ms</div>';
      feed.appendChild(item);
      while (feed.children.length > MAX) feed.removeChild(feed.firstChild);
    }}
    function connect() {{
      const dot = document.getElementById('dot{side}');
      const ws = new WebSocket('ws://' + location.host + '/ws/{side}');
      ws.onopen = () => {{
        document.getElementById('feed{side}').innerHTML = '';
      }};
      ws.onmessage = e => {{
        addItem(JSON.parse(e.data));
        dot.classList.add('active');
        setTimeout(() => dot.classList.remove('active'), 600);
      }};
      ws.onclose = () => setTimeout(connect, 1500);
    }}
    connect();
    connectConfig();

    function hardRefresh() {{
      location.replace(location.pathname + '?_=' + Date.now());
    }}

    async function restartServer() {{
      if (!confirm('Redémarrer le serveur ?')) return;
      const btn = document.getElementById('restart-btn');
      btn.textContent = '…'; btn.disabled = true;
      try {{ await fetch('/restart', {{ method: 'POST' }}); }} catch(e) {{}}
      const poll = setInterval(() => {{
        fetch('/').then(r => {{ if (r.ok) {{ clearInterval(poll); location.reload(); }} }}).catch(() => {{}});
      }}, 1000);
    }}
  </script>
  <div id="page-controls">
    <button class="page-btn" onclick="hardRefresh()">Refresh</button>
    <button class="page-btn" id="restart-btn" onclick="restartServer()">Restart</button>
  </div>
</body>
</html>"""
