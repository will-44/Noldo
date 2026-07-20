// app.js — orchestration : rail (conversations/bibliothèque), chat central (SSE), PDF escamotable.
import { loadPdf, highlightPassage, nextPage, prevPage } from "/static/pdfviewer.js";
import { iconHTML, populateIcons } from "/static/icons.js";
import { marked } from "/static/vendor/marked.esm.js";

marked.setOptions({ gfm: true, breaks: true });

const GREETING_HTML = `<div class="msg assistant"><div class="bubble">Bonjour ! Posez une question sur votre bibliothèque, ou ouvrez un article pour l'interroger spécifiquement.</div></div>`;

let allDocs = [];
let conversations = [];
let activeConversationId = null; // null = conversation pas encore créée côté serveur (1er envoi à venir)
let viewerItemKey = null;        // article actuellement affiché dans le panneau PDF (null = fermé)
let scopeLocked = false;         // recherche limitée à viewerItemKey — n'a de sens que si ouvert
let isGenerating = false;
let currentAbortController = null;

const el = (id) => document.getElementById(id);
const isMobile = () => window.matchMedia("(max-width: 900px)").matches;

// ── Init ────────────────────────────────────────────────────────────────────
async function init() {
  populateIcons();

  try { allDocs = await fetch("/api/docs").then((r) => r.json()); } catch { allDocs = []; }
  el("doc-count").textContent =
    `${allDocs.length} article${allDocs.length > 1 ? "s" : ""} indexé${allDocs.length > 1 ? "s" : ""}`;
  renderList(allDocs);
  updateFilterHint();

  await loadConversations();

  el("search-box").addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase();
    renderList(allDocs.filter((d) =>
      d.title.toLowerCase().includes(q) ||
      (d.authors || "").toLowerCase().includes(q) ||
      String(d.year || "").includes(q)
    ));
  });

  el("send-btn").addEventListener("click", () => (isGenerating ? stopGeneration() : sendQuestion()));
  el("question-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !isGenerating) { e.preventDefault(); sendQuestion(); }
  });
  el("prev-page").addEventListener("click", () => prevPage());
  el("next-page").addEventListener("click", () => nextPage());

  el("scope-lock").addEventListener("change", (e) => {
    scopeLocked = e.target.checked;
    updateFilterHint();
    renderList(filteredDocs());
  });

  el("new-conversation-btn").addEventListener("click", newConversation);
  el("close-viewer-btn").addEventListener("click", closeViewer);
  el("export-conversation-btn").addEventListener("click", exportConversation);
  el("sync-btn").addEventListener("click", startSync);
  // Reprend l'affichage d'une sync déjà en cours (lancée avant de fermer l'onglet, ou depuis
  // un autre appareil) — no-op silencieux si aucun job n'est en cours ou déjà terminé.
  pollSyncStatus();

  el("rail-tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".rail-tab");
    if (btn) switchRailTab(btn.dataset.tab);
  });

  initDrawers();
}

function switchRailTab(tabId) {
  document.querySelectorAll(".rail-tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === tabId));
  document.querySelectorAll(".rail-view").forEach((v) => { v.hidden = v.id !== tabId; });
}

// ── Tiroirs responsive (rail/viewer en overlay sous 900px) ───────────────────
function initDrawers() {
  el("open-rail-btn").addEventListener("click", () => openDrawer("left-rail"));
  el("close-rail-btn").addEventListener("click", closeDrawers);
  el("open-viewer-btn").addEventListener("click", () => openDrawer("viewer-panel"));
  el("backdrop").addEventListener("click", closeDrawers);
}
function openDrawer(panelId) {
  if (isMobile()) {
    el("left-rail").classList.remove("open");
    el("viewer-panel").classList.remove("open");
    el("backdrop").classList.add("open");
  }
  el(panelId).classList.add("open");
}
function closeDrawers() {
  el("left-rail").classList.remove("open");
  if (isMobile()) el("viewer-panel").classList.remove("open");
  el("backdrop").classList.remove("open");
}

// ── Portée de recherche (rattachée à l'article ouvert dans le viewer) ───────
function updateFilterHint() {
  el("filter-hint").innerHTML = scopeLocked
    ? `${iconHTML("search")} Recherche limitée à l'article ouvert`
    : `${iconHTML("globe")} Recherche dans toute la bibliothèque`;
}

// ── Viewer PDF (panneau droit, escamotable) ──────────────────────────────────
async function openViewer(itemKey) {
  viewerItemKey = itemKey;
  openDrawer("viewer-panel");
  el("scope-toggle").hidden = false;
  el("open-viewer-btn").hidden = false;
  renderList(filteredDocs());
}

function closeViewer() {
  viewerItemKey = null;
  scopeLocked = false;
  el("scope-lock").checked = false;
  el("viewer-panel").classList.remove("open");
  el("backdrop").classList.remove("open");
  el("scope-toggle").hidden = true;
  el("open-viewer-btn").hidden = true;
  updateFilterHint();
  renderList(filteredDocs());
}

async function selectLibraryDoc(doc) {
  await openViewer(doc.item_key);
  el("viewer-title").textContent = doc.title;
  if (doc.has_pdf) {
    try { await loadPdf(`/api/pdf/${doc.item_key}`); }
    catch { el("viewer-title").textContent = "Erreur de chargement du PDF"; }
  }
}

async function openSourceCard(src) {
  if (src.item_key !== viewerItemKey) {
    const doc = allDocs.find((d) => d.item_key === src.item_key);
    if (doc) await selectLibraryDoc(doc);
  }
  await highlightPassage(src.page, src.text);
}

// ── Bibliothèque (rail, onglet) ───────────────────────────────────────────
function renderList(docs) {
  const list = el("doc-list");
  if (!docs.length) {
    list.innerHTML = `<div style="padding:20px;text-align:center;font-size:12px;color:#585b70">Aucun résultat</div>`;
    return;
  }
  list.innerHTML = "";
  for (const d of docs) {
    const isActive = d.item_key === viewerItemKey;
    const item = document.createElement("div");
    item.className = "doc-item" + (d.has_pdf ? "" : " no-pdf") + (isActive ? " active" : "");
    item.title = d.has_pdf ? "" : "PDF non disponible localement";
    const badge = (isActive && scopeLocked)
      ? `<span class="lock-badge" title="Recherche limitée à cet article">${iconHTML("lock")}</span>`
      : "";
    item.innerHTML =
      `<div class="doc-title"><span class="title-text">${esc(d.title)}</span>${badge}</div>` +
      `<div class="doc-meta">${esc(fmtAuthors(d.authors))}${d.year ? " · " + d.year : ""}</div>`;
    item.addEventListener("click", () => selectLibraryDoc(d));
    list.appendChild(item);
  }
}

function filteredDocs() {
  const q = el("search-box").value.toLowerCase();
  if (!q) return allDocs;
  return allDocs.filter((d) =>
    d.title.toLowerCase().includes(q) || (d.authors || "").toLowerCase().includes(q));
}

// ── Conversations (rail, onglet par défaut) ──────────────────────────────
async function loadConversations() {
  try { conversations = await fetch("/api/conversations").then((r) => r.json()); }
  catch { conversations = []; }
  renderConvList();
}

function renderConvList() {
  const list = el("conv-list");
  if (!conversations.length) {
    list.innerHTML = `<div id="conv-list-empty">Aucune conversation pour l'instant</div>`;
    return;
  }
  list.innerHTML = "";
  for (const c of conversations) {
    const item = document.createElement("div");
    item.className = "conv-item" + (c.id === activeConversationId ? " active" : "");
    item.innerHTML =
      `<span class="conv-title">${esc(c.title)}</span>` +
      `<span class="conv-actions">` +
      `<button class="icon-btn rename-btn" title="Renommer">${iconHTML("pencil")}</button>` +
      `<button class="icon-btn delete-btn" title="Supprimer">${iconHTML("trash-2")}</button>` +
      `</span>`;
    item.querySelector(".conv-title").addEventListener("click", () => switchConversation(c.id));
    item.querySelector(".rename-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      startRenameConversation(item, c);
    });
    item.querySelector(".delete-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteConversation(c.id);
    });
    list.appendChild(item);
  }
}

function startRenameConversation(item, conv) {
  const titleEl = item.querySelector(".conv-title");
  const input = document.createElement("input");
  input.className = "conv-title-input";
  input.value = conv.title;
  titleEl.replaceWith(input);
  input.focus();
  input.select();

  const commit = async () => {
    const newTitle = input.value.trim() || conv.title;
    if (newTitle !== conv.title) {
      await fetch(`/api/conversations/${conv.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: newTitle }),
      });
      conv.title = newTitle;
      if (conv.id === activeConversationId) el("chat-title").textContent = newTitle;
    }
    renderConvList();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") input.blur();
    if (e.key === "Escape") { input.value = conv.title; input.blur(); }
  });
  input.addEventListener("blur", commit);
}

async function deleteConversation(id) {
  if (!confirm("Supprimer cette conversation ?")) return;
  await fetch(`/api/conversations/${id}`, { method: "DELETE" });
  conversations = conversations.filter((c) => c.id !== id);
  if (activeConversationId === id) newConversation();
  renderConvList();
}

function newConversation() {
  activeConversationId = null;
  el("chat-title").textContent = "Nouvelle conversation";
  el("messages").innerHTML = GREETING_HTML;
  el("export-conversation-btn").hidden = true;
  renderConvList();
  closeDrawers();
}

async function switchConversation(id) {
  let conv;
  try { conv = await fetch(`/api/conversations/${id}`).then((r) => r.json()); }
  catch { return; }

  activeConversationId = id;
  el("chat-title").textContent = conv.title;
  el("export-conversation-btn").hidden = false;
  renderConvList();
  closeDrawers();

  const msgs = el("messages");
  msgs.innerHTML = "";
  for (const m of conv.messages) {
    if (m.role === "user") addMessage("user", m.content);
    else if (m.role === "assistant") renderReplayedAssistantMessage(m.content, m.meta || {});
  }
  msgs.scrollTop = msgs.scrollHeight;
}

function renderReplayedAssistantMessage(content, meta) {
  const div = document.createElement("div");
  div.className = "msg assistant";
  div.innerHTML =
    `<div class="bubble markdown-body">${renderMarkdown(content)}</div>` +
    `<div class="agent-sources"></div>` +
    buildMsgActionsHTML(false) +
    `<span class="msg-time"></span>`;
  el("messages").appendChild(div);

  const corpusSources = dedupeSources(meta.sources || []);
  renderSourcesInto(div.querySelector(".agent-sources"), corpusSources, meta.external_sources || [], true);
  wireMsgActions(div, content, corpusSources);
}

// ── Chat : envoi + streaming SSE ─────────────────────────────────────────
async function sendQuestion() {
  const input = el("question-input");
  const q = input.value.trim();
  if (!q || isGenerating) return;
  input.value = "";

  addMessage("user", q);
  const itemKeyAtSend = (scopeLocked && viewerItemKey) ? viewerItemKey : null;
  const ui = addAgentMessage();
  const state = {
    answer: "", thinkingEl: null, thinkingText: "",
    corpusSources: [], externalSources: [], wasGlobalScope: !itemKeyAtSend,
  };

  currentAbortController = new AbortController();
  setGenerating(true);

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        conversation_id: activeConversationId,
        item_key: itemKeyAtSend,
        scope_locked: scopeLocked,
        selected_item_key: viewerItemKey,
      }),
      signal: currentAbortController.signal,
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    await consumeEventStream(res.body, ui, state);
  } catch (e) {
    if (e.name === "AbortError") renderAgentStopped(ui, state);
    else renderAgentError(ui, e.message);
  }

  finalizeAgentMessage(ui);
  wireMsgActions(ui.root, state.answer, dedupeSources(state.corpusSources));
  setGenerating(false);
  currentAbortController = null;
  input.focus();
}

function stopGeneration() {
  if (currentAbortController) currentAbortController.abort();
}

function setGenerating(active) {
  isGenerating = active;
  const btn = el("send-btn");
  btn.classList.toggle("stop-mode", active);
  btn.innerHTML = iconHTML(active ? "square" : "send");
  btn.title = active ? "Arrêter la génération" : "Envoyer";
}

// Lit le flux SSE et distribue chaque événement JSON à handleAgentEvent(). Les trames SSE sont
// séparées par une ligne vide ; on bufferise le fragment final potentiellement incomplet d'une
// lecture à l'autre (rien ne garantit qu'un chunk réseau s'arrête pile sur une frontière de trame).
async function consumeEventStream(body, ui, state) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const msgs = el("messages");
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const frames = buf.split("\n\n");
    buf = frames.pop();
    for (const frame of frames) {
      const dataLine = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!dataLine) continue;
      handleAgentEvent(JSON.parse(dataLine.slice(6)), ui, state);
    }
    msgs.scrollTop = msgs.scrollHeight;
  }
}

function handleAgentEvent(event, ui, state) {
  switch (event.type) {
    case "conversation": {
      activeConversationId = event.id;
      conversations.unshift({ id: event.id, title: event.title, updated_at: new Date().toISOString() });
      renderConvList();
      el("export-conversation-btn").hidden = false;
      break;
    }
    case "title": {
      el("chat-title").textContent = event.title;
      const conv = conversations.find((c) => c.id === event.id);
      if (conv) conv.title = event.title;
      renderConvList();
      break;
    }
    case "thinking":
      appendThinking(ui, state, event.text);
      break;
    case "step":
      renderStep(ui, event);
      break;
    case "token":
      state.answer += event.text;
      showBubble(ui, state.answer);
      break;
    case "sources":
      if (event.kind === "corpus") state.corpusSources.push(...event.items);
      else state.externalSources.push(...event.items);
      break;
    case "done":
      state.answer = event.answer || state.answer;
      showBubble(ui, state.answer);
      renderSourcesInto(ui.sourcesSlot, dedupeSources(state.corpusSources), state.externalSources, state.wasGlobalScope);
      break;
    case "error":
      renderAgentError(ui, event.message);
      break;
  }
}

function addAgentMessage() {
  const msgs = el("messages");
  const div = document.createElement("div");
  div.className = "msg assistant";
  div.innerHTML =
    `<div class="agent-trace"></div>` +
    `<div class="thinking"><div class="dot"></div><div class="dot"></div><div class="dot"></div><span>Analyse en cours…</span></div>` +
    `<div class="bubble markdown-body" hidden></div>` +
    `<div class="agent-sources"></div>` +
    buildMsgActionsHTML(true) +
    `<span class="msg-time" hidden></span>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return {
    root: div,
    trace: div.querySelector(".agent-trace"),
    waiting: div.querySelector(".thinking"),
    bubble: div.querySelector(".bubble"),
    sourcesSlot: div.querySelector(".agent-sources"),
    time: div.querySelector(".msg-time"),
  };
}

function showBubble(ui, text) {
  ui.waiting.hidden = true;
  ui.bubble.hidden = false;
  ui.bubble.innerHTML = renderMarkdown(text);
}

function appendThinking(ui, state, text) {
  if (!state.thinkingEl) {
    const details = document.createElement("details");
    details.className = "agent-thinking";
    details.innerHTML = `<summary>${iconHTML("brain")} Réflexion</summary><div class="agent-thinking-body"></div>`;
    ui.trace.appendChild(details);
    state.thinkingEl = details.querySelector(".agent-thinking-body");
  }
  state.thinkingText += text;
  state.thinkingEl.textContent = state.thinkingText;
}

function renderStep(ui, event) {
  const div = document.createElement("div");
  div.className = "agent-step";
  if (event.tool === "search_corpus") {
    div.innerHTML = `${iconHTML("search")} Recherche dans le corpus : « ${esc(event.args.query || "")} »`;
  } else if (event.tool === "scan_corpus") {
    div.innerHTML = `${iconHTML("search")} Balayage exhaustif : « ${esc(event.args.keyword || "")} »`;
  } else if (event.tool === "get_external_citations") {
    div.innerHTML = `${iconHTML("globe")} Citations externes : « ${esc(event.args.title || "")} »`;
  } else {
    div.innerHTML = `${iconHTML("search")} ${esc(event.tool)}`;
  }
  ui.trace.appendChild(div);
}

function finalizeAgentMessage(ui) {
  ui.waiting.remove();
  ui.bubble.hidden = false;
  ui.time.hidden = false;
  ui.time.textContent = now();
  const actions = ui.root.querySelector(".msg-actions");
  if (actions) actions.hidden = false;
}

function renderAgentError(ui, message) {
  ui.waiting.hidden = true;
  ui.bubble.hidden = false;
  ui.bubble.innerHTML = renderMarkdown(`⚠️ Erreur : ${message}`);
}

function renderAgentStopped(ui, state) {
  ui.waiting.hidden = true;
  if (!state.answer) {
    ui.bubble.hidden = false;
    ui.bubble.innerHTML = renderMarkdown("_Génération interrompue._");
  }
  // Si du texte avait déjà été streamé, on le laisse tel quel (réponse partielle) plutôt que
  // de l'écraser — l'utilisateur voit ce qui a eu le temps d'être généré avant l'arrêt.
}

// Diagnostic de recherche : ce qui a été retrouvé dans le corpus, avant la synthèse du LLM.
function buildDiagnostics(sources, wasGlobalScope) {
  if (!sources.length) return "";
  const distinctDocs = new Set(sources.map((s) => s.item_key));
  const bestScore = Math.max(...sources.map((s) => s.score));
  const nDocs = distinctDocs.size;

  let warning = "";
  if (wasGlobalScope && nDocs === 1) {
    warning = `<div class="warn">${iconHTML("triangle-alert")} Tous les passages viennent du même article : la réponse ne reflète probablement pas le reste de la bibliothèque.</div>`;
  }

  return `<details class="diagnostics">
    <summary>${iconHTML("search")} ${sources.length} passage${sources.length > 1 ? "s" : ""} · ${nDocs} article${nDocs > 1 ? "s" : ""} distinct${nDocs > 1 ? "s" : ""}</summary>
    <div class="diagnostics-body">
      Portée : ${wasGlobalScope ? "toute la bibliothèque" : "cet article"}<br>
      Meilleur score de pertinence : ${bestScore.toFixed(5)}
      ${warning}
    </div>
  </details>`;
}

// Rendu des sources (corpus + externes), partagé entre les réponses en direct et rejouées
// depuis l'historique — garantit un rendu identique dans les deux cas.
function renderSourcesInto(slot, corpus, external, wasGlobalScope) {
  const parts = [];

  if (corpus.length) {
    parts.push(buildDiagnostics(corpus, wasGlobalScope));
    parts.push(`<div class="sources">` + corpus.map((s, i) => {
      const dim = viewerItemKey && s.item_key !== viewerItemKey;
      return `<div class="source-card${dim ? " other-doc" : ""}" data-idx="${i}">
         <div class="source-card-header">
           <span class="page-badge">p.${s.page}</span>
           <span class="source-meta">${esc(fmtAuthors(s.authors))}${s.year ? " · " + s.year : ""} — ${esc(truncate(s.title, 46))}</span>
         </div>
         <div class="source-text">${esc(truncate(s.text, 160))}</div>
       </div>`;
    }).join("") + `</div>`);
  }

  if (external && external.length) {
    parts.push(`<div class="sources external-sources">` + external.map((s) =>
      `<a class="source-card external" href="${s.url || "#"}" target="_blank" rel="noopener noreferrer">
         <div class="source-card-header">
           <span class="ext-badge">${iconHTML("external-link")}</span>
           <span class="source-meta">${esc(fmtAuthors(s.authors))}${s.year ? " · " + s.year : ""} — ${esc(truncate(s.title || "", 46))}</span>
         </div>
         <div class="source-text">${esc(truncate(s.context || "", 160))}</div>
       </a>`).join("") + `</div>`);
  }

  slot.innerHTML = parts.join("");
  slot.querySelectorAll(".source-card:not(.external)").forEach((card) => {
    card.addEventListener("click", () => openSourceCard(corpus[+card.dataset.idx]));
  });
}

function dedupeSources(sources) {
  const seen = new Set();
  const out = [];
  for (const s of sources) {
    const key = `${s.item_key}|${s.page}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(s);
  }
  return out;
}

// ── Actions par message (copier / BibTeX) ────────────────────────────────
function buildMsgActionsHTML(hidden) {
  return `<div class="msg-actions"${hidden ? " hidden" : ""}>
    <button class="icon-btn action-copy" title="Copier la réponse">${iconHTML("copy")}</button>
    <button class="icon-btn action-bibtex" title="Copier les sources en BibTeX">${iconHTML("quote")}</button>
  </div>`;
}

function wireMsgActions(root, markdownText, corpusSources) {
  const copyBtn = root.querySelector(".action-copy");
  const bibtexBtn = root.querySelector(".action-bibtex");
  if (copyBtn) copyBtn.addEventListener("click", () => copyToClipboard(markdownText, copyBtn));
  if (bibtexBtn) bibtexBtn.addEventListener("click", () => copyToClipboard(buildBibtex(corpusSources), bibtexBtn));
}

async function copyToClipboard(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    const original = btn.innerHTML;
    btn.innerHTML = iconHTML("check");
    btn.classList.add("copied");
    setTimeout(() => { btn.innerHTML = original; btn.classList.remove("copied"); }, 1500);
  } catch (e) {
    console.error("Clipboard error", e);
  }
}

function buildBibtex(sources) {
  if (!sources || !sources.length) return "% Aucune source citée dans cette réponse.";
  const usedKeys = new Set();
  return sources.map((s) => {
    const firstAuthor = (s.authors || "Unknown").split(",")[0].trim().split(/\s+/).pop() || "Unknown";
    const base = `${firstAuthor}${s.year || ""}`.replace(/[^a-zA-Z0-9]/g, "");
    let key = base, i = 0;
    while (usedKeys.has(key)) { key = base + String.fromCharCode(97 + i); i++; }
    usedKeys.add(key);

    const fields = [
      `title={${s.title || ""}}`,
      `author={${(s.authors || "").split(",").map((a) => a.trim()).filter(Boolean).join(" and ")}}`,
      s.year ? `year={${s.year}}` : null,
      s.doi ? `doi={${s.doi}}` : null,
    ].filter(Boolean).join(",\n  ");
    return `@article{${key},\n  ${fields}\n}`;
  }).join("\n\n");
}

// ── Export conversation (Markdown) ───────────────────────────────────────
async function exportConversation() {
  if (!activeConversationId) return;
  let conv;
  try { conv = await fetch(`/api/conversations/${activeConversationId}`).then((r) => r.json()); }
  catch { return; }

  let md = `# ${conv.title}\n\n`;
  for (const m of conv.messages) {
    if (m.role === "user") {
      md += `## Question\n\n${m.content}\n\n`;
    } else if (m.role === "assistant") {
      md += `## Réponse\n\n${m.content}\n\n`;
      const sources = (m.meta && m.meta.sources) || [];
      if (sources.length) {
        md += `**Sources :**\n\n`;
        for (const s of sources) {
          md += `- ${fmtAuthors(s.authors)}${s.year ? " (" + s.year + ")" : ""} — ${s.title}, p.${s.page}\n`;
        }
        md += "\n";
      }
    }
  }

  const blob = new Blob([md], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${conv.title.replace(/[^a-z0-9]+/gi, "_")}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Synchronisation Zotero (indexation incrémentale depuis l'UI) ─────────
// Le job tourne côté serveur, découplé de cette page (voir webapp/sync_job.py) : fermer
// l'onglet ne l'arrête pas. On poll son état plutôt que de streamer en SSE, précisément pour
// pouvoir s'y rebrancher — y compris au chargement de la page si une sync était déjà en cours
// (lancée avant fermeture de l'onglet, ou depuis un autre appareil).
let syncPolling = false;

async function startSync() {
  try {
    await fetch("/api/index/sync", { method: "POST" });
  } catch (e) {
    el("sync-label").textContent = `Erreur : ${e.message}`;
    return;
  }
  pollSyncStatus();
}

async function pollSyncStatus() {
  if (syncPolling) return; // un cycle de polling tourne déjà, pas besoin d'en relancer un 2e
  syncPolling = true;
  try {
    while (true) {
      let snap;
      try {
        snap = await fetch("/api/index/sync").then((r) => r.json());
      } catch {
        return; // page en cours de déchargement ou serveur injoignable : abandonne sans erreur
      }
      renderSyncStatus(snap);
      if (snap.status !== "running") {
        if (snap.status === "done" && snap.added > 0) {
          allDocs = await fetch("/api/docs").then((r) => r.json());
          el("doc-count").textContent = `${allDocs.length} articles indexés`;
          renderList(filteredDocs());
        }
        return;
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
  } finally {
    syncPolling = false;
  }
}

function renderSyncStatus(snap) {
  const btn = el("sync-btn");
  const label = el("sync-label");
  if (snap.status === "running") {
    btn.disabled = true;
    btn.classList.add("spinning");
    label.textContent = snap.total ? `${snap.current}/${snap.total} — ${snap.message}` : "Démarrage…";
    return;
  }
  btn.disabled = false;
  btn.classList.remove("spinning");
  if (snap.status === "done") {
    label.textContent = snap.added > 0 ? `${snap.added} article(s) ajouté(s)` : "À jour";
  } else if (snap.status === "error") {
    label.textContent = `Erreur : ${snap.error}`;
  } else {
    return; // "idle" : rien à afficher, laisse le libellé par défaut du HTML
  }
  setTimeout(() => { label.textContent = "Synchroniser Zotero"; }, 4000);
}

// ── Rendu Markdown (réponse de l'agent) ──────────────────────────────────
// marked ne filtre pas le HTML brut qu'il rencontre dans le Markdown source ; DOMPurify
// assainit le résultat avant toute injection via innerHTML — nécessaire ici puisque la
// réponse peut recopier des extraits d'articles ou des contextes de citation externes,
// pas seulement du texte généré par le LLM lui-même.
function renderMarkdown(text) {
  const html = marked.parse(text || "");
  return window.DOMPurify ? window.DOMPurify.sanitize(html) : esc(text);
}

// ── Utils ─────────────────────────────────────────────────────────────────
function addMessage(role, text) {
  const msgs = el("messages");
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `<div class="bubble">${esc(text)}</div><span class="msg-time">${now()}</span>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

function fmtAuthors(authors) {
  if (!authors) return "";
  const parts = authors.split(",").map((a) => a.trim()).filter(Boolean);
  const names = parts.slice(0, 2).map((a) => a.split(/\s+/)[0]); // nom de famille (1er token)
  return names.join(", ") + (parts.length > 2 ? " et al." : "");
}
function truncate(s, n) { s = s || ""; return s.length > n ? s.slice(0, n) + "…" : s; }
function now() { return new Date().toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" }); }
function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

init();
