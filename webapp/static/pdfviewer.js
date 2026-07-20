// pdfviewer.js — wrapper PDF.js : rendu d'une page + surlignage par recherche de texte.
// Aucune coordonnée n'est stockée côté serveur : on localise le passage en comparant
// le texte du chunk aux items du text-content de la page (miroir de highlighter.py).
import * as pdfjsLib from "/static/pdfjs/build/pdf.mjs";

pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/pdfjs/build/pdf.worker.mjs";

const FRAGMENT_WORDS = 30; // taille des fragments recherchés (comme highlighter.py)
const FALLBACK_WORDS = 5; // repli si le fragment complet est introuvable
const SCALE = 1.5; // échelle de rendu

let pdfDoc = null;
let currentPage = 1;
let currentViewport = null;
let pendingHighlight = null; // texte à surligner une fois la page rendue

const canvas = document.getElementById("pdf-canvas");
const ctx = canvas.getContext("2d");
const hlLayer = document.getElementById("highlight-layer");
const pageContainer = document.getElementById("page-container");
const placeholder = document.getElementById("viewer-placeholder");
const pageInfo = document.getElementById("page-info");
const viewerNav = document.getElementById("viewer-nav");

// ── Chargement d'un PDF ───────────────────────────────────────────────────
export async function loadPdf(url) {
  clearHighlights();
  pdfDoc = await pdfjsLib.getDocument(url).promise;
  currentPage = 1;
  placeholder.hidden = true;
  pageContainer.hidden = false;
  viewerNav.hidden = false;
  await renderPage(1);
}

// ── Rendu d'une page ──────────────────────────────────────────────────────
export async function renderPage(n) {
  if (!pdfDoc) return;
  currentPage = Math.min(Math.max(1, n), pdfDoc.numPages);
  const page = await pdfDoc.getPage(currentPage);
  currentViewport = page.getViewport({ scale: SCALE });

  canvas.width = currentViewport.width;
  canvas.height = currentViewport.height;
  hlLayer.style.width = `${currentViewport.width}px`;
  hlLayer.style.height = `${currentViewport.height}px`;

  await page.render({ canvasContext: ctx, viewport: currentViewport }).promise;
  pageInfo.textContent = `p. ${currentPage} / ${pdfDoc.numPages}`;

  clearHighlights();
  if (pendingHighlight) {
    await applyHighlight(page, pendingHighlight);
    pendingHighlight = null;
  }
}

export function goToPage(n) { return renderPage(n); }
export function nextPage() { return renderPage(currentPage + 1); }
export function prevPage() { return renderPage(currentPage - 1); }
export function getCurrentPage() { return currentPage; }

// ── Surlignage : va à la page puis surligne le passage ─────────────────────
export async function highlightPassage(pageNumber, text) {
  pendingHighlight = text;
  if (currentPage === pageNumber && pdfDoc) {
    const page = await pdfDoc.getPage(pageNumber);
    clearHighlights();
    await applyHighlight(page, text);
    pendingHighlight = null;
  } else {
    await renderPage(pageNumber); // le highlight en attente sera appliqué au rendu
  }
}

function clearHighlights() { hlLayer.innerHTML = ""; }

// ── Cœur : localise les fragments du texte dans les items de la page ───────
async function applyHighlight(page, text) {
  const content = await page.getTextContent();
  const items = content.items
    .filter((it) => it.str && it.str.trim())
    .map((it) => ({ str: it.str, rect: itemRect(it) }));
  if (!items.length) return;

  const { norm, map } = buildNormIndex(items);
  const fragments = splitFragments(text);
  const rectsToDraw = [];

  for (const frag of fragments) {
    const needle = normalize(frag);
    if (needle.length < 8) continue;
    let range = findRange(norm, needle);
    if (!range) {
      // repli : premiers FALLBACK_WORDS mots du fragment
      const short = normalize(frag.split(/\s+/).slice(0, FALLBACK_WORDS).join(" "));
      if (short.length >= 8) range = findRange(norm, short);
    }
    if (range) collectRects(map, items, range, rectsToDraw);
  }
  drawRects(rectsToDraw);
}

// Rectangle d'un item en coordonnées viewport (canvas).
function itemRect(item) {
  const tx = pdfjsLib.Util.transform(currentViewport.transform, item.transform);
  const fontHeight = Math.hypot(tx[2], tx[3]);
  const width = item.width * currentViewport.scale;
  return { left: tx[4], top: tx[5] - fontHeight, width, height: fontHeight };
}

// Concatène le texte normalisé de tous les items + carte char→index d'item.
function buildNormIndex(items) {
  let norm = "";
  const map = []; // map[i] = index de l'item d'où provient norm[i]
  items.forEach((it, idx) => {
    const s = it.str.toLowerCase();
    for (const ch of s) {
      if (isAlnum(ch)) { norm += ch; map.push(idx); }
      else if (!norm.endsWith(" ")) { norm += " "; map.push(idx); }
    }
    if (!norm.endsWith(" ")) { norm += " "; map.push(idx); }
  });
  return { norm, map };
}

function findRange(haystack, needle) {
  const i = haystack.indexOf(needle);
  return i === -1 ? null : { start: i, end: i + needle.length };
}

// Récupère les rects des items couverts par [start, end) et fusionne par ligne.
function collectRects(map, items, range, out) {
  const idxs = new Set();
  for (let i = range.start; i < range.end && i < map.length; i++) idxs.add(map[i]);
  for (const idx of idxs) out.push(items[idx].rect);
}

function drawRects(rects) {
  for (const r of rects) {
    const div = document.createElement("div");
    div.className = "highlight-box";
    div.style.left = `${r.left}px`;
    div.style.top = `${r.top}px`;
    div.style.width = `${r.width}px`;
    div.style.height = `${r.height}px`;
    hlLayer.appendChild(div);
  }
}

// ── Utilitaires texte ──────────────────────────────────────────────────────
function splitFragments(text) {
  const words = text.split(/\s+/).filter(Boolean);
  const frags = [];
  for (let i = 0; i < words.length; i += FRAGMENT_WORDS) {
    frags.push(words.slice(i, i + FRAGMENT_WORDS).join(" "));
  }
  return frags;
}

function normalize(s) {
  return s.toLowerCase().replace(/[^0-9a-zà-ÿ]+/gi, " ").trim();
}
function isAlnum(ch) { return /[0-9a-zà-ÿ]/i.test(ch); }
