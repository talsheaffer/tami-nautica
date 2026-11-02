// prototype/js/wiki_app.js
// Client application for the 31-page Biographical & Artistic Wiki Portal (/wiki),
// integrated with Universal DOM Element Inspector, Clickable Footnote Citations, & Family Beta Feedback.

import { BetaFeedbackOverlay } from './feedback_overlay.js';
import { applyBetaFont, getSavedBetaFontId, mountPortalHeaderFontPicker } from './beta_font.js';

applyBetaFont(getSavedBetaFontId(), { persist: false, emitDiag: false });

function escapeHtml(raw) {
  return String(raw || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function unescapeHtmlEntities(str) {
  return String(str || '')
    .replace(/&quot;/g, '"')
    .replace(/&gt;/g, '>')
    .replace(/&lt;/g, '<')
    .replace(/&amp;/g, '&');
}

function decodeUriSafe(val) {
  try {
    return decodeURIComponent(val);
  } catch {
    return val;
  }
}

function slugifyHeading(rawHeading) {
  let text = String(rawHeading || '');
  text = text.replace(/\[\^[^\]]+\]/g, '');
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');
  text = text.replace(/[*_`~]/g, '');
  text = text.replace(/<[^>]+>/g, '');
  text = text.toLowerCase().trim();
  text = text.replace(/[^\p{L}\p{M}\p{N}\s_-]/gu, '');
  text = text.replace(/\s+/g, '-');
  return text.normalize('NFC');
}

function sanitizeFootnoteSlug(rawKey) {
  return String(rawKey || '')
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}_-]+/gu, '-')
    .replace(/^-+|-+$/g, '') || 'note';
}

function stripMarkdownForTooltip(rawMd) {
  let s = String(rawMd || '');
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1 ($2)');
  s = s.replace(/[*_`~]/g, '');
  s = s.replace(/\s+/g, ' ').trim();
  return s.length > 220 ? `${s.slice(0, 217)}...` : s;
}

function resolveRelativeWikiPath(baseRelPath, href) {
  const raw = String(href || '').trim();
  if (!raw || /^(?:https?:|mailto:|tel:|data:)/i.test(raw)) return null;
  const hashIdx = raw.indexOf('#');
  const cleanHref = (hashIdx >= 0 ? raw.slice(0, hashIdx) : raw).trim();
  const hash = hashIdx >= 0 ? decodeUriSafe(raw.slice(hashIdx + 1).trim()) : '';

  if (!cleanHref) return null; // pure #anchor handled separately
  if (!cleanHref.endsWith('.md')) return null;

  const baseParts = baseRelPath.split('/');
  baseParts.pop(); // directory of current file

  for (const part of cleanHref.split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') {
      if (baseParts.length === 0) {
        return null; // climbs outside wiki/ root
      }
      baseParts.pop();
    } else {
      baseParts.push(decodeUriSafe(part));
    }
  }

  if (baseParts.length === 0) return null;
  return {
    relPath: baseParts.join('/'),
    hash
  };
}

function splitTrailingUrlPunctuation(rawUrl) {
  let url = rawUrl;
  let trailing = '';
  while (url.length > 0) {
    const last = url[url.length - 1];
    if (/[.,;:!?]/.test(last)) {
      trailing = last + trailing;
      url = url.slice(0, -1);
      continue;
    }
    if (last === ')') {
      const opens = (url.match(/\(/g) || []).length;
      const closes = (url.match(/\)/g) || []).length;
      if (closes > opens) {
        trailing = last + trailing;
        url = url.slice(0, -1);
        continue;
      }
    }
    if (last === ']') {
      const opens = (url.match(/\[/g) || []).length;
      const closes = (url.match(/\]/g) || []).length;
      if (closes > opens) {
        trailing = last + trailing;
        url = url.slice(0, -1);
        continue;
      }
    }
    break;
  }
  return { url, trailing };
}

function applyEmphasis(escapedStr) {
  let out = escapedStr;
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  return out;
}

function formatInlineMarkdown(line, currentRelPath, footnoteCtx = null, insideFootnote = false) {
  let s = escapeHtml(line);

  const codeTokens = [];
  const fnTokens = [];
  const linkTokens = [];

  // 1. Protect inline code spans first
  s = s.replace(/`([^`]+)`/g, (_, codeContent) => {
    const idx = codeTokens.length;
    codeTokens.push(`<code>${codeContent}</code>`);
    return `\x00CODE_${idx}\x00`;
  });

  // 2. Unverified family callout badge
  s = s.replace(
    /\[UNVERIFIED\s*[—–-]\s*NEEDS FAMILY CONFIRMATION\]/gi,
    '<span class="wiki-unverified-badge">❓ דורש אישור / השלמה מהמשפחה</span>'
  );

  // 3. Inline footnote citations [^slug]
  if (!insideFootnote) {
    s = s.replace(/\[\^([^\]]+)\]/g, (_, escapedKey) => {
      const rawKey = unescapeHtmlEntities(escapedKey).trim();
      const idx = fnTokens.length;
      if (footnoteCtx) {
        const entry = footnoteCtx.ensureEntry(rawKey);
        const refId =
          entry.refIds.length === 0
            ? `fnref-${entry.safeSlug}`
            : `fnref-${entry.safeSlug}-${entry.refIds.length + 1}`;
        entry.refIds.push(refId);
        const tooltip = entry.rawText
          ? stripMarkdownForTooltip(entry.rawText)
          : `מקור [${entry.number}]`;
        fnTokens.push(
          `<sup class="wiki-cite-sup" id="${escapeHtml(refId)}"><a href="#fn-${escapeHtml(
            entry.safeSlug
          )}" class="wiki-cite-ref" data-fn-target="fn-${escapeHtml(
            entry.safeSlug
          )}" title="${escapeHtml(tooltip)}">[${entry.number}]</a></sup>`
        );
      } else {
        const safeSlug = sanitizeFootnoteSlug(rawKey);
        fnTokens.push(
          `<sup class="wiki-cite-sup"><a href="#fn-${escapeHtml(
            safeSlug
          )}" class="wiki-cite-ref" data-fn-target="fn-${escapeHtml(safeSlug)}">[${escapeHtml(
            rawKey
          )}]</a></sup>`
        );
      }
      return `\x00FN_${idx}\x00`;
    });
  }

  // 4. Explicit Markdown links [label](target)
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, rawLabel, escapedTarget) => {
    const formattedLabel = applyEmphasis(rawLabel);
    const rawTarget = unescapeHtmlEntities(escapedTarget).trim().replace(/^<|>$/g, '');
    const idx = linkTokens.length;

    if (rawTarget.startsWith('#')) {
      const anchorName = decodeUriSafe(rawTarget.slice(1).trim());
      linkTokens.push(
        `<a href="#${escapeHtml(anchorName)}" class="wiki-anchor-link" data-wiki-anchor="${escapeHtml(
          anchorName
        )}">${formattedLabel}</a>`
      );
      return `\x00LINK_${idx}\x00`;
    }

    const resolvedWiki = resolveRelativeWikiPath(currentRelPath, rawTarget);
    if (resolvedWiki) {
      const hashSuffix = resolvedWiki.hash ? `#${encodeURIComponent(resolvedWiki.hash)}` : '';
      const hashAttr = resolvedWiki.hash
        ? ` data-wiki-hash="${escapeHtml(resolvedWiki.hash)}"`
        : '';
      linkTokens.push(
        `<a href="/wiki?page=${encodeURIComponent(
          resolvedWiki.relPath
        )}${hashSuffix}" class="wiki-internal-link" data-wiki-link="${escapeHtml(
          resolvedWiki.relPath
        )}"${hashAttr}>${formattedLabel}</a>`
      );
      return `\x00LINK_${idx}\x00`;
    }

    if (/^(?:https?:|mailto:)/i.test(rawTarget)) {
      linkTokens.push(
        `<a href="${escapeHtml(
          rawTarget
        )}" class="wiki-external-link" target="_blank" rel="noopener noreferrer">${formattedLabel}</a>`
      );
      return `\x00LINK_${idx}\x00`;
    }

    linkTokens.push(`<span>${formattedLabel}</span>`);
    return `\x00LINK_${idx}\x00`;
  });

  // 5. Bare external URLs (https://... or http://...) outside Markdown links & code spans
  s = s.replace(/https?:\/\/[^\s<"]+/gi, (matchedEscapedUrl) => {
    const rawUrlCandidate = unescapeHtmlEntities(matchedEscapedUrl);
    const { url, trailing } = splitTrailingUrlPunctuation(rawUrlCandidate);
    if (!url) return matchedEscapedUrl;
    const idx = linkTokens.length;
    linkTokens.push(
      `<a href="${escapeHtml(
        url
      )}" class="wiki-external-link" target="_blank" rel="noopener noreferrer">${escapeHtml(
        url
      )}</a>`
    );
    return `\x00LINK_${idx}\x00${escapeHtml(trailing)}`;
  });

  // 6. Bold & Italic emphasis
  s = applyEmphasis(s);

  // 7. Restore tokens
  s = s.replace(/\x00LINK_(\d+)\x00/g, (_, idx) => linkTokens[Number(idx)] || '');
  s = s.replace(/\x00FN_(\d+)\x00/g, (_, idx) => fnTokens[Number(idx)] || '');
  s = s.replace(/\x00CODE_(\d+)\x00/g, (_, idx) => codeTokens[Number(idx)] || '');

  return s;
}

function isFootnotesSectionHeading(headingText) {
  const cleaned = String(headingText || '')
    .replace(/[*_`~]/g, '')
    .trim();
  return /^(?:\d+\.\s*)?(?:footnotes?\s*(?:&|and)\s*sources?|sources?\s*(?:&|and)\s*citations?|מקורות\s*והערות\s*שוליים)(?:\s*[|·—–/-]\s*.*)?$/i.test(
    cleaned
  );
}

function createFootnoteContext(lines) {
  const defs = new Map();
  const consumedLines = new Set();
  const usedSlugs = new Set();
  let inFence = false;
  let lastDefKey = null;

  const allocateUniqueSlug = (key) => {
    const base = sanitizeFootnoteSlug(key);
    let candidate = base;
    let counter = 2;
    while (usedSlugs.has(candidate)) {
      candidate = `${base}-${counter}`;
      counter += 1;
    }
    usedSlugs.add(candidate);
    return candidate;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    if (trimmed.startsWith('```') || trimmed.startsWith('~~~')) {
      inFence = !inFence;
      lastDefKey = null;
      continue;
    }
    if (inFence) continue;

    const defMatch = line.match(/^\s*\[\^([^\]]+)\]:\s*(.+)$/);
    if (defMatch) {
      const key = defMatch[1].trim();
      const body = defMatch[2].trim();
      consumedLines.add(i);
      if (defs.has(key)) {
        const existing = defs.get(key);
        existing.rawText = `${existing.rawText} ${body}`.trim();
      } else {
        defs.set(key, {
          key,
          safeSlug: allocateUniqueSlug(key),
          rawText: body,
          number: null,
          refIds: []
        });
      }
      lastDefKey = key;
      continue;
    }

    if (lastDefKey && /^(?: {2,}|\t+)\S/.test(line)) {
      const existing = defs.get(lastDefKey);
      existing.rawText = `${existing.rawText} ${trimmed}`.trim();
      consumedLines.add(i);
      continue;
    }

    lastDefKey = null;
  }

  let nextNumber = 1;
  const ordered = [];

  const ensureEntry = (rawKey) => {
    let entry = defs.get(rawKey);
    if (!entry) {
      entry = {
        key: rawKey,
        safeSlug: allocateUniqueSlug(rawKey),
        rawText: '',
        number: null,
        refIds: []
      };
      defs.set(rawKey, entry);
    }
    if (entry.number === null) {
      entry.number = nextNumber++;
      ordered.push(entry);
    }
    return entry;
  };

  const finalizeOrdered = () => {
    for (const entry of defs.values()) {
      if (entry.number === null) {
        entry.number = nextNumber++;
        ordered.push(entry);
      }
    }
    return ordered;
  };

  return {
    consumedLines,
    ensureEntry,
    finalizeOrdered
  };
}

function renderMarkdownToHtml(md, currentRelPath) {
  const lines = md.replace(/\r\n/g, '\n').split('\n');
  const footnoteCtx = createFootnoteContext(lines);
  const out = [];
  const headingSlugCounts = new Map();

  let inCode = false;
  let codeBuf = [];
  let inList = null; // 'ul' | 'ol'
  let inTable = false;
  let tableRows = [];
  let inBlockquote = false;
  let blockquoteStartLine = 0;
  let blockquoteParas = [];
  let currentQuotePara = [];

  const flushList = () => {
    if (inList) {
      out.push(`</${inList}>`);
      inList = null;
    }
  };

  const flushBlockquote = () => {
    if (!inBlockquote) return;
    if (currentQuotePara.length > 0) {
      blockquoteParas.push(currentQuotePara.join(' '));
      currentQuotePara = [];
    }
    if (blockquoteParas.length > 0) {
      const innerHtml = blockquoteParas
        .map((p) => `<p dir="auto">${formatInlineMarkdown(p, currentRelPath, footnoteCtx)}</p>`)
        .join('');
      out.push(`<blockquote id="wiki-q-${blockquoteStartLine}" dir="auto">${innerHtml}</blockquote>`);
    }
    inBlockquote = false;
    blockquoteParas = [];
  };

  const flushTable = () => {
    if (!inTable) return;
    inTable = false;
    if (tableRows.length === 0) return;
    let html = '<table>';
    tableRows.forEach((cells, rIdx) => {
      const tag = rIdx === 0 ? 'th' : 'td';
      html +=
        '<tr>' +
        cells
          .map(
            (c) =>
              `<${tag} dir="auto">${formatInlineMarkdown(
                c.trim(),
                currentRelPath,
                footnoteCtx
              )}</${tag}>`
          )
          .join('') +
        '</tr>';
    });
    html += '</table>';
    out.push(html);
    tableRows = [];
  };

  for (let i = 0; i < lines.length; i++) {
    if (footnoteCtx.consumedLines.has(i)) {
      flushList();
      flushTable();
      flushBlockquote();
      continue;
    }

    const line = lines[i];
    if (line.trim().startsWith('```')) {
      flushList();
      flushTable();
      flushBlockquote();
      if (!inCode) {
        inCode = true;
        codeBuf = [];
      } else {
        inCode = false;
        out.push(`<pre><code>${escapeHtml(codeBuf.join('\n'))}</code></pre>`);
      }
      continue;
    }
    if (inCode) {
      codeBuf.push(line);
      continue;
    }

    const trimmed = line.trim();
    if (!trimmed) {
      flushList();
      flushTable();
      flushBlockquote();
      continue;
    }

    // Blockquote lines (> ... or bare >)
    const quoteMatch = trimmed.match(/^>\s?(.*)$/);
    if (quoteMatch) {
      flushList();
      flushTable();
      if (!inBlockquote) {
        inBlockquote = true;
        blockquoteStartLine = i;
        blockquoteParas = [];
        currentQuotePara = [];
      }
      const qBody = quoteMatch[1].trim();
      if (!qBody) {
        if (currentQuotePara.length > 0) {
          blockquoteParas.push(currentQuotePara.join(' '));
          currentQuotePara = [];
        }
      } else {
        currentQuotePara.push(qBody);
      }
      continue;
    } else {
      flushBlockquote();
    }

    // Table detection
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      flushList();
      const inner = trimmed.slice(1, -1).split('|');
      const isSeparator = inner.every((c) => /^[:\s-]+$/.test(c));
      if (!isSeparator) {
        inTable = true;
        tableRows.push(inner);
      }
      continue;
    } else {
      flushTable();
    }

    if (/^---+$/.test(trimmed)) {
      flushList();
      if (out.length === 0 || !out[out.length - 1].startsWith('<hr ')) {
        out.push(
          '<hr style="border:none;border-top:1px solid rgba(244,239,234,0.1);margin:18px 0;" />'
        );
      }
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      flushList();
      const lvl = headingMatch[1].length;
      const rawHeadingText = headingMatch[2];
      if (isFootnotesSectionHeading(rawHeadingText)) {
        continue;
      }
      const baseSlug = slugifyHeading(rawHeadingText);
      let finalSlug = baseSlug;
      if (baseSlug) {
        const seenCount = headingSlugCounts.get(baseSlug) || 0;
        headingSlugCounts.set(baseSlug, seenCount + 1);
        if (seenCount > 0) {
          finalSlug = `${baseSlug}-${seenCount}`;
        }
      }
      const slugAttr = finalSlug ? ` data-heading-slug="${escapeHtml(finalSlug)}"` : '';
      out.push(
        `<h${lvl} id="wiki-h-${i}"${slugAttr} dir="auto">${formatInlineMarkdown(
          rawHeadingText,
          currentRelPath,
          footnoteCtx
        )}</h${lvl}>`
      );
      continue;
    }

    const ulMatch = trimmed.match(/^[-*]\s+(.+)$/);
    if (ulMatch) {
      if (inList !== 'ul') {
        flushList();
        inList = 'ul';
        out.push('<ul dir="auto">');
      }
      out.push(
        `<li id="wiki-li-${i}" dir="auto">${formatInlineMarkdown(
          ulMatch[1],
          currentRelPath,
          footnoteCtx
        )}</li>`
      );
      continue;
    }

    const olMatch = trimmed.match(/^\d+\.\s+(.+)$/);
    if (olMatch) {
      if (inList !== 'ol') {
        flushList();
        inList = 'ol';
        out.push('<ol dir="auto">');
      }
      out.push(
        `<li id="wiki-li-${i}" dir="auto">${formatInlineMarkdown(
          olMatch[1],
          currentRelPath,
          footnoteCtx
        )}</li>`
      );
      continue;
    }

    flushList();
    out.push(
      `<p id="wiki-p-${i}" dir="auto">${formatInlineMarkdown(
        trimmed,
        currentRelPath,
        footnoteCtx
      )}</p>`
    );
  }

  flushList();
  flushTable();
  flushBlockquote();

  // Trim trailing <hr> if immediately preceding our canonical footnotes section
  while (out.length > 0 && out[out.length - 1].startsWith('<hr ')) {
    out.pop();
  }

  const orderedFootnotes = footnoteCtx.finalizeOrdered();
  if (orderedFootnotes.length > 0) {
    const itemsHtml = orderedFootnotes
      .map((entry) => {
        const renderedDef = entry.rawText
          ? formatInlineMarkdown(entry.rawText, currentRelPath, footnoteCtx, true)
          : `<span>מקור ארכיוני (${escapeHtml(entry.key)})</span>`;
        const backrefs =
          entry.refIds.length <= 1
            ? entry.refIds
                .map(
                  (refId) =>
                    `<a href="#${escapeHtml(
                      refId
                    )}" class="wiki-fn-backref" data-fn-backref="${escapeHtml(
                      refId
                    )}" title="חזרה לציטוט בגוף הערך · Back to citation">↩</a>`
                )
                .join(' ')
            : entry.refIds
                .map(
                  (refId, idx) =>
                    `<a href="#${escapeHtml(
                      refId
                    )}" class="wiki-fn-backref" data-fn-backref="${escapeHtml(
                      refId
                    )}" title="חזרה למופע ${idx + 1} · Back to citation ${idx + 1}">↩<sup>${
                      idx + 1
                    }</sup></a>`
                )
                .join(' ');
        return `<li id="fn-${escapeHtml(
          entry.safeSlug
        )}" class="wiki-footnote-item" value="${entry.number}" dir="auto"><span class="wiki-footnote-content">${renderedDef}</span> ${backrefs}</li>`;
      })
      .join('\n');

    out.push(
      `<section class="wiki-footnotes-section" id="wiki-footnotes" dir="auto" aria-label="מקורות והערות שוליים · Sources &amp; Citations">` +
        `<h2 class="wiki-footnotes-heading" id="wiki-h-footnotes" data-heading-slug="footnotes-sources">📚 מקורות והערות שוליים · Sources &amp; Citations</h2>` +
        `<ol class="wiki-footnotes-list">\n${itemsHtml}\n</ol>` +
        `</section>`
    );
  }

  return out.join('\n');
}

class WikiPortalApp {
  constructor() {
    this.navSectionsEl = document.getElementById('wiki-nav-sections');
    this.searchInput = document.getElementById('wiki-search-input');
    this.metaPillsEl = document.getElementById('wiki-meta-pills');
    this.articleBodyEl = document.getElementById('wiki-article-body');
    this.commentPageBtn = document.getElementById('wiki-comment-page-btn');

    this.sections = [];
    this.currentPageRelPath = 'README.md';
    this.currentPageTitle = '';

    this.feedbackOverlay = new BetaFeedbackOverlay(null, {
      defaultSceneId: 'wiki',
      getTargetRef: () => this.currentPageRelPath,
      onNavigateTargetRef: async (targetRef) => {
        if (targetRef && targetRef.endsWith('.md') && targetRef !== this.currentPageRelPath) {
          await this.loadPage(targetRef, false);
        }
      }
    });

    this.init();
  }

  async init() {
    const params = new URLSearchParams(window.location.search);
    const initialPage = params.get('page') || 'README.md';
    const initialHash = window.location.hash ? decodeUriSafe(window.location.hash.slice(1)) : '';

    const resp = await fetch('/api/wiki', { cache: 'no-store' });
    const data = await resp.json();
    this.sections = data.sections || [];
    this.renderSidebar('');

    this.searchInput.addEventListener('input', () => {
      this.renderSidebar(this.searchInput.value.trim().toLowerCase());
    });

    this.commentPageBtn.addEventListener('click', () => {
      const firstHeading = this.articleBodyEl.querySelector('h1') || this.articleBodyEl;
      this.feedbackOverlay.openCommentForElement(
        firstHeading,
        'story_bio',
        `בנוגע לערך ״${this.currentPageTitle}״: `
      );
    });

    window.addEventListener('popstate', () => {
      const p = new URLSearchParams(window.location.search).get('page') || 'README.md';
      const h = window.location.hash ? decodeUriSafe(window.location.hash.slice(1)) : '';
      this.loadPage(p, false, h);
    });

    await this.loadPage(initialPage, false, initialHash);
  }

  renderSidebar(query) {
    this.navSectionsEl.innerHTML = '';
    for (const sec of this.sections) {
      const matchingPages = sec.pages.filter((p) => {
        if (!query) return true;
        const hay = `${p.title} ${p.slug} ${p.era} ${(p.people || []).join(' ')}`.toLowerCase();
        return hay.includes(query);
      });
      if (matchingPages.length === 0) continue;

      const wrap = document.createElement('div');
      wrap.style.marginBottom = '12px';
      wrap.innerHTML = `<div class="wiki-sec-title">${sec.icon} ${escapeHtml(sec.label_he)}</div>`;
      const list = document.createElement('div');
      list.className = 'wiki-nav-list';

      for (const page of matchingPages) {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `wiki-nav-item ${page.rel_path === this.currentPageRelPath ? 'active' : ''}`;
        btn.setAttribute('data-wiki-page', page.rel_path);
        const shortTitle = page.title.includes('|') ? page.title.split('|')[1].trim() : page.title;
        btn.innerHTML = `
          <span>${escapeHtml(shortTitle)}</span>
          ${page.era ? `<span class="wiki-era-tag">${escapeHtml(page.era)}</span>` : ''}
        `;
        btn.addEventListener('click', () => this.loadPage(page.rel_path, true));
        list.appendChild(btn);
      }

      wrap.appendChild(list);
      this.navSectionsEl.appendChild(wrap);
    }
  }

  scrollToTargetAnchor(anchorId, immediate = false) {
    if (!anchorId) return false;
    const normSlug = slugifyHeading(anchorId);
    const targetEl =
      document.getElementById(anchorId) ||
      this.articleBodyEl.querySelector(`[data-heading-slug="${CSS.escape(anchorId.toLowerCase())}"]`) ||
      (normSlug
        ? this.articleBodyEl.querySelector(`[data-heading-slug="${CSS.escape(normSlug)}"]`)
        : null);

    if (!targetEl) return false;
    targetEl.scrollIntoView({ behavior: immediate ? 'auto' : 'smooth', block: 'center' });
    targetEl.classList.remove('wiki-fn-highlight');
    void targetEl.offsetWidth; // trigger reflow for pulse animation
    targetEl.classList.add('wiki-fn-highlight');
    window.setTimeout(() => {
      targetEl.classList.remove('wiki-fn-highlight');
    }, 2200);
    return true;
  }

  async loadPage(relPath, pushHistory = true, targetHash = '') {
    const resp = await fetch(`/api/wiki?page=${encodeURIComponent(relPath)}`, { cache: 'no-store' });
    if (!resp.ok) {
      this.articleBodyEl.innerHTML = `<p>לא ניתן לטעון את הערך המבוקש (${escapeHtml(relPath)}).</p>`;
      return;
    }
    const { page } = await resp.json();
    this.currentPageRelPath = page.rel_path;
    this.currentPageTitle = page.title;

    if (pushHistory) {
      const url = new URL(window.location.href);
      url.searchParams.set('page', page.rel_path);
      url.searchParams.delete('focus_fb');
      url.hash = targetHash ? `#${encodeURIComponent(targetHash)}` : '';
      window.history.pushState({}, '', url.toString());
    }

    this.navSectionsEl.querySelectorAll('.wiki-nav-item').forEach((btn) => {
      btn.classList.toggle('active', btn.getAttribute('data-wiki-page') === page.rel_path);
    });

    this.metaPillsEl.innerHTML = `
      <span class="portal-stat-pill">📄 ${escapeHtml(page.rel_path)}</span>
      ${page.era ? `<span class="portal-stat-pill">🕰️ תקופה: ${escapeHtml(page.era)}</span>` : ''}
      <span class="portal-stat-pill">🎯 לחצו על ״סמן והגב״ למעלה כדי להדגיש כל פסקה או משפט!</span>
    `;

    this.articleBodyEl.innerHTML = renderMarkdownToHtml(page.markdown_body, page.rel_path);

    // Internal cross-page wiki links
    this.articleBodyEl.querySelectorAll('a[data-wiki-link]').forEach((a) => {
      a.addEventListener('click', async (e) => {
        e.preventDefault();
        const target = a.getAttribute('data-wiki-link');
        const hash = a.getAttribute('data-wiki-hash') || '';
        if (target) {
          if (target === this.currentPageRelPath && hash) {
            this.scrollToTargetAnchor(hash);
          } else {
            await this.loadPage(target, true, hash);
            if (!hash) {
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }
          }
        }
      });
    });

    // Inline citation footnote links [n] -> #fn-...
    this.articleBodyEl.querySelectorAll('a.wiki-cite-ref[data-fn-target]').forEach((a) => {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        const fnId = a.getAttribute('data-fn-target');
        if (fnId) {
          const url = new URL(window.location.href);
          url.hash = `#${fnId}`;
          window.history.replaceState({}, '', url.toString());
          this.scrollToTargetAnchor(fnId);
        }
      });
    });

    // Footnote back-links ↩ -> #fnref-...
    this.articleBodyEl.querySelectorAll('a.wiki-fn-backref[data-fn-backref]').forEach((a) => {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        const refId = a.getAttribute('data-fn-backref');
        if (refId) {
          const url = new URL(window.location.href);
          url.hash = `#${refId}`;
          window.history.replaceState({}, '', url.toString());
          this.scrollToTargetAnchor(refId);
        }
      });
    });

    // Same-page heading anchor links #...
    this.articleBodyEl.querySelectorAll('a.wiki-anchor-link[data-wiki-anchor]').forEach((a) => {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        const anchor = a.getAttribute('data-wiki-anchor');
        if (anchor) {
          const url = new URL(window.location.href);
          url.hash = `#${encodeURIComponent(anchor)}`;
          window.history.replaceState({}, '', url.toString());
          this.scrollToTargetAnchor(anchor);
        }
      });
    });

    if (targetHash) {
      window.requestAnimationFrame(() => {
        this.scrollToTargetAnchor(targetHash, !pushHistory);
      });
    }

    if (this.feedbackOverlay) {
      this.feedbackOverlay.notifyPageContentUpdated();
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  mountPortalHeaderFontPicker(document.getElementById('portal-beta-actions'));
  new WikiPortalApp();
});
