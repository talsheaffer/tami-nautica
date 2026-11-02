// prototype/js/dom_snapshot.js
// Universal DOM UI Element Inspector (Marker.io / BugHerd / DevTools style)
// + Zero-CDN Composite HTML + WebGL Viewport Snapshot Renderer.

const IGNORED_BETA_CONTAINERS =
  '#beta-drawer, #beta-highlight-banner, #beta-highlight-surface, #beta-hover-inspector, #beta-pins-layer, #beta-lightbox, #beta-focus-bar';

const SEMANTIC_UI_SELECTOR = [
  'button',
  'a[href]',
  'input',
  'select',
  'textarea',
  '[role="button"]',
  '.portal-photo-card',
  '.title-badge',
  '.scene-btn',
  '.aux-btn',
  '.icon-btn',
  '.story-card',
  '.upload-dropzone',
  '.wiki-callout',
  '.wiki-nav-item',
  '.portal-stat-pill',
  'h1',
  'h2',
  'h3',
  'h4',
  'blockquote',
  'li',
  'p'
].join(', ');

function cssEscapeIdent(raw) {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(raw);
  }
  return String(raw).replace(/([^a-zA-Z0-9_-])/g, '\\$1');
}

export function getUniqueCssSelector(el) {
  if (!el || el.nodeType !== 1) return null;
  if (el.id) {
    return `#${cssEscapeIdent(el.id)}`;
  }
  const mediaId = el.getAttribute('data-media-id');
  if (mediaId) {
    return `[data-media-id="${cssEscapeIdent(mediaId)}"]`;
  }
  const sceneAttr = el.getAttribute('data-scene');
  if (sceneAttr) {
    return `${el.tagName.toLowerCase()}[data-scene="${cssEscapeIdent(sceneAttr)}"]`;
  }
  const wikiLinkAttr = el.getAttribute('data-wiki-page');
  if (wikiLinkAttr) {
    return `[data-wiki-page="${cssEscapeIdent(wikiLinkAttr)}"]`;
  }

  const parts = [];
  let curr = el;
  while (curr && curr.nodeType === 1 && curr !== document.body && parts.length < 5) {
    if (curr.id) {
      parts.unshift(`#${cssEscapeIdent(curr.id)}`);
      break;
    }
    const tag = curr.tagName.toLowerCase();
    let nth = 1;
    let sib = curr.previousElementSibling;
    while (sib) {
      if (sib.tagName === curr.tagName) nth += 1;
      sib = sib.previousElementSibling;
    }
    parts.unshift(`${tag}:nth-of-type(${nth})`);
    curr = curr.parentElement;
  }
  return parts.join(' > ');
}

function classifyElementPrefixHe(el) {
  const tag = el.tagName.toLowerCase();
  if (el.classList.contains('portal-photo-card')) return 'תמונה בקטלוג';
  if (tag === 'button' || el.getAttribute('role') === 'button') return 'כפתור ממשק';
  if (tag === 'a') return 'קישור ניווט';
  if (tag === 'input' || tag === 'textarea' || tag === 'select') return 'שדה קלט';
  if (/^h[1-4]$/.test(tag)) return 'כותרת';
  if (tag === 'p') return 'פסקה';
  if (tag === 'li') return 'סעיף ברשימה';
  if (tag === 'blockquote') return 'ציטוט';
  return 'רכיב ממשק';
}

export function describeUiElement(el) {
  if (!el || el === document.body || el === document.documentElement) return null;
  if (
    el.id === 'spatial-canvas' ||
    el.id === 'canvas-container' ||
    el.id === 'ui-overlay' ||
    el.id === 'loading-overlay'
  ) {
    return null;
  }
  if (el.closest(IGNORED_BETA_CONTAINERS) || el.closest('#loading-overlay') || el.closest('.hidden')) {
    return null;
  }

  const target = el.closest(SEMANTIC_UI_SELECTOR);
  if (!target || target.closest(IGNORED_BETA_CONTAINERS) || target.closest('#loading-overlay')) {
    return null;
  }
  if (target === document.body || target === document.documentElement) return null;
  if (target.closest('.hidden')) return null;

  const infoDrawer = target.closest('#info-drawer');
  if (infoDrawer && !infoDrawer.classList.contains('open')) return null;

  const spatialSidebar = target.closest('#spatial-sidebar');
  if (spatialSidebar && spatialSidebar.classList.contains('collapsed')) return null;

  const cs = window.getComputedStyle(target);
  if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity || '1') < 0.05) {
    return null;
  }

  const rect = target.getBoundingClientRect();
  if (rect.width < 4 || rect.height < 4) return null;

  const mediaId =
    target.getAttribute('data-media-id') ||
    (target.closest('[data-media-id]') &&
      target.closest('[data-media-id]').getAttribute('data-media-id')) ||
    null;

  const rawText = (
    target.getAttribute('data-feedback-label') ||
    target.getAttribute('aria-label') ||
    target.getAttribute('placeholder') ||
    target.innerText ||
    target.textContent ||
    ''
  )
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 68);

  const selector = getUniqueCssSelector(target);
  const prefix = classifyElementPrefixHe(target);
  const labelHe = rawText ? `${prefix}: ${rawText}` : `${prefix} (${selector || target.tagName.toLowerCase()})`;

  return {
    element: target,
    rect,
    selector,
    labelHe,
    targetRef: mediaId
  };
}

export function inspectUiElementAtPoint(clientX, clientY, surfaceElementsToHide = []) {
  const hiddenSet = new Set(surfaceElementsToHide.filter(Boolean));
  const stack =
    typeof document.elementsFromPoint === 'function'
      ? document.elementsFromPoint(clientX, clientY)
      : [document.elementFromPoint(clientX, clientY)].filter(Boolean);

  for (const candidate of stack) {
    if (!candidate || hiddenSet.has(candidate)) continue;
    let insideHiddenSurface = false;
    for (const surf of hiddenSet) {
      if (surf.contains(candidate)) {
        insideHiddenSurface = true;
        break;
      }
    }
    if (insideHiddenSurface) continue;

    const desc = describeUiElement(candidate);
    if (desc) {
      return desc;
    }
  }
  return null;
}

export function inspectUiElementInRect(rectPx, surfaceElementsToHide = []) {
  const x0 = Math.min(rectPx.x0, rectPx.x1);
  const y0 = Math.min(rectPx.y0, rectPx.y1);
  const x1 = Math.max(rectPx.x0, rectPx.x1);
  const y1 = Math.max(rectPx.y0, rectPx.y1);
  const cx = 0.5 * (x0 + x1);
  const cy = 0.5 * (y0 + y1);

  // 1. Check center first (most natural when boxing a button or card)
  const centerDesc = inspectUiElementAtPoint(cx, cy, surfaceElementsToHide);
  if (centerDesc) {
    return centerDesc;
  }

  // 2. Sample 3x3 interior points in case user boxed slightly off-center around a button
  const counts = new Map();
  const bySelector = new Map();
  for (let iy = 0; iy < 3; iy++) {
    for (let ix = 0; ix < 3; ix++) {
      const sx = x0 + ((ix + 0.5) / 3) * (x1 - x0);
      const sy = y0 + ((iy + 0.5) / 3) * (y1 - y0);
      const desc = inspectUiElementAtPoint(sx, sy, surfaceElementsToHide);
      if (desc && desc.selector) {
        counts.set(desc.selector, (counts.get(desc.selector) || 0) + 1);
        bySelector.set(desc.selector, desc);
      }
    }
  }

  let bestSel = null;
  let bestCount = 0;
  for (const [sel, cnt] of counts.entries()) {
    if (cnt > bestCount) {
      bestCount = cnt;
      bestSel = sel;
    }
  }
  return bestSel ? bySelector.get(bestSel) : null;
}

function drawRoundedRectPath(ctx, x, y, w, h, radius) {
  const r = Math.max(0, Math.min(radius, w * 0.5, h * 0.5));
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function isElementVisibleInViewport(el, vw, vh) {
  if (!el || el.closest(IGNORED_BETA_CONTAINERS)) return false;
  if (el.classList.contains('hidden')) return false;
  const cs = window.getComputedStyle(el);
  if (cs.display === 'none' || cs.visibility === 'hidden' || parseFloat(cs.opacity || '1') < 0.05) {
    return false;
  }
  const r = el.getBoundingClientRect();
  if (r.width < 2 || r.height < 2) return false;
  if (r.bottom < 0 || r.top > vh || r.right < 0 || r.left > vw) return false;
  return true;
}

function drawImageCoverOrContain(ctx, imgEl, dx, dy, dw, dh, objectFit) {
  const nw = imgEl.naturalWidth;
  const nh = imgEl.naturalHeight;
  if (!nw || !nh) return;

  if (objectFit === 'cover') {
    const imgRatio = nw / nh;
    const boxRatio = dw / dh;
    let sx = 0;
    let sy = 0;
    let sw = nw;
    let sh = nh;
    if (imgRatio > boxRatio) {
      sw = nh * boxRatio;
      sx = (nw - sw) * 0.5;
    } else {
      sh = nw / boxRatio;
      sy = (nh - sh) * 0.5;
    }
    ctx.drawImage(imgEl, sx, sy, sw, sh, dx, dy, dw, dh);
  } else {
    ctx.drawImage(imgEl, dx, dy, dw, dh);
  }
}

export function captureCompositeViewportSnapshot({ sceneManager, highlightRecord, pickingCfg }) {
  const vw = Math.max(1, window.innerWidth);
  const vh = Math.max(1, window.innerHeight);
  const maxW = pickingCfg ? pickingCfg.snapshot_max_width_px : 720;
  const jpegQuality = pickingCfg ? pickingCfg.snapshot_jpeg_quality : 0.84;

  const scale = Math.min(1.0, maxW / vw);
  const outW = Math.max(32, Math.round(vw * scale));
  const outH = Math.max(32, Math.round(vh * scale));

  const snapCanvas = document.createElement('canvas');
  snapCanvas.width = outW;
  snapCanvas.height = outH;
  const ctx = snapCanvas.getContext('2d');

  // 1. Page Background + WebGL Canvas (if present)
  const bodyBg = window.getComputedStyle(document.body).backgroundColor;
  ctx.fillStyle = bodyBg && bodyBg !== 'rgba(0, 0, 0, 0)' ? bodyBg : '#121314';
  ctx.fillRect(0, 0, outW, outH);

  if (sceneManager && sceneManager.canvas) {
    sceneManager.renderCleanFrame();
    ctx.drawImage(sceneManager.canvas, 0, 0, outW, outH);
  }

  // 2. Composite Visible DOM Surfaces (bars, cards, buttons, pills)
  const surfaceNodes = document.querySelectorAll(
    '.beta-hub-bar, .portal-topbar, .wiki-sidebar, .wiki-article-card, .portal-card, .portal-photo-card, .upload-panel, .title-badge, .scene-selector, #info-drawer.open, .story-card, button, .portal-nav-link, .portal-stat-pill, blockquote'
  );
  surfaceNodes.forEach((el) => {
    if (!isElementVisibleInViewport(el, vw, vh)) return;
    const cs = window.getComputedStyle(el);
    const bg = cs.backgroundColor;
    const borderCol = cs.borderTopColor;
    const borderW = parseFloat(cs.borderTopWidth || '0');
    const radius = parseFloat(cs.borderTopLeftRadius || '0') * scale;
    const r = el.getBoundingClientRect();
    const x = r.left * scale;
    const y = r.top * scale;
    const w = r.width * scale;
    const h = r.height * scale;

    const hasBg = bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent';
    const hasBorder = borderW > 0 && borderCol && borderCol !== 'rgba(0, 0, 0, 0)';
    if (!hasBg && !hasBorder) return;

    ctx.save();
    drawRoundedRectPath(ctx, x, y, w, h, radius);
    if (hasBg) {
      ctx.fillStyle = bg;
      ctx.fill();
    }
    if (hasBorder) {
      ctx.strokeStyle = borderCol;
      ctx.lineWidth = Math.max(1, borderW * scale);
      ctx.stroke();
    }
    ctx.restore();
  });

  // 3. Composite Visible Raster Images (Catalog photos, Wiki inline photos)
  document.querySelectorAll('img').forEach((imgEl) => {
    if (!isElementVisibleInViewport(imgEl, vw, vh)) return;
    if (!imgEl.complete || !imgEl.naturalWidth) return;
    const r = imgEl.getBoundingClientRect();
    const cs = window.getComputedStyle(imgEl);
    const radius = parseFloat(cs.borderTopLeftRadius || '0') * scale;
    ctx.save();
    if (radius > 0) {
      drawRoundedRectPath(ctx, r.left * scale, r.top * scale, r.width * scale, r.height * scale, radius);
      ctx.clip();
    }
    try {
      drawImageCoverOrContain(
        ctx,
        imgEl,
        r.left * scale,
        r.top * scale,
        r.width * scale,
        r.height * scale,
        cs.objectFit || 'fill'
      );
    } catch (_) {
      // Cross-origin image taint guard (all project images are same-origin)
    }
    ctx.restore();
  });

  // 4. Composite Visible UI Text Labels (buttons, headings, badges, paragraphs)
  const textNodes = document.querySelectorAll(
    'button, .portal-nav-link, h1, h2, h3, h4, p, li, .photo-card-title, .photo-card-caption, .portal-stat-pill'
  );
  textNodes.forEach((el) => {
    if (!isElementVisibleInViewport(el, vw, vh)) return;
    // Avoid double-drawing container text if child buttons/headings exist inside
    if (el.tagName === 'LI' && el.querySelector('p, h1, h2, h3, h4')) return;
    const text = (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
    if (!text) return;

    const r = el.getBoundingClientRect();
    const cs = window.getComputedStyle(el);
    const fontSize = Math.max(9, Math.min(22, parseFloat(cs.fontSize || '13') * scale));
    const fontWeight = cs.fontWeight || '500';
    const color = cs.color || '#F4EFEA';
    const dir = cs.direction || 'rtl';

    ctx.save();
    ctx.beginPath();
    ctx.rect(r.left * scale, r.top * scale, r.width * scale, r.height * scale);
    ctx.clip();

    ctx.font = `${fontWeight} ${fontSize.toFixed(1)}px -apple-system, BlinkMacSystemFont, "Heebo", sans-serif`;
    ctx.fillStyle = color;
    ctx.textBaseline = 'middle';
    ctx.direction = dir;

    const padX = Math.min(10 * scale, r.width * scale * 0.15);
    const isCentered = cs.textAlign === 'center' || el.tagName === 'BUTTON';
    let tx = r.left * scale + r.width * scale * 0.5;
    if (isCentered) {
      ctx.textAlign = 'center';
    } else if (dir === 'rtl') {
      ctx.textAlign = 'right';
      tx = r.right * scale - padX;
    } else {
      ctx.textAlign = 'left';
      tx = r.left * scale + padX;
    }

    const ty = r.top * scale + Math.min(r.height * scale * 0.5, fontSize * 1.15);
    ctx.fillText(text.slice(0, 95), tx, ty);
    ctx.restore();
  });

  // 5. Highlight Overlay + Dimmed Surround + Reticle + Target Badge
  if (highlightRecord && highlightRecord.screen_rect_norm) {
    const r = highlightRecord.screen_rect_norm;
    const rx = r.x0 * outW;
    const ry = r.y0 * outH;
    const rw = Math.max(14, (r.x1 - r.x0) * outW);
    const rh = Math.max(14, (r.y1 - r.y0) * outH);

    ctx.save();
    ctx.fillStyle = 'rgba(8, 10, 14, 0.42)';
    ctx.beginPath();
    ctx.rect(0, 0, outW, outH);
    ctx.rect(rx, ry, rw, rh);
    ctx.fill('evenodd');

    ctx.strokeStyle = '#E6B980';
    ctx.lineWidth = 2.6;
    ctx.shadowColor = 'rgba(230, 185, 128, 0.85)';
    ctx.shadowBlur = 12;
    ctx.strokeRect(rx, ry, rw, rh);

    if (highlightRecord.pin_screen_norm) {
      const px = highlightRecord.pin_screen_norm[0] * outW;
      const py = highlightRecord.pin_screen_norm[1] * outH;
      ctx.shadowBlur = 0;
      ctx.fillStyle = '#E6B980';
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#121314';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    if (highlightRecord.layer_label_he) {
      ctx.shadowBlur = 0;
      const labelText = highlightRecord.layer_label_he.slice(0, 56);
      ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "Heebo", sans-serif';
      const metrics = ctx.measureText(labelText);
      const bw = Math.min(outW - 12, metrics.width + 18);
      const bh = 22;
      const bx = Math.max(6, Math.min(outW - bw - 6, rx + rw - bw));
      const by = ry > bh + 8 ? ry - bh - 5 : Math.min(outH - bh - 6, ry + rh + 5);

      drawRoundedRectPath(ctx, bx, by, bw, bh, 6);
      ctx.fillStyle = 'rgba(18, 20, 24, 0.92)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(230, 185, 128, 0.75)';
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.fillStyle = '#E6B980';
      ctx.direction = 'rtl';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(labelText, bx + bw * 0.5, by + bh * 0.5);
    }
    ctx.restore();
  }

  return snapCanvas.toDataURL('image/jpeg', jpegQuality);
}
