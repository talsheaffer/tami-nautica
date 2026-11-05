// prototype/js/collage_renderer.js
// Pure client-side justified-row gallery collage & composite-strip renderer.
// Preserves every photo's natural aspect ratio (zero face-cropping or stretching).

const COLLAGE_THEMES = Object.freeze({
  dark: {
    bg: '#15171C',
    cardBg: '#0E1014',
    border: 'rgba(196, 155, 113, 0.45)',
    titleColor: '#F5EFE8',
    subtitleColor: '#C49B71'
  },
  paper: {
    bg: '#F4EDE2',
    cardBg: '#EAE0D2',
    border: 'rgba(122, 96, 68, 0.35)',
    titleColor: '#231F1C',
    subtitleColor: '#7A5C3E'
  }
});

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`לא ניתן לטעון את התמונה: ${src}`));
    img.src = src;
  });
}

function partitionIntoRows(count) {
  if (count <= 0) return [];
  if (count === 1) return [1];
  if (count === 2) return [2];
  if (count === 3) return [2, 1];
  if (count === 4) return [2, 2];
  if (count === 5) return [2, 3];
  if (count === 6) return [3, 3];
  const rows = [];
  let rem = count;
  while (rem > 0) {
    if (rem === 4) {
      rows.push(2, 2);
      break;
    }
    const take = rem >= 3 ? 3 : rem;
    rows.push(take);
    rem -= take;
  }
  return rows;
}

/**
 * Renders a justified multi-photo collage onto a canvas.
 * @param {HTMLCanvasElement} canvas
 * @param {Array<{imgUrl: string, title?: string}>} items
 * @param {object} studioCfg - CreativeStudioConfig from /api/beta-config
 * @param {object} opts - { titleText, theme: 'dark' | 'paper', targetWidthPx, footerText }
 */
export async function renderCollageToCanvas(canvas, items, studioCfg, opts = {}) {
  if (!items || items.length === 0) {
    throw new Error('לא נבחרו תמונות ליצירת הקולאז׳.');
  }

  const loaded = await Promise.all(
    items.map(async (item) => {
      const img = await loadImageElement(item.imgUrl);
      const w = img.naturalWidth || img.width || 800;
      const h = img.naturalHeight || img.height || 600;
      return { item, img, aspect: Math.max(0.35, Math.min(2.8, w / h)) };
    })
  );

  const canvasW = opts.targetWidthPx || studioCfg.collage_canvas_width_px;
  const scale = canvasW / studioCfg.collage_canvas_width_px;
  const pad = Math.round(studioCfg.collage_padding_px * scale);
  const gap = Math.round(studioCfg.collage_gap_px * scale);
  const titleText = String(opts.titleText || '').trim();
  const footerText = String(opts.footerText || '').trim();
  const headerH = titleText ? Math.round(studioCfg.collage_header_height_px * scale) : 0;
  const footerH = footerText ? Math.round(48 * scale) : 0;
  const theme = COLLAGE_THEMES[opts.theme] || COLLAGE_THEMES.dark;

  const innerW = canvasW - pad * 2;
  const rowCounts = partitionIntoRows(loaded.length);
  const rowLayouts = [];
  let cursor = 0;
  let totalGridH = 0;

  for (const rCount of rowCounts) {
    const slice = loaded.slice(cursor, cursor + rCount);
    cursor += rCount;
    const sumAspect = slice.reduce((acc, s) => acc + s.aspect, 0);
    const availW = innerW - gap * (slice.length - 1);
    const rowH = Math.round(Math.max(180 * scale, Math.min(760 * scale, availW / sumAspect)));
    const cells = [];
    let xOff = pad;
    slice.forEach((entry, idx) => {
      const cellW =
        idx === slice.length - 1
          ? pad + innerW - xOff
          : Math.round(rowH * entry.aspect);
      cells.push({ entry, x: xOff, w: cellW, h: rowH });
      xOff += cellW + gap;
    });
    rowLayouts.push({ rowH, cells });
    totalGridH += rowH;
  }
  totalGridH += gap * Math.max(0, rowLayouts.length - 1);

  const canvasH = pad * 2 + headerH + totalGridH + footerH;
  canvas.width = canvasW;
  canvas.height = canvasH;

  const ctx = canvas.getContext('2d');
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, canvasW, canvasH);

  let yCursor = pad;
  if (titleText) {
    ctx.save();
    ctx.direction = 'rtl';
    ctx.textAlign = 'center';
    ctx.fillStyle = theme.titleColor;
    ctx.font = `700 ${Math.round(38 * scale)}px Heebo, Assistant, -apple-system, sans-serif`;
    ctx.fillText(titleText, canvasW / 2, yCursor + Math.round(52 * scale), innerW);
    yCursor += headerH;
    ctx.restore();
  }

  for (const row of rowLayouts) {
    for (const cell of row.cells) {
      const { img } = cell.entry;
      ctx.save();
      ctx.fillStyle = theme.cardBg;
      ctx.fillRect(cell.x, yCursor, cell.w, cell.h);

      // Center-cover inside the aspect-matched cell (minimal rounding delta only)
      const imgAspect = (img.naturalWidth || img.width) / (img.naturalHeight || img.height);
      const cellAspect = cell.w / cell.h;
      let sx = 0;
      let sy = 0;
      let sw = img.naturalWidth || img.width;
      let sh = img.naturalHeight || img.height;
      if (imgAspect > cellAspect) {
        sw = sh * cellAspect;
        sx = ((img.naturalWidth || img.width) - sw) / 2;
      } else {
        sh = sw / cellAspect;
        sy = ((img.naturalHeight || img.height) - sh) / 2;
      }
      ctx.drawImage(img, sx, sy, sw, sh, cell.x, yCursor, cell.w, cell.h);

      ctx.strokeStyle = theme.border;
      ctx.lineWidth = Math.max(1, Math.round(2 * scale));
      ctx.strokeRect(cell.x + 0.5, yCursor + 0.5, cell.w - 1, cell.h - 1);
      ctx.restore();
    }
    yCursor += row.rowH + gap;
  }

  if (footerText) {
    ctx.save();
    ctx.direction = 'rtl';
    ctx.textAlign = 'center';
    ctx.fillStyle = theme.subtitleColor;
    ctx.font = `500 ${Math.round(20 * scale)}px Heebo, Assistant, -apple-system, sans-serif`;
    ctx.fillText(footerText, canvasW / 2, canvasH - pad + Math.round(6 * scale), innerW);
    ctx.restore();
  }

  const jpegQuality = studioCfg.collage_jpeg_quality || 0.92;
  return {
    width: canvasW,
    height: canvasH,
    jpegDataUrl: canvas.toDataURL('image/jpeg', jpegQuality)
  };
}
