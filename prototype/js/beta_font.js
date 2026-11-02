// prototype/js/beta_font.js
// Global Beta Typography Switcher shared across /, /wiki, and /catalog.
// Persists selection in localStorage and applies CSS custom properties globally.

export const BETA_FONT_STORAGE_KEY = 'tami70_beta_font';
export const DEFAULT_BETA_FONT_ID = 'ibm_plex';

export const BETA_FONT_OPTIONS = Object.freeze([
  Object.freeze({
    id: 'ibm_plex',
    labelHe: 'IBM Plex · מוזיאלי מודרני',
    shortHe: 'IBM Plex',
    sampleFamily: "'IBM Plex Sans Hebrew', sans-serif"
  }),
  Object.freeze({
    id: 'assistant',
    labelHe: 'Assistant · אוורירי ונקי',
    shortHe: 'Assistant',
    sampleFamily: "'Assistant', sans-serif"
  }),
  Object.freeze({
    id: 'rubik',
    labelHe: 'Rubik · גיאומטרי וחם',
    shortHe: 'Rubik',
    sampleFamily: "'Rubik', sans-serif"
  }),
  Object.freeze({
    id: 'frank_curatorial',
    labelHe: 'Frank Ruhl · אוצרותי ספרותי',
    shortHe: 'Frank Ruhl',
    sampleFamily: "'Frank Ruhl Libre', serif"
  }),
  Object.freeze({
    id: 'fredoka',
    labelHe: 'Fredoka · עגול ועדין',
    shortHe: 'Fredoka',
    sampleFamily: "'Fredoka', sans-serif"
  }),
  Object.freeze({
    id: 'heebo',
    labelHe: 'Heebo · הגופן המקורי',
    shortHe: 'Heebo',
    sampleFamily: "'Heebo', sans-serif"
  })
]);

const VALID_FONT_IDS = new Set(BETA_FONT_OPTIONS.map((opt) => opt.id));

export function getSavedBetaFontId() {
  try {
    const raw = localStorage.getItem(BETA_FONT_STORAGE_KEY);
    if (raw && VALID_FONT_IDS.has(raw)) {
      return raw;
    }
  } catch (_) {
    // Ignore storage privacy restrictions
  }
  return DEFAULT_BETA_FONT_ID;
}

export function applyBetaFont(fontId, { persist = true, emitDiag = false } = {}) {
  const resolved = VALID_FONT_IDS.has(fontId) ? fontId : DEFAULT_BETA_FONT_ID;
  if (typeof document !== 'undefined' && document.documentElement) {
    document.documentElement.setAttribute('data-beta-font', resolved);
  }
  if (persist) {
    try {
      localStorage.setItem(BETA_FONT_STORAGE_KEY, resolved);
    } catch (_) {
      // Ignore storage privacy restrictions
    }
  }
  if (typeof document !== 'undefined') {
    document.querySelectorAll('[data-beta-font-option]').forEach((btn) => {
      const active = btn.getAttribute('data-beta-font-option') === resolved;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
    document.querySelectorAll('select[data-beta-font-select]').forEach((sel) => {
      if (sel.value !== resolved) {
        sel.value = resolved;
      }
    });
  }
  if (emitDiag && typeof window !== 'undefined' && typeof window.__sendDiag === 'function') {
    window.__sendDiag('beta_font_changed', { fontId: resolved });
  }
  return resolved;
}

export function mountBetaFontChipSelector(containerEl, { blurOnPick = true } = {}) {
  const current = applyBetaFont(getSavedBetaFontId(), { persist: false, emitDiag: false });
  if (!containerEl) return current;

  containerEl.innerHTML = '';
  for (const opt of BETA_FONT_OPTIONS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `beta-font-chip${opt.id === current ? ' active' : ''}`;
    btn.setAttribute('data-beta-font-option', opt.id);
    btn.setAttribute('aria-pressed', opt.id === current ? 'true' : 'false');
    btn.style.fontFamily = opt.sampleFamily;
    btn.textContent = opt.labelHe;
    btn.addEventListener('click', () => {
      applyBetaFont(opt.id, { persist: true, emitDiag: true });
      if (blurOnPick && typeof btn.blur === 'function') {
        btn.blur();
      }
    });
    containerEl.appendChild(btn);
  }
  return current;
}

export function mountPortalHeaderFontPicker(actionsContainerEl) {
  const current = applyBetaFont(getSavedBetaFontId(), { persist: false, emitDiag: false });
  if (!actionsContainerEl) return current;
  if (actionsContainerEl.querySelector('[data-beta-font-select-wrap]')) return current;

  const wrap = document.createElement('label');
  wrap.className = 'portal-font-picker-wrap';
  wrap.setAttribute('data-beta-font-select-wrap', '1');
  wrap.title = 'בחירת גופן לבחינה בבטא (חל על כל העמודים)';

  const labelSpan = document.createElement('span');
  labelSpan.className = 'portal-font-picker-label';
  labelSpan.textContent = 'גופן:';

  const select = document.createElement('select');
  select.className = 'portal-font-picker-select';
  select.setAttribute('data-beta-font-select', '1');
  select.setAttribute('aria-label', 'בחירת גופן תצוגה בבטא');

  for (const opt of BETA_FONT_OPTIONS) {
    const o = document.createElement('option');
    o.value = opt.id;
    o.textContent = opt.labelHe;
    if (opt.id === current) o.selected = true;
    select.appendChild(o);
  }

  select.addEventListener('change', () => {
    applyBetaFont(select.value, { persist: true, emitDiag: true });
    select.blur();
  });

  wrap.appendChild(labelSpan);
  wrap.appendChild(select);
  actionsContainerEl.prepend(wrap);
  return current;
}
