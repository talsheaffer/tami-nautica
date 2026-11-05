// prototype/js/theme.js
// UI color scheme switcher (sidebar). Persists to localStorage; applies <html data-theme>.

export const THEME_STORAGE_KEY = 'tami70_theme';
export const DEFAULT_THEME_ID = 'ocean';

export const THEME_OPTIONS = Object.freeze([
  Object.freeze({ id: 'ocean', labelHe: 'אוקיינוס', accent: '#6EA8C8', bg: '#0F1316' }),
  Object.freeze({ id: 'classic', labelHe: 'קלאסי · שחור וחול', accent: '#C49B71', bg: '#121314' }),
  Object.freeze({ id: 'forest', labelHe: 'יער', accent: '#8AAA78', bg: '#101412' }),
  Object.freeze({ id: 'plum', labelHe: 'שזיף', accent: '#B884A8', bg: '#141114' }),
  Object.freeze({ id: 'ember', labelHe: 'גחלת', accent: '#D67054', bg: '#151110' }),
  Object.freeze({ id: 'mono', labelHe: 'מונו', accent: '#C8C8C8', bg: '#0F0F10' })
]);

const VALID = new Set(THEME_OPTIONS.map((t) => t.id));

export function getSavedThemeId() {
  try {
    const raw = localStorage.getItem(THEME_STORAGE_KEY);
    if (raw && VALID.has(raw)) return raw;
  } catch (_) { /* storage blocked */ }
  return DEFAULT_THEME_ID;
}

export function applyTheme(id, { persist = true } = {}) {
  const resolved = VALID.has(id) ? id : DEFAULT_THEME_ID;
  document.documentElement.setAttribute('data-theme', resolved);
  if (persist) {
    try { localStorage.setItem(THEME_STORAGE_KEY, resolved); } catch (_) { /* storage blocked */ }
  }
  document.querySelectorAll('[data-theme-option]').forEach((btn) => {
    const active = btn.getAttribute('data-theme-option') === resolved;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
  return resolved;
}

export function mountThemeSelector(containerEl) {
  const current = applyTheme(getSavedThemeId(), { persist: false });
  if (!containerEl) return current;
  containerEl.innerHTML = '';
  for (const opt of THEME_OPTIONS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `theme-swatch${opt.id === current ? ' active' : ''}`;
    btn.setAttribute('data-theme-option', opt.id);
    btn.setAttribute('aria-pressed', opt.id === current ? 'true' : 'false');
    btn.setAttribute('aria-label', opt.labelHe);
    btn.title = opt.labelHe;
    btn.style.setProperty('--sw-accent', opt.accent);
    btn.style.setProperty('--sw-bg', opt.bg);
    btn.addEventListener('click', () => { applyTheme(opt.id); btn.blur(); });
    containerEl.appendChild(btn);
  }
  return current;
}

// ---------- UI style (chrome design) ----------
export const UI_STYLE_STORAGE_KEY = 'tami70_ui_style';
export const DEFAULT_UI_STYLE_ID = 'glass-light';

export const UI_STYLE_OPTIONS = Object.freeze([
  Object.freeze({ id: 'classic', labelHe: 'קלאסי' }),
  Object.freeze({ id: 'glass', labelHe: 'זכוכית · כהה' }),
  Object.freeze({ id: 'glass-light', labelHe: 'זכוכית · בהיר' }),
  Object.freeze({ id: 'quiet', labelHe: 'שקט · מינימלי' }),
  Object.freeze({ id: 'museum', labelHe: 'מוזיאלי · עריכה' })
]);

const VALID_UI = new Set(UI_STYLE_OPTIONS.map((o) => o.id));

export function getSavedUiStyleId() {
  try {
    const raw = localStorage.getItem(UI_STYLE_STORAGE_KEY);
    if (raw && VALID_UI.has(raw)) return raw;
  } catch (_) { /* storage blocked */ }
  return DEFAULT_UI_STYLE_ID;
}

export function applyUiStyle(id, { persist = true } = {}) {
  const resolved = VALID_UI.has(id) ? id : DEFAULT_UI_STYLE_ID;
  document.documentElement.setAttribute('data-ui', resolved);
  if (persist) {
    try { localStorage.setItem(UI_STYLE_STORAGE_KEY, resolved); } catch (_) { /* storage blocked */ }
  }
  document.querySelectorAll('[data-ui-option]').forEach((btn) => {
    const active = btn.getAttribute('data-ui-option') === resolved;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
  return resolved;
}

export function mountUiStyleSelector(containerEl) {
  const current = applyUiStyle(getSavedUiStyleId(), { persist: false });
  if (!containerEl) return current;
  containerEl.innerHTML = '';
  for (const opt of UI_STYLE_OPTIONS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `ui-style-chip${opt.id === current ? ' active' : ''}`;
    btn.setAttribute('data-ui-option', opt.id);
    btn.setAttribute('aria-pressed', opt.id === current ? 'true' : 'false');
    btn.textContent = opt.labelHe;
    btn.addEventListener('click', () => { applyUiStyle(opt.id); btn.blur(); });
    containerEl.appendChild(btn);
  }
  return current;
}
