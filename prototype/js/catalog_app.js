// prototype/js/catalog_app.js
// Client application for the Family Photo Archive, Custom Tagging, Multi-Select Creative Studio
// (Collage / Fuse into One Scene / Open Verbal Description), and Drop-Folder Uploader (/catalog).

import { BetaFeedbackOverlay } from './feedback_overlay.js';
import { renderCollageToCanvas } from './collage_renderer.js';
import { applyBetaFont, getSavedBetaFontId, mountPortalHeaderFontPicker } from './beta_font.js';

applyBetaFont(getSavedBetaFontId(), { persist: false, emitDiag: false });

const AUTHOR_STORAGE_KEY = 'tami70_beta_author';

const PERSON_LABELS_HE = Object.freeze({
  "Tamar 'Tami' Sheaffer": 'תמי (תמר שפר)',
  'Gad Sheaffer': 'גדי שפר',
  'Guy Sheaffer': 'גיא שפר',
  'Tal Sheaffer': 'טל שפר',
  "Ya'ara": 'יערה',
  Carmel: 'כרמל',
  'Michael Fried': 'מיכאל פריד',
  'Ahuva Fried': 'אהובה פריד',
  "Er'ela Sadan": 'אראלה סדן',
  'Ada Wertheimer': 'עדה ורטהיימר',
  'Naama': 'נעמה',
  'Beni': 'בני',
  'Peri': 'פרי'
});

const SOURCE_LABELS_HE = Object.freeze({
  drop_folder_imports: '📥 העלאת משפחה',
  wordpress_blog: '📰 בלוג דרכי עצים',
  darchei_etzim_pdfs: '📕 ספרי דרכי עצים',
  portfolio_site: '🎨 אתר תמר שפר',
  binyamin_gallery: '🖼️ גלריה בנימין',
  museum_press: '🏛️ מוזיאון רמת גן / עיתונות'
});

function escapeHtml(raw) {
  return String(raw || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error(`שגיאה בקריאת הקובץ ${file.name}`));
    reader.readAsDataURL(file);
  });
}

function buildItemImageUrl(item) {
  // display_rel_path points at the upright derivative when the catalog declares a display rotation.
  if (!item.display_rel_path) {
    throw new Error(`Catalog item ${item.id} is missing display_rel_path — rebuild with scripts/build_catalog.py`);
  }
  return `/${item.display_rel_path.split('/').map(encodeURIComponent).join('/')}`;
}

class CatalogPortalApp {
  constructor() {
    this.items = [];
    this.pendingFiles = [];
    this.activeFilter = 'all';
    this.selectedPeopleFilters = new Set();
    this._personLabelMap = new Map(Object.entries(PERSON_LABELS_HE));
    this.searchQuery = '';
    this.editingItem = null;

    this.selectedMediaIds = [];
    this.activeStudioModeId = 'collage';
    this.activeCollageTheme = 'dark';
    this.lastRenderedCollageDataUrl = null;
    this._collageRenderTimer = null;

    this.countPillEl = document.getElementById('catalog-count-pill');
    this.fileInput = document.getElementById('upload-file-input');
    this.dropzone = document.getElementById('upload-dropzone');
    this.previewStrip = document.getElementById('upload-preview-strip');
    this.metaFieldsWrap = document.getElementById('upload-metadata-fields');
    this.uploadAuthorChips = document.getElementById('upload-author-chips');
    this.uploadAuthorInput = document.getElementById('upload-author-input');
    this.uploadEraInput = document.getElementById('upload-era-input');
    this.uploadPlaceChips = document.getElementById('upload-place-chips');
    this.uploadCustomPlaceInput = document.getElementById('upload-custom-place-input');
    this.uploadCustomPlaceAddBtn = document.getElementById('upload-custom-place-add');
    this.uploadPeopleChips = document.getElementById('upload-people-chips');
    this.uploadCustomPersonInput = document.getElementById('upload-custom-person-input');
    this.uploadCustomPersonAddBtn = document.getElementById('upload-custom-person-add');
    this.uploadCaptionInput = document.getElementById('upload-caption-input');
    this.uploadStatusMsg = document.getElementById('upload-status-msg');
    this.uploadSubmitBtn = document.getElementById('upload-submit-btn');
    this.uploadClearBtn = document.getElementById('upload-clear-btn');

    this.searchInput = document.getElementById('catalog-search-input');
    this.peopleFilterChipsEl = document.getElementById('catalog-people-filter-chips');
    this.gridEl = document.getElementById('catalog-photo-grid');

    this.multiSelectBar = document.getElementById('multi-select-bar');
    this.multiSelectCountLabel = document.getElementById('multi-select-count-label');
    this.multiSelectThumbs = document.getElementById('multi-select-thumbs');
    this.multiSelectClearBtn = document.getElementById('multi-select-clear-btn');
    this.multiSelectOpenStudioBtn = document.getElementById('multi-select-open-studio-btn');

    this.studioBackdrop = document.getElementById('creative-studio-backdrop');
    this.studioSubtitleEl = document.getElementById('studio-selected-subtitle');
    this.studioModeCardsEl = document.getElementById('studio-mode-cards');
    this.studioAuthorChips = document.getElementById('studio-author-chips');
    this.studioAuthorInput = document.getElementById('studio-author-input');
    this.studioCollageControls = document.getElementById('studio-collage-controls');
    this.studioThemeChips = document.getElementById('studio-theme-chips');
    this.studioCollageTitleInput = document.getElementById('studio-collage-title-input');
    this.studioPromptLabel = document.getElementById('studio-prompt-label');
    this.studioPromptInput = document.getElementById('studio-prompt-input');
    this.studioPreviewCanvas = document.getElementById('studio-preview-canvas');
    this.studioStatusMsg = document.getElementById('studio-status-msg');
    this.studioDownloadBtn = document.getElementById('studio-download-collage-btn');
    this.studioSaveCatalogBtn = document.getElementById('studio-save-catalog-btn');
    this.studioSubmitProposalBtn = document.getElementById('studio-submit-proposal-btn');

    this.editBackdrop = document.getElementById('edit-meta-backdrop');
    this.editIdEl = document.getElementById('edit-meta-id');
    this.editTitleInput = document.getElementById('edit-meta-title');
    this.editCaptionInput = document.getElementById('edit-meta-caption');
    this.editEraInput = document.getElementById('edit-meta-era');
    this.editPlaceChips = document.getElementById('edit-place-chips');
    this.editCustomPlaceInput = document.getElementById('edit-custom-place-input');
    this.editCustomPlaceAddBtn = document.getElementById('edit-custom-place-add');
    this.editPeopleChips = document.getElementById('edit-people-chips');
    this.editCustomPersonInput = document.getElementById('edit-custom-person-input');
    this.editCustomPersonAddBtn = document.getElementById('edit-custom-person-add');

    this.feedbackOverlay = new BetaFeedbackOverlay(null, {
      defaultSceneId: 'catalog',
      getTargetRef: () => {
        if (this.selectedPeopleFilters.size > 0) {
          return `${this.activeFilter}|people:${Array.from(this.selectedPeopleFilters).join(',')}`;
        }
        return this.activeFilter;
      },
      onNavigateTargetRef: async (targetRef) => {
        if (targetRef && targetRef.startsWith('IMG_')) {
          this.focusMediaCard(targetRef);
        }
      },
      onReady: (overlay) => {
        this._populatePresetChipsFromConfig(overlay.config);
      }
    });

    this.init();
  }

  async init() {
    this._bindEvents();
    await this.loadCatalog();

    const mediaParam = new URLSearchParams(window.location.search).get('media');
    if (mediaParam) {
      this.focusMediaCard(mediaParam);
    }
  }

  _bindAuthorPresetRow(chipsContainer, inputEl) {
    const cfg = this.feedbackOverlay && this.feedbackOverlay.config;
    if (!cfg || !chipsContainer || !inputEl) return;
    const savedAuthor = localStorage.getItem(AUTHOR_STORAGE_KEY) || '';
    if (!inputEl.value) {
      inputEl.value = savedAuthor;
    }
    chipsContainer.innerHTML = (cfg.reviewer_presets || [])
      .map(
        (p) =>
          `<button type="button" class="beta-chip ${
            inputEl.value === p.label_he ? 'active' : ''
          }" data-up-author="${escapeHtml(p.label_he)}">${escapeHtml(p.label_he)}</button>`
      )
      .join('');
    chipsContainer.querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', () => {
        const val = btn.getAttribute('data-up-author') || '';
        inputEl.value = val;
        localStorage.setItem(AUTHOR_STORAGE_KEY, val);
        chipsContainer.querySelectorAll('button').forEach((b) => {
          b.classList.toggle('active', b === btn);
        });
      });
    });
  }

  _populatePresetChipsFromConfig(cfg) {
    if (!cfg) return;
    this._bindAuthorPresetRow(this.uploadAuthorChips, this.uploadAuthorInput);
    this._bindAuthorPresetRow(this.studioAuthorChips, this.studioAuthorInput);

    const placePresets = (cfg.uploads && cfg.uploads.place_presets) || [];
    const peoplePresets = (cfg.uploads && cfg.uploads.people_presets) || [];
    for (const p of peoplePresets) {
      if (p && p.canonical && p.label_he) {
        const cleanLabel = String(p.label_he).replace(/^[^\p{L}\p{N}]+\s*/u, '').trim();
        if (cleanLabel) {
          this._personLabelMap.set(p.canonical, cleanLabel);
        }
      }
    }

    this._renderMultiSelectChips(this.uploadPlaceChips, placePresets);
    this._renderMultiSelectChips(this.uploadPeopleChips, peoplePresets);
    this._renderMultiSelectChips(this.editPlaceChips, placePresets);
    this._renderMultiSelectChips(this.editPeopleChips, peoplePresets);
    this._renderPeopleFilterBar();
    this._renderStudioModeCards();
  }

  _renderMultiSelectChips(container, presets) {
    container.innerHTML = presets
      .map(
        (p) =>
          `<button type="button" class="beta-chip" data-canonical="${escapeHtml(
            p.canonical
          )}">${escapeHtml(p.label_he)}</button>`
      )
      .join('');
    container.querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', () => btn.classList.toggle('active'));
    });
  }

  _addCustomChip(container, rawValue, prefixIcon = '') {
    const cleaned = String(rawValue || '').trim();
    if (!cleaned) return;
    const existing = Array.from(container.querySelectorAll('button')).find(
      (b) => (b.getAttribute('data-canonical') || '').toLowerCase() === cleaned.toLowerCase()
    );
    if (existing) {
      existing.classList.add('active');
      return;
    }
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'beta-chip active custom-chip';
    btn.setAttribute('data-canonical', cleaned);
    btn.setAttribute('data-custom-chip', '1');
    btn.textContent = `${prefixIcon ? prefixIcon + ' ' : ''}${cleaned}`;
    btn.addEventListener('click', () => btn.classList.toggle('active'));
    container.appendChild(btn);
  }

  _commitPendingCustomInput(container, inputEl, prefixIcon) {
    if (!inputEl) return;
    const val = inputEl.value.trim();
    if (val) {
      val
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .forEach((part) => this._addCustomChip(container, part, prefixIcon));
      inputEl.value = '';
    }
  }

  _getSelectedCanonicals(container) {
    return Array.from(container.querySelectorAll('button.active'))
      .map((b) => b.getAttribute('data-canonical') || '')
      .filter(Boolean);
  }

  _setSelectedCanonicals(container, values, prefixIcon = '') {
    // Remove previously appended custom chips first so preset order stays clean
    container.querySelectorAll('button[data-custom-chip="1"]').forEach((el) => el.remove());
    const presetSet = new Set();
    container.querySelectorAll('button').forEach((b) => {
      presetSet.add(b.getAttribute('data-canonical') || '');
    });
    const wanted = new Set(values || []);
    container.querySelectorAll('button').forEach((b) => {
      b.classList.toggle('active', wanted.has(b.getAttribute('data-canonical')));
    });
    for (const v of values || []) {
      if (v && !presetSet.has(v)) {
        this._addCustomChip(container, v, prefixIcon);
      }
    }
  }

  _bindCustomChipAdder(container, inputEl, addBtn, prefixIcon) {
    if (!inputEl || !addBtn) return;
    const commit = () => {
      this._commitPendingCustomInput(container, inputEl, prefixIcon);
      inputEl.focus();
    };
    addBtn.addEventListener('click', commit);
    inputEl.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter') {
        ev.preventDefault();
        commit();
      }
    });
  }

  _bindEvents() {
    this.dropzone.addEventListener('click', () => this.fileInput.click());
    this.dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.dropzone.classList.add('dragover');
    });
    this.dropzone.addEventListener('dragleave', () => {
      this.dropzone.classList.remove('dragover');
    });
    this.dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      this.dropzone.classList.remove('dragover');
      if (e.dataTransfer && e.dataTransfer.files) {
        this.handleSelectedFiles(e.dataTransfer.files);
      }
    });

    this.fileInput.addEventListener('change', () => {
      if (this.fileInput.files) {
        this.handleSelectedFiles(this.fileInput.files);
      }
    });

    this._bindCustomChipAdder(
      this.uploadPlaceChips,
      this.uploadCustomPlaceInput,
      this.uploadCustomPlaceAddBtn,
      '📍'
    );
    this._bindCustomChipAdder(
      this.uploadPeopleChips,
      this.uploadCustomPersonInput,
      this.uploadCustomPersonAddBtn,
      '👤'
    );
    this._bindCustomChipAdder(
      this.editPlaceChips,
      this.editCustomPlaceInput,
      this.editCustomPlaceAddBtn,
      '📍'
    );
    this._bindCustomChipAdder(
      this.editPeopleChips,
      this.editCustomPersonInput,
      this.editCustomPersonAddBtn,
      '👤'
    );

    this.uploadClearBtn.addEventListener('click', () => this.clearPendingUploads());
    this.uploadSubmitBtn.addEventListener('click', () => this.submitUploads());

    this.searchInput.addEventListener('input', () => {
      this.searchQuery = this.searchInput.value.trim().toLowerCase();
      this.renderGrid();
    });

    document.querySelectorAll('#catalog-filter-chips button').forEach((btn) => {
      btn.addEventListener('click', () => {
        this.activeFilter = btn.getAttribute('data-cat-filter') || 'all';
        document.querySelectorAll('#catalog-filter-chips button').forEach((b) => {
          b.classList.toggle('active', b === btn);
        });
        this.renderGrid();
      });
    });

    this.multiSelectClearBtn.addEventListener('click', () => this.clearMediaSelection());
    this.multiSelectOpenStudioBtn.addEventListener('click', () => this.openStudioModal());

    document.getElementById('studio-modal-close').addEventListener('click', () => this.closeStudioModal());
    document.getElementById('studio-cancel-btn').addEventListener('click', () => this.closeStudioModal());

    this.studioThemeChips.querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', () => {
        this.activeCollageTheme = btn.getAttribute('data-collage-theme') || 'dark';
        this.studioThemeChips.querySelectorAll('button').forEach((b) => {
          b.classList.toggle('active', b === btn);
        });
        this._scheduleCollagePreviewRender();
      });
    });

    this.studioCollageTitleInput.addEventListener('input', () => {
      this._scheduleCollagePreviewRender();
    });

    this.studioDownloadBtn.addEventListener('click', () => this.downloadCollageJpg());
    this.studioSaveCatalogBtn.addEventListener('click', () => this.saveCollageToCatalog());
    this.studioSubmitProposalBtn.addEventListener('click', () => this.submitStudioProposal());

    document.getElementById('edit-meta-close').addEventListener('click', () => this.closeEditModal());
    document.getElementById('edit-meta-cancel').addEventListener('click', () => this.closeEditModal());
    document.getElementById('edit-meta-save').addEventListener('click', () => this.saveEditModal());
  }

  _getStudioConfig() {
    const cfg = this.feedbackOverlay && this.feedbackOverlay.config;
    if (cfg && cfg.creative_studio) {
      return cfg.creative_studio;
    }
    return {
      max_selected_photos: 12,
      collage_canvas_width_px: 1600,
      collage_padding_px: 36,
      collage_gap_px: 20,
      collage_header_height_px: 108,
      collage_jpeg_quality: 0.92,
      modes: []
    };
  }

  toggleMediaSelection(mediaId) {
    const idx = this.selectedMediaIds.indexOf(mediaId);
    if (idx !== -1) {
      this.selectedMediaIds.splice(idx, 1);
    } else {
      const maxSel = this._getStudioConfig().max_selected_photos;
      if (this.selectedMediaIds.length >= maxSel) {
        window.alert(`ניתן לבחור עד ${maxSel} תמונות בכל יצירה.`);
        return;
      }
      this.selectedMediaIds.push(mediaId);
    }
    this.renderMultiSelectBar();
    this._syncSelectedCardClasses();
  }

  clearMediaSelection() {
    this.selectedMediaIds = [];
    this.renderMultiSelectBar();
    this._syncSelectedCardClasses();
  }

  _syncSelectedCardClasses() {
    const selMap = new Map(this.selectedMediaIds.map((id, i) => [id, i + 1]));
    this.gridEl.querySelectorAll('.portal-photo-card').forEach((card) => {
      const id = card.getAttribute('data-media-id') || '';
      const order = selMap.get(id);
      card.classList.toggle('selected-for-studio', Boolean(order));
      const btn = card.querySelector('.photo-select-btn');
      if (btn) {
        btn.classList.toggle('selected', Boolean(order));
        btn.textContent = order ? `✓ נבחרה (#${order})` : '☑️ בחר ליצירה';
      }
    });
  }

  renderMultiSelectBar() {
    const count = this.selectedMediaIds.length;
    if (count === 0) {
      this.multiSelectBar.classList.add('hidden');
      this.multiSelectThumbs.innerHTML = '';
      return;
    }
    this.multiSelectBar.classList.remove('hidden');
    this.multiSelectCountLabel.textContent =
      count === 1 ? 'נבחרה תמונה אחת' : `נבחרו ${count} תמונות`;

    const byId = new Map(this.items.map((it) => [it.id, it]));
    this.multiSelectThumbs.innerHTML = this.selectedMediaIds
      .map((id) => {
        const it = byId.get(id);
        if (!it) return '';
        const url = buildItemImageUrl(it);
        return `
          <div class="multi-select-thumb-item" title="${escapeHtml(it.title)}">
            <img src="${url}" alt="${escapeHtml(it.title)}" />
            <button type="button" class="multi-select-thumb-remove" data-remove-sel="${escapeHtml(
              id
            )}" aria-label="הסר מבחירה">✕</button>
          </div>
        `;
      })
      .join('');

    this.multiSelectThumbs.querySelectorAll('[data-remove-sel]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const id = btn.getAttribute('data-remove-sel');
        if (id) this.toggleMediaSelection(id);
      });
    });
  }

  _getSelectedItems() {
    const byId = new Map(this.items.map((it) => [it.id, it]));
    return this.selectedMediaIds.map((id) => byId.get(id)).filter(Boolean);
  }

  _renderStudioModeCards() {
    const modes = this._getStudioConfig().modes || [];
    if (!this.studioModeCardsEl || modes.length === 0) return;

    this.studioModeCardsEl.innerHTML = modes
      .map(
        (m) => `
        <button type="button" class="studio-mode-card ${
          m.mode_id === this.activeStudioModeId ? 'active' : ''
        }" data-studio-mode="${escapeHtml(m.mode_id)}">
          <div class="studio-mode-title">${escapeHtml(m.icon)} ${escapeHtml(m.label_he)}</div>
          <div class="studio-mode-sub">${escapeHtml(m.subtitle_he)}</div>
        </button>
      `
      )
      .join('');

    this.studioModeCardsEl.querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', () => {
        this.activeStudioModeId = btn.getAttribute('data-studio-mode') || 'collage';
        this.studioModeCardsEl.querySelectorAll('button').forEach((b) => {
          b.classList.toggle('active', b === btn);
        });
        this._syncStudioModeUi();
      });
    });
  }

  _syncStudioModeUi() {
    const modes = this._getStudioConfig().modes || [];
    const activeSpec = modes.find((m) => m.mode_id === this.activeStudioModeId) || modes[0];
    const isCollage = this.activeStudioModeId === 'collage';

    this.studioCollageControls.classList.toggle('hidden', !isCollage);
    this.studioDownloadBtn.style.display = isCollage ? 'inline-flex' : 'none';
    this.studioSaveCatalogBtn.style.display = isCollage ? 'inline-flex' : 'none';

    if (isCollage) {
      this.studioPromptLabel.textContent = 'הערה / סיפור נלווה לקולאז׳ (אופציונלי — לשיתוף בתיבת המשוב או בקטלוג)';
      this.studioPromptInput.placeholder =
        'אופציונלי: הוסיפו כיתוב או הקשר למשפחה (או הורידו/שמרו את הקולאז׳ ישירות למטה)...';
      this.studioSubmitProposalBtn.textContent = '💬 שתף גם בתיבת ההצעות';
    } else if (this.activeStudioModeId === 'fuse_scene') {
      this.studioPromptLabel.textContent = '🏛️ איך למזג את התמונות שנבחרו לסצנה מרחבית אחת?';
      this.studioPromptInput.placeholder =
        (activeSpec && activeSpec.placeholder_he) ||
        'תארו כיצד לחבר את התמונות לסצנה אחת בסיור...';
      this.studioSubmitProposalBtn.textContent = '🚀 שלח בקשה למיזוג סצנה';
    } else {
      this.studioPromptLabel.textContent = '✍️ תיאור מילולי פתוח — מה תרצו שניצור מהתמונות שנבחרו?';
      this.studioPromptInput.placeholder =
        (activeSpec && activeSpec.placeholder_he) ||
        'כתבו בחופשיות כל רעיון, סיפור מחבר או בקשה פתוחה...';
      this.studioSubmitProposalBtn.textContent = '🚀 שלח רעיון פתוח למשפחה';
    }

    this._scheduleCollagePreviewRender();
  }

  openStudioModal() {
    const selected = this._getSelectedItems();
    if (selected.length === 0) return;

    this._bindAuthorPresetRow(this.studioAuthorChips, this.studioAuthorInput);
    this._renderStudioModeCards();
    this.studioSubtitleEl.textContent = `נבחרו ${selected.length} תמונות: ${selected
      .map((s) => s.id)
      .join(', ')}`;
    this.studioStatusMsg.classList.add('hidden');
    this.studioBackdrop.classList.remove('hidden');
    this._syncStudioModeUi();
  }

  closeStudioModal() {
    this.studioBackdrop.classList.add('hidden');
  }

  _showStudioStatus(msg, isError = false) {
    this.studioStatusMsg.textContent = msg;
    this.studioStatusMsg.className = `beta-status-msg ${isError ? 'error' : 'success'}`;
  }

  _scheduleCollagePreviewRender() {
    if (this._collageRenderTimer) {
      clearTimeout(this._collageRenderTimer);
    }
    this._collageRenderTimer = setTimeout(() => {
      this._renderPreviewCanvasNow();
    }, 90);
  }

  async _renderPreviewCanvasNow() {
    const selected = this._getSelectedItems();
    if (selected.length === 0 || this.studioBackdrop.classList.contains('hidden')) return;
    try {
      const studioCfg = this._getStudioConfig();
      const isCollage = this.activeStudioModeId === 'collage';
      const titleText = isCollage ? this.studioCollageTitleInput.value.trim() : '';
      const res = await renderCollageToCanvas(
        this.studioPreviewCanvas,
        selected.map((it) => ({ imgUrl: buildItemImageUrl(it), title: it.title })),
        studioCfg,
        {
          titleText,
          theme: isCollage ? this.activeCollageTheme : 'dark',
          footerText: isCollage && titleText ? 'מתוך הארכיון המשפחתי — תמר (תמי) שפר · 70' : ''
        }
      );
      this.lastRenderedCollageDataUrl = res.jpegDataUrl;
    } catch (err) {
      this._showStudioStatus(`שגיאה בעיבוד תצוגה מקדימה: ${err.message}`, true);
    }
  }

  async downloadCollageJpg() {
    await this._renderPreviewCanvasNow();
    if (!this.lastRenderedCollageDataUrl) return;
    const a = document.createElement('a');
    const slug = (this.studioCollageTitleInput.value.trim() || 'tami_family_collage')
      .replace(/\s+/g, '_')
      .slice(0, 40);
    a.href = this.lastRenderedCollageDataUrl;
    a.download = `${slug}.jpg`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    this._showStudioStatus('הקולאז׳ הורד בהצלחה!', false);
  }

  async saveCollageToCatalog() {
    const author = this.studioAuthorInput.value.trim();
    if (!author) {
      this._showStudioStatus('נא לבחור או להזין את שמכם לפני שמירת הקולאז׳ בקטלוג.', true);
      this.studioAuthorInput.focus();
      return;
    }
    localStorage.setItem(AUTHOR_STORAGE_KEY, author);

    await this._renderPreviewCanvasNow();
    if (!this.lastRenderedCollageDataUrl) return;

    const selected = this._getSelectedItems();
    const unionPlaces = Array.from(new Set(selected.flatMap((it) => it.places || [])));
    const unionPeople = Array.from(new Set(selected.flatMap((it) => it.people || [])));
    const customTitle =
      this.studioCollageTitleInput.value.trim() ||
      `קולאז׳ משפחתי (${selected.length} תמונות)`;
    const note = this.studioPromptInput.value.trim();
    const caption =
      note ||
      `קולאז׳ משפחתי שנוצר על ידי ${author} מתוך התמונות: ${selected
        .map((s) => s.title)
        .join(' · ')}`;

    this.studioSaveCatalogBtn.disabled = true;
    this._showStudioStatus('שומר את הקולאז׳ בקטלוג המשפחתי...', false);
    try {
      const resp = await fetch('/api/uploads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          uploader: author,
          caption,
          places: unionPlaces,
          people: unionPeople,
          files: [
            {
              filename: `family_collage_${Date.now()}.jpg`,
              data_url: this.lastRenderedCollageDataUrl,
              title: customTitle
            }
          ]
        })
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }
      this._showStudioStatus('הקולאז׳ נשמר והתווסף לקטלוג המשפחתי בהצלחה!', false);
      await this.loadCatalog();
    } catch (err) {
      this._showStudioStatus(`שגיאה בשמירה לקטלוג: ${err.message}`, true);
    } finally {
      this.studioSaveCatalogBtn.disabled = false;
    }
  }

  async submitStudioProposal() {
    const author = this.studioAuthorInput.value.trim();
    if (!author) {
      this._showStudioStatus('נא לבחור או להזין את שמכם לפני שליחת ההצעה.', true);
      this.studioAuthorInput.focus();
      return;
    }
    localStorage.setItem(AUTHOR_STORAGE_KEY, author);

    const selected = this._getSelectedItems();
    if (selected.length === 0) return;

    const rawPrompt = this.studioPromptInput.value.trim();
    const collageTitle = this.studioCollageTitleInput.value.trim();
    if (this.activeStudioModeId !== 'collage' && !rawPrompt) {
      this._showStudioStatus('נא לכתוב תיאור קצר כדי שנדע מה ליצור מהתמונות שבחרתם.', true);
      this.studioPromptInput.focus();
      return;
    }

    const modeLabels = {
      collage: '🖼️ קולאז׳ משפחתי',
      fuse_scene: '🏛️ מיזוג לסצנה מרחבית אחת',
      open_prompt: '✍️ תיאור מילולי פתוח'
    };
    const modeLabel = modeLabels[this.activeStudioModeId] || this.activeStudioModeId;
    const bodyText =
      rawPrompt || collageTitle || `הצעת קולאז׳ מ-${selected.length} תמונות נבחרות`;
    const comment = `[${modeLabel} · ${selected.map((s) => s.id).join(', ')}] ${bodyText}`;

    this.studioSubmitProposalBtn.disabled = true;
    this._showStudioStatus('שולח את ההצעה והתמונות שנבחרו...', false);

    try {
      // Render a compact snapshot (<= 720px wide) so it fits feedback snapshot byte limits cleanly
      const snapCanvas = document.createElement('canvas');
      const pickingCfg = (this.feedbackOverlay.config && this.feedbackOverlay.config.picking) || {};
      const snapRes = await renderCollageToCanvas(
        snapCanvas,
        selected.map((it) => ({ imgUrl: buildItemImageUrl(it), title: it.title })),
        {
          ...this._getStudioConfig(),
          collage_jpeg_quality: pickingCfg.snapshot_jpeg_quality || 0.84
        },
        {
          titleText: this.activeStudioModeId === 'collage' ? collageTitle : '',
          theme: this.activeStudioModeId === 'collage' ? this.activeCollageTheme : 'dark',
          targetWidthPx: pickingCfg.snapshot_max_width_px || 720
        }
      );

      const resp = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          author,
          category_id: 'suggestion',
          scene_id: 'catalog',
          comment,
          creation_mode: this.activeStudioModeId,
          selected_media_ids: selected.map((s) => s.id),
          snapshot_data_url: snapRes.jpegDataUrl,
          viewport: [window.innerWidth, window.innerHeight],
          page_path: window.location.pathname,
          target_ref: selected[0].id
        })
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      if (this.feedbackOverlay && typeof this.feedbackOverlay.refreshEntries === 'function') {
        await this.feedbackOverlay.refreshEntries();
      }
      this._showStudioStatus('ההצעה נשמרה בהצלחה בתיבת המשוב המשפחתית!', false);
      this.studioPromptInput.value = '';
    } catch (err) {
      this._showStudioStatus(`שגיאה בשליחה: ${err.message}`, true);
    } finally {
      this.studioSubmitProposalBtn.disabled = false;
    }
  }

  async handleSelectedFiles(fileList) {
    const arr = Array.from(fileList).slice(0, 15);
    for (const file of arr) {
      const dataUrl = await readFileAsDataUrl(file);
      this.pendingFiles.push({
        filename: file.name,
        dataUrl,
        customTitle: file.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ')
      });
    }
    this.renderPendingPreviews();
  }

  renderPendingPreviews() {
    this.previewStrip.innerHTML = '';
    if (this.pendingFiles.length === 0) {
      this.metaFieldsWrap.style.display = 'none';
      return;
    }
    this.metaFieldsWrap.style.display = 'flex';
    this.pendingFiles.forEach((pf, idx) => {
      const item = document.createElement('div');
      item.className = 'upload-preview-item';
      const isHeic = /\.(heic|heif)$/i.test(pf.filename);
      const previewImgHtml = isHeic
        ? `<div style="height:100px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;color:#E6B980;padding:8px;text-align:center;">📱 ${escapeHtml(
            pf.filename
          )}<br/>(יומר אוטומטית בשרת)</div>`
        : `<img src="${pf.dataUrl}" alt="${escapeHtml(pf.filename)}" />`;
      item.innerHTML = `
        ${previewImgHtml}
        <input type="text" value="${escapeHtml(pf.customTitle)}" placeholder="כותרת קצרה..." />
      `;
      item.querySelector('input').addEventListener('input', (ev) => {
        this.pendingFiles[idx].customTitle = ev.target.value;
      });
      this.previewStrip.appendChild(item);
    });
  }

  clearPendingUploads() {
    this.pendingFiles = [];
    this.fileInput.value = '';
    this.uploadCaptionInput.value = '';
    this.uploadEraInput.value = '';
    if (this.uploadCustomPlaceInput) this.uploadCustomPlaceInput.value = '';
    if (this.uploadCustomPersonInput) this.uploadCustomPersonInput.value = '';
    this._setSelectedCanonicals(this.uploadPlaceChips, [], '📍');
    this._setSelectedCanonicals(this.uploadPeopleChips, [], '👤');
    this.renderPendingPreviews();
  }

  _showUploadStatus(msg, isError = false) {
    this.uploadStatusMsg.textContent = msg;
    this.uploadStatusMsg.className = `beta-status-msg ${isError ? 'error' : 'success'}`;
  }

  async submitUploads() {
    const uploader = this.uploadAuthorInput.value.trim();
    if (!uploader) {
      this._showUploadStatus('נא לבחור או להזין את שמכם (למשל גדי, גיא, טל או נעמה).', true);
      this.uploadAuthorInput.focus();
      return;
    }
    if (this.pendingFiles.length === 0) {
      this._showUploadStatus('לא נבחרו תמונות להעלאה.', true);
      return;
    }

    // Auto-commit any text typed in the custom location / custom person inputs
    this._commitPendingCustomInput(this.uploadPlaceChips, this.uploadCustomPlaceInput, '📍');
    this._commitPendingCustomInput(this.uploadPeopleChips, this.uploadCustomPersonInput, '👤');

    localStorage.setItem(AUTHOR_STORAGE_KEY, uploader);
    this.uploadSubmitBtn.disabled = true;
    this._showUploadStatus(`מעלה ${this.pendingFiles.length} תמונות ושומר ב-drop_folder...`, false);

    try {
      const payload = {
        uploader,
        caption: this.uploadCaptionInput.value.trim(),
        date_or_era: this.uploadEraInput.value.trim(),
        places: this._getSelectedCanonicals(this.uploadPlaceChips),
        people: this._getSelectedCanonicals(this.uploadPeopleChips),
        files: this.pendingFiles.map((pf) => ({
          filename: pf.filename,
          data_url: pf.dataUrl,
          title: pf.customTitle
        }))
      };

      const resp = await fetch('/api/uploads', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      this.clearPendingUploads();
      this._showUploadStatus(
        `הועלו בהצלחה ${data.ingested_count} תמונות חדשות לקטלוג! (${
          data.duplicate_count
            ? data.duplicate_count + ' כפילויות דולגו'
            : 'סה״כ ' + data.catalog_total_count + ' בקטלוג'
        })`,
        false
      );
      this.activeFilter = 'src:drop_folder_imports';
      document.querySelectorAll('#catalog-filter-chips button').forEach((b) => {
        b.classList.toggle('active', b.getAttribute('data-cat-filter') === this.activeFilter);
      });
      await this.loadCatalog();
    } catch (err) {
      this._showUploadStatus(`שגיאה בהעלאה: ${err.message}`, true);
    } finally {
      this.uploadSubmitBtn.disabled = false;
    }
  }

  async loadCatalog() {
    const resp = await fetch('/api/catalog', { cache: 'no-store' });
    const data = await resp.json();
    const rawItems = (data.items || []).slice();
    // Surface family uploads (drop_folder_imports, newest first) at the top of the default view
    const familyUploads = rawItems
      .filter((it) => it.source_group === 'drop_folder_imports')
      .reverse();
    const archiveItems = rawItems.filter((it) => it.source_group !== 'drop_folder_imports');
    this.items = [...familyUploads, ...archiveItems];
    this.countPillEl.textContent = `סה״כ בקטלוג: ${this.items.length} תמונות מאומתות`;
    this._renderPeopleFilterBar();
    this.renderGrid();
    this.renderMultiSelectBar();
  }

  _getPersonDisplayLabel(canonical) {
    return this._personLabelMap.get(canonical) || PERSON_LABELS_HE[canonical] || canonical;
  }

  _togglePersonFilter(personCanonical) {
    if (!personCanonical || personCanonical === '__all__') {
      this.selectedPeopleFilters.clear();
    } else if (this.selectedPeopleFilters.has(personCanonical)) {
      this.selectedPeopleFilters.delete(personCanonical);
    } else {
      this.selectedPeopleFilters.add(personCanonical);
    }
    this._syncPeopleFilterActiveStates();
    this.renderGrid();
  }

  _syncPeopleFilterActiveStates() {
    if (!this.peopleFilterChipsEl) return;
    const isAll = this.selectedPeopleFilters.size === 0;
    this.peopleFilterChipsEl.querySelectorAll('button[data-person-filter]').forEach((btn) => {
      const key = btn.getAttribute('data-person-filter') || '';
      const active = key === '__all__' ? isAll : this.selectedPeopleFilters.has(key);
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
  }

  _renderPeopleFilterBar() {
    if (!this.peopleFilterChipsEl) return;
    const counts = new Map();
    for (const item of this.items) {
      for (const p of item.people || []) {
        if (!p) continue;
        counts.set(p, (counts.get(p) || 0) + 1);
      }
    }
    // Prune any selected filter no longer present
    for (const sel of Array.from(this.selectedPeopleFilters)) {
      if (!counts.has(sel)) {
        this.selectedPeopleFilters.delete(sel);
      }
    }
    const sortedPeople = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
    const isAll = this.selectedPeopleFilters.size === 0;

    let html = `<button type="button" class="beta-chip${
      isAll ? ' active' : ''
    }" data-person-filter="__all__" aria-pressed="${isAll ? 'true' : 'false'}">כל האנשים</button>`;

    for (const [canonical, cnt] of sortedPeople) {
      const active = this.selectedPeopleFilters.has(canonical);
      const label = this._getPersonDisplayLabel(canonical);
      html += `<button type="button" class="beta-chip${
        active ? ' active' : ''
      }" data-person-filter="${escapeHtml(canonical)}" aria-pressed="${
        active ? 'true' : 'false'
      }">${escapeHtml(label)} (${cnt})</button>`;
    }

    this.peopleFilterChipsEl.innerHTML = html;
    this.peopleFilterChipsEl.querySelectorAll('button[data-person-filter]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const key = btn.getAttribute('data-person-filter') || '__all__';
        this._togglePersonFilter(key);
      });
    });
  }

  _matchesFilter(item) {
    if (this.selectedPeopleFilters.size > 0) {
      const itemPeople = new Set(item.people || []);
      for (const reqPerson of this.selectedPeopleFilters) {
        if (!itemPeople.has(reqPerson)) {
          return false;
        }
      }
    }
    if (this.activeFilter === 'all') return true;
    if (this.activeFilter.startsWith('src:')) {
      return item.source_group === this.activeFilter.slice(4);
    }
    if (this.activeFilter.startsWith('place:')) {
      const sub = this.activeFilter.slice(6).toLowerCase();
      return (item.places || []).some((p) => p.toLowerCase().includes(sub));
    }
    if (this.activeFilter === 'art') {
      return ['portfolio_site', 'binyamin_gallery', 'museum_press'].includes(item.source_group);
    }
    return true;
  }

  renderGrid() {
    const filtered = this.items.filter((item) => {
      if (!this._matchesFilter(item)) return false;
      if (!this.searchQuery) return true;
      const hay = [
        item.id,
        item.title,
        item.caption,
        item.filename,
        item.date_or_era,
        ...(item.places || []),
        ...(item.people || []),
        ...(item.art_themes || [])
      ]
        .join(' ')
        .toLowerCase();
      return hay.includes(this.searchQuery);
    });

    if (filtered.length === 0) {
      this.gridEl.innerHTML = `
        <div class="portal-card" style="grid-column:1/-1;text-align:center;color:#B4AEA6;">
          לא נמצאו תמונות בסינון הנוכחי — נשמח שתעלו לכאן תמונות של הבית/המשפחה בתיבה למעלה!
        </div>
      `;
      return;
    }

    const selOrderMap = new Map(this.selectedMediaIds.map((id, i) => [id, i + 1]));
    this.gridEl.innerHTML = '';
    for (const item of filtered) {
      const imgUrl = buildItemImageUrl(item);
      const selOrder = selOrderMap.get(item.id);
      const card = document.createElement('article');
      card.className = `portal-photo-card${selOrder ? ' selected-for-studio' : ''}`;
      card.setAttribute('data-media-id', item.id);
      card.setAttribute('data-feedback-label', `${item.id} — ${item.title}`);

      const placeTags = (item.places || [])
        .map((p) => `<span class="photo-tag place">${escapeHtml(p)}</span>`)
        .join('');
      const peopleTags = (item.people || [])
        .map((p) => {
          const isSelectedPerson = this.selectedPeopleFilters.has(p);
          const disp = this._getPersonDisplayLabel(p);
          return `<button type="button" class="photo-tag person-filterable${
            isSelectedPerson ? ' active-person' : ''
          }" data-card-person="${escapeHtml(p)}" title="סנן לפי ${escapeHtml(disp)}">${escapeHtml(
            disp
          )}</button>`;
        })
        .join('');

      card.innerHTML = `
        <div class="photo-thumb-wrap">
          <button type="button" class="photo-select-btn${selOrder ? ' selected' : ''}">${
            selOrder ? `✓ נבחרה (#${selOrder})` : '☑️ בחר ליצירה'
          }</button>
          <img src="${imgUrl}" alt="${escapeHtml(item.title)}" loading="lazy" />
          <span class="photo-source-badge">${escapeHtml(
            SOURCE_LABELS_HE[item.source_group] || item.source_group
          )}</span>
        </div>
        <div class="photo-card-body">
          <div class="photo-card-title">${escapeHtml(item.title)}</div>
          ${
            item.caption
              ? `<div class="photo-card-caption">${escapeHtml(item.caption.slice(0, 140))}</div>`
              : `<div class="photo-card-caption" style="opacity:0.55;">ללא כיתוב עדיין — לחצו על ״ערוך כיתוב״ להוספת זיהוי/סיפור</div>`
          }
          <div class="photo-tags-row">
            ${item.date_or_era ? `<span class="photo-tag">🕰️ ${escapeHtml(item.date_or_era)}</span>` : ''}
            ${placeTags}
            ${peopleTags}
          </div>
          <div class="photo-card-actions">
            <button type="button" class="edit-meta-btn">✏️ ערוך כיתוב / תיוג</button>
            <button type="button" class="comment-card-btn">💬 הגב על תמונה זו</button>
          </div>
        </div>
      `;

      const selectBtn = card.querySelector('.photo-select-btn');
      selectBtn.addEventListener('click', (ev) => {
        ev.stopPropagation();
        this.toggleMediaSelection(item.id);
      });

      card.querySelector('.photo-thumb-wrap').addEventListener('click', (ev) => {
        if (ev.target.closest('.photo-select-btn')) return;
        this.feedbackOverlay.openLightbox(
          imgUrl,
          `${item.title} (${item.width}×${item.height}) — ${item.caption || ''}`
        );
      });

      card.querySelectorAll('[data-card-person]').forEach((pBtn) => {
        pBtn.addEventListener('click', (ev) => {
          ev.stopPropagation();
          const personKey = pBtn.getAttribute('data-card-person');
          if (personKey) {
            this._togglePersonFilter(personKey);
          }
        });
      });

      card.querySelector('.edit-meta-btn').addEventListener('click', () => {
        this.openEditModal(item);
      });

      card.querySelector('.comment-card-btn').addEventListener('click', () => {
        this.feedbackOverlay.openCommentForElement(
          card,
          'visual_element',
          `לגבי התמונה "${item.title}" (${item.id}): `
        );
      });

      this.gridEl.appendChild(card);
    }

    if (this.feedbackOverlay) {
      this.feedbackOverlay.notifyPageContentUpdated();
    }
  }

  focusMediaCard(mediaId) {
    this.activeFilter = 'all';
    this.selectedPeopleFilters.clear();
    this._syncPeopleFilterActiveStates();
    this.searchQuery = '';
    this.searchInput.value = '';
    this.renderGrid();
    const card = this.gridEl.querySelector(`[data-media-id="${CSS.escape(mediaId)}"]`);
    if (card) {
      card.classList.add('focused-media');
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  openEditModal(item) {
    this.editingItem = item;
    this.editIdEl.textContent = `${item.id} · ${item.filename} (${item.width}×${item.height})`;
    this.editTitleInput.value = item.title || '';
    this.editCaptionInput.value = item.caption || '';
    this.editEraInput.value = item.date_or_era || '';
    if (this.editCustomPlaceInput) this.editCustomPlaceInput.value = '';
    if (this.editCustomPersonInput) this.editCustomPersonInput.value = '';
    this._setSelectedCanonicals(this.editPlaceChips, item.places || [], '📍');
    this._setSelectedCanonicals(this.editPeopleChips, item.people || [], '👤');
    this.editBackdrop.classList.remove('hidden');
  }

  closeEditModal() {
    this.editingItem = null;
    this.editBackdrop.classList.add('hidden');
  }

  async saveEditModal() {
    if (!this.editingItem) return;
    this._commitPendingCustomInput(this.editPlaceChips, this.editCustomPlaceInput, '📍');
    this._commitPendingCustomInput(this.editPeopleChips, this.editCustomPersonInput, '👤');

    const editor = localStorage.getItem(AUTHOR_STORAGE_KEY) || 'משפחה';
    const payload = {
      editor,
      title: this.editTitleInput.value.trim(),
      caption: this.editCaptionInput.value.trim(),
      date_or_era: this.editEraInput.value.trim(),
      places: this._getSelectedCanonicals(this.editPlaceChips),
      people: this._getSelectedCanonicals(this.editPeopleChips)
    };
    const resp = await fetch(`/api/catalog/${encodeURIComponent(this.editingItem.id)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (resp.ok) {
      this.closeEditModal();
      await this.loadCatalog();
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  mountPortalHeaderFontPicker(document.getElementById('portal-beta-actions'));
  new CatalogPortalApp();
});
