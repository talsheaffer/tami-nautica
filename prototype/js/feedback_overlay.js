// prototype/js/feedback_overlay.js
// Universal Family Beta Review, Suggestion Box, Live DOM UI Element Inspector
// (Marker.io / BugHerd style), 3D Spatial Raycast Highlighting, and Cross-Portal Pin Sync.

import {
  captureCompositeViewportSnapshot,
  describeUiElement,
  inspectUiElementAtPoint,
  inspectUiElementInRect
} from './dom_snapshot.js';

const AUTHOR_STORAGE_KEY = 'tami70_beta_author';

const SPATIAL_SCENE_IDS = new Set([
  'shelf_portal',
  'paper_capsules',
  'tzofen_brick_table',
  'bene_beitcha',
  'bene_beitcha_scroll_foot',
  'purim_costume_1966',
  'vitrine_keys_pen',
  'humpty_dumpty_booth',
  'humpty_dumpty_bundle',
  'denya_model',
  'denya_living_room',
  'denya_couch_guitar',
  'plan_b_pipe',
  'plan_b_pipe_v1',
  'plan_b_pipe_v3',
  'wandering_drawing_1',
  'scroll_palm_world',
  'wandering_drawing',
  'fried_yard_installation',
  'fried_house_railing',
  'fried_table_hybrid',
  'herzliya_lawn_carpet',
  'herzliya_tree_canopy',
  'bar_giora_bedroom',
  'paris_cite_white_table',
  'toddler_tami_1958',
  'marseille_mirror',
  'marseille_mirror_v1'
]);

const SCENE_LABELS_HE = Object.freeze({
  shelf_portal: 'קיר המדפים: שער התערוכות (3D)',
  paper_capsules: 'קפסולות נייר (3D)',
  tzofen_brick_table: 'צופן: שולחן לבנים וחלון לעץ (3D)',
  bene_beitcha: 'בנה ביתך (3D)',
  bene_beitcha_scroll_foot: 'מגילה וכף רגל (3D)',
  purim_costume_1966: 'תחפושת פורים, 1966 (3D)',
  vitrine_keys_pen: 'ויטרינת המפתחות והעט (3D)',
  humpty_dumpty_booth: 'האמפטי דאמפטי בתא (3D)',
  humpty_dumpty_bundle: 'צרור עטוף בבד וחבל (3D)',
  denya_model: 'דגם בית דניה (3D)',
  denya_living_room: 'סלון דניה: שולחן ושערים (3D)',
  denya_couch_guitar: 'ספת דניה: גיטרה ושיטוט (3D)',
  plan_b_pipe: 'צינור Plan B (3D)',
  plan_b_pipe_v1: 'צינור Plan B: גרסה 1 (3D)',
  plan_b_pipe_v3: 'צינור Plan B: תמי ויערה (3D)',
  wandering_drawing_1: 'שיטוט ברישום 1 (3D)',
  scroll_palm_world: 'עולם המגילה: דקל ומוטות בחול (3D)',
  wandering_drawing: 'שיטוט ברישום 2 (3D)',
  fried_yard_installation: 'חצר בית פריד: מרחב לא מוגן (3D)',
  fried_house_railing: 'בית פריד בהרצליה: כרית ואבן (3D)',
  fried_table_hybrid: 'שולחן בית פריד: היבריד (3D)',
  herzliya_lawn_carpet: 'הדשא בהרצליה: שטיח ואדמה (3D)',
  herzliya_tree_canopy: 'צמרת העץ בחצר בהרצליה (3D)',
  bar_giora_bedroom: 'חדר השינה בבר גיורא (3D)',
  paris_cite_white_table: 'סטודיו פריז: השולחן הלבן (3D)',
  toddler_tami_1958: 'תמי הפעוטה, 1958 (3D)',
  marseille_mirror: 'מרסיי: מתחת למראה (3D)',
  marseille_mirror_v1: 'מרסיי: גרסה 1 (3D)',
  wiki: 'ויקי ביוגרפי',
  catalog: 'קטלוג ותמונות'
});

function formatShortTimestamp(isoStr) {
  if (!isoStr) return '';
  const dt = new Date(isoStr);
  if (Number.isNaN(dt.getTime())) return isoStr;
  return dt.toLocaleString('he-IL', {
    day: '2-digit',
    month: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  });
}

function sceneRouteHref(sceneId, targetRef, focusEntryId) {
  const params = new URLSearchParams();
  if (focusEntryId) params.set('focus_fb', focusEntryId);
  if (sceneId === 'wiki') {
    if (targetRef) params.set('page', targetRef);
    return `/wiki?${params.toString()}`;
  }
  if (sceneId === 'catalog') {
    if (targetRef) params.set('media', targetRef);
    return `/catalog?${params.toString()}`;
  }
  if (SPATIAL_SCENE_IDS.has(sceneId) && sceneId !== 'shelf_portal') {
    params.set('scene', sceneId);
  }
  const qs = params.toString();
  return qs ? `/?${qs}` : '/';
}

export class BetaFeedbackOverlay {
  constructor(sceneManager = null, options = {}) {
    this.sceneManager = sceneManager;
    this.options = options;
    this.defaultSceneId = options.defaultSceneId || 'shelf_portal';
    this.config = null;
    this.entries = [];
    this.selectedEntryId = null;
    this.activeHighlightDraft = null;
    this.activeSnapshotDataUrl = null;
    this.activeUploadedImageLabel = null;
    this.activeTargetRefDraft = null;
    this._pickingFromDrawer = false;
    this.selectedCategoryId = 'suggestion';
    this.sceneFilter = 'all';
    this.showPins = true;
    this.isPicking = false;
    this._dragState = null;
    this._pollTimer = null;
    this._unsubscribeFrame = null;
    this._pinElements = new Map();

    this.init();
  }

  get currentContextSceneId() {
    if (this.sceneManager) {
      return this.sceneManager.currentSceneId;
    }
    return this.defaultSceneId;
  }

  get currentTargetRef() {
    if (typeof this.options.getTargetRef === 'function') {
      return this.options.getTargetRef();
    }
    return this.currentContextSceneId;
  }

  async init() {
    let cfg = null;
    try {
      const resp = await fetch('/api/beta-config', { cache: 'no-store' });
      if (resp.status === 404) {
        return;
      }
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status} loading /api/beta-config`);
      }
      cfg = await resp.json();
    } catch (err) {
      if (location.protocol === 'file:') {
        return;
      }
      console.error('Beta config load failed:', err);
      return;
    }

    if (!cfg || !cfg.enabled) {
      return;
    }

    this.config = cfg;
    if (cfg.categories && cfg.categories.length > 0) {
      this.selectedCategoryId = cfg.categories[0].category_id;
    }

    this._mountTopBarControls();
    this._mountOverlayDom();
    this._bindEvents();

    if (this.sceneManager) {
      this._unsubscribeFrame = this.sceneManager.addFrameListener(() => {
        this._updateProjectedPinsAndHighlight();
      });
    } else {
      this._onScrollOrResize = () => this._updateProjectedPinsAndHighlight();
      window.addEventListener('scroll', this._onScrollOrResize, { passive: true });
      window.addEventListener('resize', this._onScrollOrResize);
    }

    await this.refreshEntries();
    this._checkUrlFocusParam();
    this._startPolling();

    if (typeof this.options.onReady === 'function') {
      this.options.onReady(this);
    }
    window.__BETA_FEEDBACK__ = this;
  }

  _checkUrlFocusParam() {
    const params = new URLSearchParams(window.location.search);
    const focusId = params.get('focus_fb');
    if (focusId) {
      setTimeout(() => {
        this.selectEntry(focusId, { flyCamera: true, collapseOnMobile: true });
      }, 350);
    }
  }

  _mountTopBarControls() {
    const titleH1 = document.querySelector('.title-badge h1');
    if (titleH1 && !titleH1.querySelector('.beta-pill-tag')) {
      const tag = document.createElement('span');
      tag.className = 'beta-pill-tag';
      tag.textContent = this.config.beta_badge_label_he;
      titleH1.appendChild(tag);
    }

    // Mount Secret Family Portal Switcher Bar inside #spatial-sidebar-hub-slot (or fallback)
    if (this.sceneManager && this.config.portal_nav_items && !document.querySelector('.beta-hub-bar')) {
      const hubBar = document.createElement('nav');
      hubBar.className = 'beta-hub-bar';
      hubBar.dir = 'rtl';
      hubBar.setAttribute('aria-label', 'ניווט סביבת העבודה המשפחתית');
      const linksHtml = this.config.portal_nav_items
        .map(
          (item) =>
            `<a class="beta-hub-link ${item.nav_id === 'prototype' ? 'active' : ''}" href="${
              item.href
            }">${item.label_he}</a>`
        )
        .join('');
      hubBar.innerHTML = `<span class="hub-secret-pill">סביבת בטא</span>${linksHtml}`;
      const sidebarHubSlot = document.querySelector('#spatial-sidebar-hub-slot');
      const bottomDock = document.querySelector('.bottom-dock');
      if (sidebarHubSlot) {
        sidebarHubSlot.appendChild(hubBar);
      } else if (bottomDock) {
        bottomDock.prepend(hubBar);
      } else {
        document.body.appendChild(hubBar);
      }
    }

    const actionsContainer =
      document.querySelector('#portal-beta-actions') || document.querySelector('.top-actions');
    if (!actionsContainer) return;

    this.quickPickBtn = document.createElement('button');
    this.quickPickBtn.id = 'beta-highlight-btn';
    this.quickPickBtn.type = 'button';
    this.quickPickBtn.className = 'icon-btn beta-action-btn';
    this.quickPickBtn.setAttribute('aria-label', 'סמן רכיב ממשק, טקסט או אלמנט והוסף הערה');
    this.quickPickBtn.innerHTML = '<span>סמן והגב</span>';

    this.drawerToggleBtn = document.createElement('button');
    this.drawerToggleBtn.id = 'beta-drawer-btn';
    this.drawerToggleBtn.type = 'button';
    this.drawerToggleBtn.className = 'icon-btn beta-action-btn';
    this.drawerToggleBtn.setAttribute('aria-label', 'פתח תיבת משוב והצעות');
    this.countBadge = document.createElement('span');
    this.countBadge.id = 'beta-count-badge';
    this.countBadge.className = 'beta-count-badge';
    this.countBadge.textContent = '0';
    this.drawerToggleBtn.append('משוב ', this.countBadge);

    actionsContainer.prepend(this.quickPickBtn, this.drawerToggleBtn);
  }

  _mountOverlayDom() {
    // 1. Pins & Active Highlight Frame Layer
    this.pinsLayer = document.createElement('div');
    this.pinsLayer.id = 'beta-pins-layer';

    this.projectedBoxEl = document.createElement('div');
    this.projectedBoxEl.id = 'beta-projected-box';
    this.projectedBoxEl.className = 'hidden';
    this.pinsLayer.appendChild(this.projectedBoxEl);
    document.body.appendChild(this.pinsLayer);

    // 2. Interactive Highlight Surface + Live DOM Hover Inspector + Banner
    this.highlightSurface = document.createElement('div');
    this.highlightSurface.id = 'beta-highlight-surface';
    this.highlightSurface.className = 'hidden';

    this.dragRectEl = document.createElement('div');
    this.dragRectEl.id = 'beta-drag-rect';
    this.dragRectEl.className = 'hidden';
    this.highlightSurface.appendChild(this.dragRectEl);
    document.body.appendChild(this.highlightSurface);

    this.hoverInspectorEl = document.createElement('div');
    this.hoverInspectorEl.id = 'beta-hover-inspector';
    this.hoverInspectorEl.className = 'hidden';
    this.hoverInspectorEl.dir = 'rtl';
    this.hoverInspectorEl.innerHTML = `<div class="inspector-badge" id="beta-inspector-badge"></div>`;
    document.body.appendChild(this.hoverInspectorEl);
    this.inspectorBadgeEl = this.hoverInspectorEl.querySelector('#beta-inspector-badge');

    this.highlightBanner = document.createElement('div');
    this.highlightBanner.id = 'beta-highlight-banner';
    this.highlightBanner.className = 'hidden';
    this.highlightBanner.dir = 'rtl';
    this.highlightBanner.innerHTML = `
      <span class="banner-text">🎯 <strong>מצב סימון חכם:</strong> רחפו או הקישו על כל כפתור ממשק, טקסט, תמונה או מיצב — או גררו מלבן</span>
      <button type="button" class="cancel-pick-btn" id="beta-cancel-pick-btn">✕ ביטול</button>
    `;
    document.body.appendChild(this.highlightBanner);

    // 3. Floating Focus Bar
    this.focusBar = document.createElement('div');
    this.focusBar.id = 'beta-focus-bar';
    this.focusBar.className = 'hidden';
    this.focusBar.dir = 'rtl';
    document.body.appendChild(this.focusBar);

    // 4. Collaborative Beta Drawer
    const savedAuthor = localStorage.getItem(AUTHOR_STORAGE_KEY) || '';
    const presetChipsHtml = this.config.reviewer_presets
      .map(
        (p) =>
          `<button type="button" class="beta-chip author-preset-chip ${
            savedAuthor === p.label_he ? 'active' : ''
          }" data-author="${p.label_he}">${p.label_he}</button>`
      )
      .join('');

    const categoryChipsHtml = this.config.categories
      .map(
        (c, idx) =>
          `<button type="button" class="beta-chip category-chip ${
            idx === 0 ? 'active' : ''
          }" data-category="${c.category_id}">${c.icon} ${c.label_he}</button>`
      )
      .join('');

    const poseCheckboxHtml = this.sceneManager
      ? `<label class="beta-pose-check">
           <input type="checkbox" id="beta-include-pose" checked />
           <span>שמור זווית מצלמה נוכחית</span>
         </label>`
      : `<span class="beta-pose-check">📍 ההקשר בדף נשמר אוטומטית</span>`;

    this.drawer = document.createElement('aside');
    this.drawer.id = 'beta-drawer';
    this.drawer.dir = 'rtl';
    this.drawer.setAttribute('aria-label', 'משוב בטא ותיבת הצעות');
    this.drawer.innerHTML = `
      <div class="beta-drawer-header">
        <h2>💬 משוב והצעות (משפחה)</h2>
        <div class="beta-header-controls">
          <button type="button" id="beta-pins-toggle" class="beta-small-toggle active" title="הצג/הסתר סיכות על המסך">
            📍 סיכות
          </button>
          <button type="button" id="beta-close-drawer-btn" class="close-btn" aria-label="סגור משוב">✕</button>
        </div>
      </div>
      <div class="beta-secret-banner">${this._escapeHtml(this.config.secret_notice_he || '')}</div>

      <div class="beta-drawer-scroll">
        <form id="beta-composer-form" class="beta-composer" novalidate>
          <div>
            <span class="beta-row-label">מי כותב/ת?</span>
            <div class="beta-chips-row">
              ${presetChipsHtml}
              <input
                type="text"
                id="beta-author-input"
                class="beta-author-input"
                placeholder="שם אחר..."
                maxlength="${this.config.limits.max_author_chars}"
                value="${savedAuthor.replace(/"/g, '&quot;')}"
              />
            </div>
          </div>

          <div>
            <span class="beta-row-label">סוג ההערה / הרעיון</span>
            <div class="beta-chips-row" id="beta-category-row">
              ${categoryChipsHtml}
            </div>
          </div>

          <div id="beta-attach-actions-row" class="beta-attach-actions-row">
            <button type="button" id="beta-drawer-upload-btn" class="beta-upload-cta-btn">
              📷 צרף תמונה / צילום מסך
            </button>
            <button type="button" id="beta-drawer-pick-btn" class="beta-pick-cta-btn">
              🎯 סמן נקודה במסך
            </button>
            <input type="file" id="beta-file-input" accept="image/*" hidden />
          </div>

          <div id="beta-highlight-attachment" class="beta-highlight-attachment hidden">
            <img id="beta-attach-thumb" class="beta-attach-thumb" alt="צילום מצורף למשוב" />
            <div class="beta-attach-meta">
              <div id="beta-attach-layer" class="beta-attach-layer"></div>
              <div id="beta-attach-coords" class="beta-attach-coords"></div>
              <div class="beta-attach-btns">
                <button type="button" id="beta-replace-upload-btn">📷 החלף תמונה</button>
                <button type="button" id="beta-repick-btn">🎯 סמן במסך</button>
                <button type="button" id="beta-clear-pick-btn">✕ הסר</button>
              </div>
            </div>
          </div>

          <textarea
            id="beta-comment-input"
            class="beta-textarea"
            placeholder="כתבו כאן מחשבה, תיקון ביוגרפי, הערה על כפתור/ממשק, או רעיון לעיצוב..."
            maxlength="${this.config.limits.max_comment_chars}"
          ></textarea>

          <div id="beta-status-msg" class="beta-status-msg hidden"></div>

          <div class="beta-composer-footer">
            ${poseCheckboxHtml}
            <button type="submit" id="beta-submit-btn" class="beta-submit-btn">שלח משוב</button>
          </div>
        </form>

        <div class="beta-feed-header">
          <span class="beta-feed-title">הערות והצעות שנשלחו</span>
          <div class="beta-chips-row" id="beta-filter-row">
            <button type="button" class="beta-chip filter-chip active" data-filter="all">הכל</button>
            <button type="button" class="beta-chip filter-chip" data-filter="shelf_portal">⊞ מדפים</button>
            <button type="button" class="beta-chip filter-chip" data-filter="paper_capsules">🫧 קפסולות</button>
            <button type="button" class="beta-chip filter-chip" data-filter="bene_beitcha">🏛️ בנה ביתך</button>
            <button type="button" class="beta-chip filter-chip" data-filter="denya_model">🏠 דניה</button>
            <button type="button" class="beta-chip filter-chip" data-filter="plan_b_pipe">🕳️ Plan B</button>
            <button type="button" class="beta-chip filter-chip" data-filter="wandering_drawing">✏️ שיטוט</button>
            <button type="button" class="beta-chip filter-chip" data-filter="marseille_mirror">🪞 מרסיי</button>
            <button type="button" class="beta-chip filter-chip" data-filter="wiki">📖 ויקי</button>
            <button type="button" class="beta-chip filter-chip" data-filter="catalog">🖼️ קטלוג</button>
          </div>
        </div>

        <div id="beta-entries-list" class="beta-entries-list"></div>
      </div>
    `;
    document.body.appendChild(this.drawer);

    // 5. Lightbox for full-size annotated snapshots
    this.lightbox = document.createElement('div');
    this.lightbox.id = 'beta-lightbox';
    this.lightbox.className = 'hidden';
    this.lightbox.dir = 'rtl';
    this.lightbox.innerHTML = `
      <img id="beta-lightbox-img" alt="צילום מסך מסומן" />
      <div id="beta-lightbox-caption" class="lightbox-caption"></div>
      <button type="button" class="icon-btn" id="beta-lightbox-close">✕ סגור תצוגה</button>
    `;
    document.body.appendChild(this.lightbox);

    // Cache interactive elements
    this.authorInput = this.drawer.querySelector('#beta-author-input');
    this.commentInput = this.drawer.querySelector('#beta-comment-input');
    this.includePoseCheck = this.drawer.querySelector('#beta-include-pose');
    this.submitBtn = this.drawer.querySelector('#beta-submit-btn');
    this.statusMsgEl = this.drawer.querySelector('#beta-status-msg');
    this.attachmentBox = this.drawer.querySelector('#beta-highlight-attachment');
    this.attachThumb = this.drawer.querySelector('#beta-attach-thumb');
    this.attachLayerEl = this.drawer.querySelector('#beta-attach-layer');
    this.attachCoordsEl = this.drawer.querySelector('#beta-attach-coords');
    this.attachActionsRow = this.drawer.querySelector('#beta-attach-actions-row');
    this.pickCtaBtn = this.drawer.querySelector('#beta-drawer-pick-btn');
    this.uploadCtaBtn = this.drawer.querySelector('#beta-drawer-upload-btn');
    this.fileInputEl = this.drawer.querySelector('#beta-file-input');
    this.entriesListEl = this.drawer.querySelector('#beta-entries-list');
    this.pinsToggleBtn = this.drawer.querySelector('#beta-pins-toggle');
  }

  _bindEvents() {
    if (this.quickPickBtn) {
      this.quickPickBtn.addEventListener('click', () => this.startHighlightMode());
    }
    if (this.drawerToggleBtn) {
      this.drawerToggleBtn.addEventListener('click', () => this.toggleDrawer());
    }
    this.drawer.querySelector('#beta-close-drawer-btn').addEventListener('click', () => {
      this.closeDrawer();
    });
    this.pickCtaBtn.addEventListener('click', () => this.startHighlightMode());
    this.uploadCtaBtn.addEventListener('click', () => {
      if (this.fileInputEl) this.fileInputEl.click();
    });
    this.drawer.querySelector('#beta-replace-upload-btn').addEventListener('click', () => {
      if (this.fileInputEl) this.fileInputEl.click();
    });
    this.fileInputEl.addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (file) {
        this._handleUploadedImageFile(file);
      }
      e.target.value = '';
    });
    this.drawer.querySelector('#beta-repick-btn').addEventListener('click', () => {
      this.startHighlightMode();
    });
    this.drawer.querySelector('#beta-clear-pick-btn').addEventListener('click', () => {
      this.clearHighlightDraft();
    });

    this.highlightBanner
      .querySelector('#beta-cancel-pick-btn')
      .addEventListener('click', () => this.cancelHighlightMode({ reopenDrawerIfFromDrawer: true }));

    this.pinsToggleBtn.addEventListener('click', () => {
      this.showPins = !this.showPins;
      this.pinsToggleBtn.classList.toggle('active', this.showPins);
      this.pinsLayer.style.display = this.showPins ? 'block' : 'none';
    });

    this.drawer.querySelectorAll('.author-preset-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        const name = chip.getAttribute('data-author') || '';
        this.authorInput.value = name;
        localStorage.setItem(AUTHOR_STORAGE_KEY, name);
        this._syncAuthorChipHighlight();
      });
    });

    this.authorInput.addEventListener('input', () => {
      this._syncAuthorChipHighlight();
    });

    this.drawer.querySelectorAll('.category-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        this.setCategory(chip.getAttribute('data-category') || 'suggestion');
      });
    });

    this.drawer.querySelectorAll('.filter-chip').forEach((chip) => {
      chip.addEventListener('click', () => {
        this.sceneFilter = chip.getAttribute('data-filter') || 'all';
        this.drawer.querySelectorAll('.filter-chip').forEach((c) => {
          c.classList.toggle('active', c === chip);
        });
        this._renderEntriesList();
      });
    });

    this.attachThumb.addEventListener('click', () => {
      if (this.activeSnapshotDataUrl) {
        this.openLightbox(
          this.activeSnapshotDataUrl,
          this.activeHighlightDraft ? this.activeHighlightDraft.layer_label_he : ''
        );
      }
    });

    this.lightbox.addEventListener('click', (e) => {
      if (e.target === this.lightbox || e.target.id === 'beta-lightbox-close') {
        this.lightbox.classList.add('hidden');
      }
    });

    this.drawer.querySelector('#beta-composer-form').addEventListener('submit', (e) => {
      e.preventDefault();
      this.submitFeedbackEntry();
    });

    // Pointer Highlight Selection + Live Hover DOM Inspector
    this._onPointerDown = (e) => {
      if (!this.isPicking) return;
      if (e.button !== undefined && e.button !== 0) return;
      e.preventDefault();
      this.highlightSurface.setPointerCapture(e.pointerId);
      this._dragState = {
        pointerId: e.pointerId,
        startX: e.clientX,
        startY: e.clientY,
        currX: e.clientX,
        currY: e.clientY
      };
      this.dragRectEl.classList.add('hidden');
      this._updateLiveHoverInspector(e.clientX, e.clientY);
    };

    this._onPointerMove = (e) => {
      if (!this.isPicking) return;
      if (!this._dragState) {
        this._updateLiveHoverInspector(e.clientX, e.clientY);
        return;
      }
      if (e.pointerId !== this._dragState.pointerId) return;
      e.preventDefault();
      this._dragState.currX = e.clientX;
      this._dragState.currY = e.clientY;
      const dist = Math.hypot(
        this._dragState.currX - this._dragState.startX,
        this._dragState.currY - this._dragState.startY
      );
      if (dist >= this.config.picking.min_region_drag_px) {
        this.hoverInspectorEl.classList.add('hidden');
        const left = Math.min(this._dragState.startX, this._dragState.currX);
        const top = Math.min(this._dragState.startY, this._dragState.currY);
        const width = Math.abs(this._dragState.currX - this._dragState.startX);
        const height = Math.abs(this._dragState.currY - this._dragState.startY);
        this.dragRectEl.style.left = `${left}px`;
        this.dragRectEl.style.top = `${top}px`;
        this.dragRectEl.style.width = `${width}px`;
        this.dragRectEl.style.height = `${height}px`;
        this.dragRectEl.classList.remove('hidden');
      } else {
        this._updateLiveHoverInspector(e.clientX, e.clientY);
      }
    };

    this._onPointerUp = (e) => {
      if (!this.isPicking || !this._dragState || e.pointerId !== this._dragState.pointerId) return;
      e.preventDefault();
      const drag = this._dragState;
      this._dragState = null;
      this.dragRectEl.classList.add('hidden');
      this.hoverInspectorEl.classList.add('hidden');
      this._completeSpatialSelection(drag.startX, drag.startY, e.clientX, e.clientY);
    };

    this.highlightSurface.addEventListener('pointerdown', this._onPointerDown);
    this.highlightSurface.addEventListener('pointermove', this._onPointerMove);
    this.highlightSurface.addEventListener('pointerup', this._onPointerUp);
    this.highlightSurface.addEventListener('pointercancel', () => {
      this._dragState = null;
      this.dragRectEl.classList.add('hidden');
      this.hoverInspectorEl.classList.add('hidden');
    });

    this._onEscapeKey = (e) => {
      if (e.key === 'Escape') {
        if (this.isPicking) {
          this.cancelHighlightMode();
        } else if (!this.lightbox.classList.contains('hidden')) {
          this.lightbox.classList.add('hidden');
        }
      }
    };
    window.addEventListener('keydown', this._onEscapeKey);
  }

  _updateLiveHoverInspector(clientX, clientY) {
    const uiDesc = inspectUiElementAtPoint(clientX, clientY, [
      this.highlightSurface,
      this.hoverInspectorEl
    ]);
    if (!uiDesc) {
      this.hoverInspectorEl.classList.add('hidden');
      return;
    }
    const r = uiDesc.rect;
    this.hoverInspectorEl.style.left = `${r.left}px`;
    this.hoverInspectorEl.style.top = `${r.top}px`;
    this.hoverInspectorEl.style.width = `${r.width}px`;
    this.hoverInspectorEl.style.height = `${r.height}px`;
    this.hoverInspectorEl.classList.toggle('badge-below', r.top < 38);
    this.inspectorBadgeEl.textContent = `🧩 ${uiDesc.labelHe}`;
    this.hoverInspectorEl.classList.remove('hidden');
  }

  _syncAuthorChipHighlight() {
    const val = this.authorInput.value.trim();
    this.drawer.querySelectorAll('.author-preset-chip').forEach((chip) => {
      chip.classList.toggle('active', chip.getAttribute('data-author') === val);
    });
  }

  setCategory(categoryId) {
    this.selectedCategoryId = categoryId;
    this.drawer.querySelectorAll('.category-chip').forEach((chip) => {
      chip.classList.toggle('active', chip.getAttribute('data-category') === categoryId);
    });
  }

  openDrawer() {
    if (!this.drawer) return;
    const infoDrawer = document.getElementById('info-drawer');
    if (infoDrawer) infoDrawer.classList.remove('open');
    document.body.classList.remove('info-drawer-open');
    const spatialSidebar = document.getElementById('spatial-sidebar');
    if (spatialSidebar) {
      spatialSidebar.classList.add('collapsed');
      const sbToggle = document.getElementById('sidebar-toggle-btn');
      if (sbToggle) {
        sbToggle.classList.remove('active');
        sbToggle.setAttribute('aria-expanded', 'false');
      }
    }
    if (this.focusBar) this.focusBar.classList.add('hidden');
    this.drawer.classList.add('open');
    document.body.classList.add('beta-drawer-open');
    if (this.drawerToggleBtn) this.drawerToggleBtn.classList.add('active');
  }

  closeDrawer() {
    if (!this.drawer) return;
    if (
      document.activeElement &&
      this.drawer.contains(document.activeElement) &&
      typeof document.activeElement.blur === 'function'
    ) {
      document.activeElement.blur();
    }
    this.drawer.classList.remove('open');
    document.body.classList.remove('beta-drawer-open');
    if (this.drawerToggleBtn) this.drawerToggleBtn.classList.remove('active');
  }

  toggleDrawer() {
    if (!this.drawer) return;
    if (this.drawer.classList.contains('open')) {
      this.closeDrawer();
    } else {
      this.openDrawer();
      this.refreshEntries();
    }
  }

  startHighlightMode() {
    if (!this.drawer) return;
    this._pickingFromDrawer = this.drawer.classList.contains('open');
    this.closeDrawer();
    const infoDrawer = document.getElementById('info-drawer');
    if (infoDrawer) infoDrawer.classList.remove('open');
    document.body.classList.remove('info-drawer-open');
    const centerHint = document.getElementById('center-hint');
    if (centerHint) centerHint.classList.add('hidden');
    if (this.focusBar) this.focusBar.classList.add('hidden');

    this.isPicking = true;
    if (this.sceneManager) {
      this.sceneManager.setInteractionCapture(true);
    }
    this.highlightSurface.classList.remove('hidden');
    this.highlightBanner.classList.remove('hidden');
    if (this.quickPickBtn) this.quickPickBtn.classList.add('active');
  }

  cancelHighlightMode(options = {}) {
    const shouldReopen = Boolean(options.reopenDrawerIfFromDrawer && this._pickingFromDrawer);
    this._pickingFromDrawer = false;
    this.isPicking = false;
    this._dragState = null;
    this.dragRectEl.classList.add('hidden');
    this.hoverInspectorEl.classList.add('hidden');
    this.highlightSurface.classList.add('hidden');
    this.highlightBanner.classList.add('hidden');
    if (this.quickPickBtn) this.quickPickBtn.classList.remove('active');
    if (this.sceneManager) {
      this.sceneManager.setInteractionCapture(false);
    }
    if (shouldReopen) {
      this.openDrawer();
    }
  }

  _buildUiHighlightRecord(uiDesc, customRectPx = null) {
    const vw = Math.max(1, window.innerWidth);
    const vh = Math.max(1, window.innerHeight);
    const r = uiDesc.element.getBoundingClientRect();
    const x0 = Math.max(0, Math.min(1, (customRectPx ? Math.min(r.left, customRectPx.x0) : r.left) / vw));
    const y0 = Math.max(0, Math.min(1, (customRectPx ? Math.min(r.top, customRectPx.y0) : r.top) / vh));
    const x1 = Math.max(0, Math.min(1, (customRectPx ? Math.max(r.right, customRectPx.x1) : r.right) / vw));
    const y1 = Math.max(0, Math.min(1, (customRectPx ? Math.max(r.bottom, customRectPx.y1) : r.bottom) / vh));
    return {
      mode: 'ui',
      layer_id: 'ui',
      layer_label_he: uiDesc.labelHe,
      screen_rect_norm: {
        x0: Number(x0.toFixed(5)),
        y0: Number(y0.toFixed(5)),
        x1: Number(x1.toFixed(5)),
        y1: Number(y1.toFixed(5))
      },
      pin_screen_norm: [
        Number((0.5 * (x0 + x1)).toFixed(5)),
        Number((0.5 * (y0 + y1)).toFixed(5))
      ],
      world_point: null,
      world_box_corners: null,
      texture_uv: null,
      texture_pixel: null,
      ui_selector: uiDesc.selector
    };
  }

  openCommentForElement(domElement, presetCategory = 'visual_element', presetPrefix = '') {
    const uiDesc = describeUiElement(domElement);
    if (!uiDesc) return;
    const highlightRecord = this._buildUiHighlightRecord(uiDesc);
    const snapDataUrl = captureCompositeViewportSnapshot({
      sceneManager: this.sceneManager,
      highlightRecord,
      pickingCfg: this.config.picking
    });
    this.activeHighlightDraft = highlightRecord;
    this.activeSnapshotDataUrl = snapDataUrl;
    this.activeTargetRefDraft = uiDesc.targetRef || this.currentTargetRef;
    this.setCategory(presetCategory);
    this._renderHighlightDraftAttachment();
    if (presetPrefix && !this.commentInput.value.trim()) {
      this.commentInput.value = presetPrefix;
    }
    this.openDrawer();
    setTimeout(() => {
      if (this.commentInput) this.commentInput.focus();
    }, 100);
  }

  _completeSpatialSelection(startX, startY, endX, endY) {
    const pickingCfg = this.config.picking;
    const dist = Math.hypot(endX - startX, endY - startY);
    const vw = Math.max(1, window.innerWidth);
    const vh = Math.max(1, window.innerHeight);

    this.highlightSurface.classList.add('hidden');
    this.highlightBanner.classList.add('hidden');
    this.hoverInspectorEl.classList.add('hidden');

    let highlightRecord = null;
    let detectedTargetRef = null;
    let autoCategory = 'visual_element';

    if (dist >= pickingCfg.min_region_drag_px) {
      const boxPx = {
        x0: Math.min(startX, endX),
        y0: Math.min(startY, endY),
        x1: Math.max(startX, endX),
        y1: Math.max(startY, endY)
      };
      // Check DOM UI element inside dragged box FIRST (so boxing a button picks the button!)
      const boxedUi = inspectUiElementInRect(boxPx, [this.highlightSurface, this.hoverInspectorEl]);
      if (boxedUi) {
        highlightRecord = this._buildUiHighlightRecord(boxedUi);
        detectedTargetRef = boxedUi.targetRef;
        autoCategory = this.sceneManager ? 'ui_ux' : this.defaultSceneId === 'wiki' ? 'story_bio' : 'visual_element';
      } else if (this.sceneManager) {
        highlightRecord = this.sceneManager.pickSpatialRegion(
          { x0: startX, y0: startY, x1: endX, y1: endY },
          pickingCfg
        );
        autoCategory = 'visual_element';
      } else {
        const x0 = Math.max(0, Math.min(1, boxPx.x0 / vw));
        const y0 = Math.max(0, Math.min(1, boxPx.y0 / vh));
        const x1 = Math.max(0, Math.min(1, boxPx.x1 / vw));
        const y1 = Math.max(0, Math.min(1, boxPx.y1 / vh));
        highlightRecord = {
          mode: 'ui',
          layer_id: 'ui',
          layer_label_he: 'אזור מסומן בעמוד',
          screen_rect_norm: {
            x0: Number(x0.toFixed(5)),
            y0: Number(y0.toFixed(5)),
            x1: Number(x1.toFixed(5)),
            y1: Number(y1.toFixed(5))
          },
          pin_screen_norm: [
            Number((0.5 * (x0 + x1)).toFixed(5)),
            Number((0.5 * (y0 + y1)).toFixed(5))
          ],
          world_point: null,
          world_box_corners: null,
          texture_uv: null,
          texture_pixel: null,
          ui_selector: null
        };
      }
    } else {
      const uiDesc = inspectUiElementAtPoint(endX, endY, [
        this.highlightSurface,
        this.hoverInspectorEl
      ]);
      if (uiDesc) {
        highlightRecord = this._buildUiHighlightRecord(uiDesc);
        detectedTargetRef = uiDesc.targetRef;
        autoCategory = this.sceneManager ? 'ui_ux' : this.defaultSceneId === 'wiki' ? 'story_bio' : 'visual_element';
      } else if (this.sceneManager) {
        const hit3d = this.sceneManager.pickSpatialTarget(endX, endY, pickingCfg);
        const radX = pickingCfg.default_point_box_radius_norm;
        const radY = radX * (vw / vh);
        const nx = hit3d ? hit3d.pinScreenNorm[0] : endX / vw;
        const ny = hit3d ? hit3d.pinScreenNorm[1] : endY / vh;
        const x0 = Math.max(0, nx - radX);
        const y0 = Math.max(0, ny - radY);
        const x1 = Math.min(1, nx + radX);
        const y1 = Math.min(1, ny + radY);

        let worldBoxCorners = null;
        if (hit3d && hit3d.worldPoint) {
          const planeZ = hit3d.worldPoint[2];
          const c0 = this.sceneManager._intersectPlaneZ(x0 * vw, y0 * vh, planeZ);
          const c1 = this.sceneManager._intersectPlaneZ(x1 * vw, y0 * vh, planeZ);
          const c2 = this.sceneManager._intersectPlaneZ(x1 * vw, y1 * vh, planeZ);
          const c3 = this.sceneManager._intersectPlaneZ(x0 * vw, y1 * vh, planeZ);
          if (c0 && c1 && c2 && c3) {
            worldBoxCorners = [c0, c1, c2, c3];
          }
        }

        highlightRecord = {
          mode: 'point',
          layer_id: hit3d ? hit3d.layerId : 'background',
          layer_label_he: hit3d ? hit3d.layerLabelHe : 'רקע המרחב המורחב',
          screen_rect_norm: {
            x0: Number(x0.toFixed(5)),
            y0: Number(y0.toFixed(5)),
            x1: Number(x1.toFixed(5)),
            y1: Number(y1.toFixed(5))
          },
          pin_screen_norm: [Number(nx.toFixed(5)), Number(ny.toFixed(5))],
          world_point: hit3d ? hit3d.worldPoint : null,
          world_box_corners: worldBoxCorners,
          texture_uv: hit3d ? hit3d.textureUv : null,
          texture_pixel: hit3d ? hit3d.texturePixel : null,
          ui_selector: null
        };
      } else {
        const radX = pickingCfg.default_point_box_radius_norm;
        const radY = radX * (vw / vh);
        const nx = endX / vw;
        const ny = endY / vh;
        highlightRecord = {
          mode: 'ui',
          layer_id: 'ui',
          layer_label_he: 'נקודה בעמוד',
          screen_rect_norm: {
            x0: Number(Math.max(0, nx - radX).toFixed(5)),
            y0: Number(Math.max(0, ny - radY).toFixed(5)),
            x1: Number(Math.min(1, nx + radX).toFixed(5)),
            y1: Number(Math.min(1, ny + radY).toFixed(5))
          },
          pin_screen_norm: [Number(nx.toFixed(5)), Number(ny.toFixed(5))],
          world_point: null,
          world_box_corners: null,
          texture_uv: null,
          texture_pixel: null,
          ui_selector: null
        };
      }
    }

    const snapDataUrl = captureCompositeViewportSnapshot({
      sceneManager: this.sceneManager,
      highlightRecord,
      pickingCfg
    });
    this.cancelHighlightMode();

    this.activeHighlightDraft = highlightRecord;
    this.activeSnapshotDataUrl = snapDataUrl;
    this.activeTargetRefDraft = detectedTargetRef || this.currentTargetRef;
    if (this.includePoseCheck) this.includePoseCheck.checked = true;
    this.setCategory(autoCategory);
    this._renderHighlightDraftAttachment();
    this.openDrawer();
    setTimeout(() => {
      if (this.commentInput) this.commentInput.focus();
    }, 110);
  }

  clearHighlightDraft() {
    this.activeHighlightDraft = null;
    this.activeSnapshotDataUrl = null;
    this.activeUploadedImageLabel = null;
    this.activeTargetRefDraft = null;
    this._pickingFromDrawer = false;
    this._renderHighlightDraftAttachment();
  }

  async _handleUploadedImageFile(file) {
    if (!file || !file.type || !file.type.startsWith('image/')) {
      this._showStatus('נא לבחור קובץ תמונה תקין מהמכשיר.', true);
      return;
    }
    try {
      const rawDataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ''));
        reader.onerror = () => reject(new Error('קריאת התמונה נכשלה'));
        reader.readAsDataURL(file);
      });
      const img = await new Promise((resolve, reject) => {
        const el = new Image();
        el.onload = () => resolve(el);
        el.onerror = () => reject(new Error('טעינת התמונה נכשלה'));
        el.src = rawDataUrl;
      });
      const maxDim = Math.max(320, this.config.picking.snapshot_max_width_px);
      const scale = Math.min(1, maxDim / Math.max(img.naturalWidth || 1, img.naturalHeight || 1));
      const outW = Math.max(1, Math.round((img.naturalWidth || 1) * scale));
      const outH = Math.max(1, Math.round((img.naturalHeight || 1) * scale));
      const canvas = document.createElement('canvas');
      canvas.width = outW;
      canvas.height = outH;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, outW, outH);
      const jpegDataUrl = canvas.toDataURL('image/jpeg', this.config.picking.snapshot_jpeg_quality);

      this.activeHighlightDraft = null;
      this.activeSnapshotDataUrl = jpegDataUrl;
      this.activeUploadedImageLabel = `${file.name || 'צילום מסך'} (${outW}×${outH})`;
      this._renderHighlightDraftAttachment();
      this._showStatus('📷 התמונה צורפה לטופס המשוב.', false);
    } catch (err) {
      this._showStatus(`שגיאה בצירוף התמונה: ${err.message}`, true);
    }
  }

  _renderHighlightDraftAttachment() {
    if (!this.activeHighlightDraft && !this.activeSnapshotDataUrl) {
      this.attachmentBox.classList.add('hidden');
      if (this.attachActionsRow) this.attachActionsRow.style.display = 'flex';
      return;
    }
    if (this.attachActionsRow) this.attachActionsRow.style.display = 'none';
    this.attachmentBox.classList.remove('hidden');
    if (this.activeSnapshotDataUrl) {
      this.attachThumb.src = this.activeSnapshotDataUrl;
    }
    const h = this.activeHighlightDraft;
    if (!h) {
      this.attachLayerEl.textContent = '📷 תמונה / צילום מסך מצורף מהמכשיר';
      this.attachCoordsEl.textContent = this.activeUploadedImageLabel || 'תמונה מצורפת';
      return;
    }
    this.attachLayerEl.textContent = `${h.mode === 'ui' ? '🧩' : '🎯'} ${h.layer_label_he}`;
    if (h.texture_pixel) {
      this.attachCoordsEl.textContent = `px (${h.texture_pixel[0]}, ${h.texture_pixel[1]}) · ${
        h.mode === 'box' ? 'אזור 3D' : 'נקודה 3D'
      }`;
    } else if (h.ui_selector) {
      this.attachCoordsEl.textContent = h.ui_selector;
    } else {
      this.attachCoordsEl.textContent = 'מיקום במסך נשמר';
    }
  }

  _showStatus(message, isError = false) {
    this.statusMsgEl.textContent = message;
    this.statusMsgEl.className = `beta-status-msg ${isError ? 'error' : 'success'}`;
    if (!isError) {
      setTimeout(() => {
        if (this.statusMsgEl.textContent === message) {
          this.statusMsgEl.classList.add('hidden');
        }
      }, 3600);
    }
  }

  async submitFeedbackEntry() {
    const author = this.authorInput.value.trim();
    const comment = this.commentInput.value.trim();
    if (!author) {
      this._showStatus('נא לבחור או להזין שם (טל / נעמה / גדי / גיא).', true);
      this.authorInput.focus();
      return;
    }
    if (!comment) {
      this._showStatus('נא לכתוב הערה או הצעה בתיבת הטקסט.', true);
      this.commentInput.focus();
      return;
    }

    localStorage.setItem(AUTHOR_STORAGE_KEY, author);
    const shouldIncludePose =
      Boolean(this.sceneManager) &&
      ((this.includePoseCheck && this.includePoseCheck.checked) || Boolean(this.activeHighlightDraft));

    const payload = {
      author,
      category_id: this.selectedCategoryId,
      scene_id: this.currentContextSceneId,
      comment,
      camera_pose: shouldIncludePose ? this.sceneManager.getCameraPoseSnapshot() : null,
      highlight: this.activeHighlightDraft,
      snapshot_data_url: this.activeSnapshotDataUrl,
      viewport: [window.innerWidth, window.innerHeight],
      page_path: window.location.pathname,
      target_ref: this.activeTargetRefDraft || this.currentTargetRef
    };

    this.submitBtn.disabled = true;
    this.statusMsgEl.classList.add('hidden');

    try {
      const resp = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || `HTTP ${resp.status}`);
      }

      this.commentInput.value = '';
      if (typeof this.commentInput.blur === 'function') {
        this.commentInput.blur();
      }
      this.clearHighlightDraft();
      this._showStatus(`תודה ${author}! ההערה (#${data.entry.seq_num}) נשמרה.`, false);
      await this.refreshEntries();
      this.selectEntry(data.entry.entry_id, { flyCamera: false, collapseOnMobile: false });
    } catch (err) {
      this._showStatus(`שגיאה בשליחת המשוב: ${err.message}`, true);
    } finally {
      this.submitBtn.disabled = false;
    }
  }

  async refreshEntries() {
    try {
      const resp = await fetch('/api/feedback', { cache: 'no-store' });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const data = await resp.json();
      this.entries = Array.isArray(data.entries) ? data.entries : [];
      this._updateBadgeCount();
      this._syncPinDomElements();
      this._updateProjectedPinsAndHighlight();
      this._renderEntriesList();
    } catch (err) {
      console.error('Failed to refresh beta feedback entries:', err);
    }
  }

  notifyPageContentUpdated() {
    this._updateProjectedPinsAndHighlight();
  }

  _updateBadgeCount() {
    const openCount = this.entries.filter((e) => e.status === 'open').length;
    if (this.countBadge) {
      this.countBadge.textContent = String(openCount);
    }
  }

  _syncPinDomElements() {
    const validIds = new Set();
    for (const entry of this.entries) {
      if (!entry.highlight) continue;
      validIds.add(entry.entry_id);
      let pinEl = this._pinElements.get(entry.entry_id);
      if (!pinEl) {
        pinEl = document.createElement('button');
        pinEl.type = 'button';
        pinEl.className = 'beta-pin-marker';
        pinEl.addEventListener('click', (ev) => {
          ev.stopPropagation();
          this.selectEntry(entry.entry_id, { flyCamera: true, collapseOnMobile: true });
        });
        this.pinsLayer.appendChild(pinEl);
        this._pinElements.set(entry.entry_id, pinEl);
      }
      pinEl.textContent = `#${entry.seq_num}`;
      pinEl.title = `#${entry.seq_num} ${entry.author}: ${entry.comment.slice(0, 80)}`;
      pinEl.classList.toggle('ui-pin', entry.highlight.mode === 'ui');
      pinEl.classList.toggle('resolved', entry.status === 'resolved');
      pinEl.classList.toggle('selected', entry.entry_id === this.selectedEntryId);
    }

    for (const [id, el] of this._pinElements.entries()) {
      if (!validIds.has(id)) {
        el.remove();
        this._pinElements.delete(id);
      }
    }
  }

  _resolveDomElementRect(uiSelector) {
    if (!uiSelector) return null;
    try {
      const el = document.querySelector(uiSelector);
      if (!el) return null;
      const r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return null;
      return r;
    } catch (_) {
      return null;
    }
  }

  _isEntryVisibleInCurrentView(entry) {
    if (entry.status === 'resolved' && entry.entry_id !== this.selectedEntryId) {
      return false;
    }
    const currentScene = this.currentContextSceneId;
    if (entry.scene_id !== currentScene) {
      // Allow Prototype UI chrome comments (like #gyro-toggle) to show across all 3D spatial scenes,
      // EXCEPT scene-specific pill selectors (`data-scene=`) or selectors whose DOM element is currently hidden.
      const uiSel = entry.highlight && entry.highlight.mode === 'ui' ? entry.highlight.ui_selector : '';
      const isSharedPrototypeUi =
        Boolean(this.sceneManager) &&
        SPATIAL_SCENE_IDS.has(entry.scene_id) &&
        Boolean(uiSel) &&
        !uiSel.includes('data-scene=') &&
        Boolean(this._resolveDomElementRect(uiSel));
      if (!isSharedPrototypeUi) return false;
    }
    if (currentScene === 'wiki' && entry.target_ref && entry.target_ref !== this.currentTargetRef) {
      return false;
    }
    return true;
  }

  _updateProjectedPinsAndHighlight() {
    if (!this.config || !this.pinsLayer) return;
    const nearClip = this.config.picking.pin_near_clip_z;

    for (const entry of this.entries) {
      const pinEl = this._pinElements.get(entry.entry_id);
      if (!pinEl) continue;

      if (!this.showPins || !entry.highlight || !this._isEntryVisibleInCurrentView(entry)) {
        pinEl.style.display = 'none';
        continue;
      }

      const h = entry.highlight;
      if (h.mode === 'ui' && h.ui_selector) {
        const domRect = this._resolveDomElementRect(h.ui_selector);
        if (!domRect) {
          if (entry.scene_id !== this.currentContextSceneId || h.ui_selector.includes('data-scene=')) {
            pinEl.style.display = 'none';
            continue;
          }
        } else {
          if (domRect.bottom < 0 || domRect.top > window.innerHeight) {
            pinEl.style.display = 'none';
            continue;
          }
          pinEl.style.display = 'flex';
          pinEl.style.left = `${Math.min(window.innerWidth - 18, Math.max(18, domRect.right - 8))}px`;
          pinEl.style.top = `${Math.min(window.innerHeight - 18, Math.max(18, domRect.top + 8))}px`;
          continue;
        }
      }

      if (h.world_point && this.sceneManager) {
        const proj = this.sceneManager.projectWorldPointToViewport(h.world_point, nearClip);
        if (!proj || !proj.visible) {
          pinEl.style.display = 'none';
          continue;
        }
        pinEl.style.display = 'flex';
        pinEl.style.left = `${proj.clientX}px`;
        pinEl.style.top = `${proj.clientY}px`;
      } else if (h.pin_screen_norm) {
        pinEl.style.display = 'flex';
        pinEl.style.left = `${h.pin_screen_norm[0] * window.innerWidth}px`;
        pinEl.style.top = `${h.pin_screen_norm[1] * window.innerHeight}px`;
      } else {
        pinEl.style.display = 'none';
      }
    }

    const selected = this.entries.find((e) => e.entry_id === this.selectedEntryId);
    if (!selected || !selected.highlight || !this._isEntryVisibleInCurrentView(selected)) {
      this.projectedBoxEl.classList.add('hidden');
      return;
    }

    const sh = selected.highlight;
    if (sh.mode === 'ui' && sh.ui_selector) {
      const domRect = this._resolveDomElementRect(sh.ui_selector);
      if (domRect) {
        this.projectedBoxEl.classList.remove('hidden');
        this.projectedBoxEl.style.left = `${domRect.left - 3}px`;
        this.projectedBoxEl.style.top = `${domRect.top - 3}px`;
        this.projectedBoxEl.style.width = `${Math.max(18, domRect.width + 6)}px`;
        this.projectedBoxEl.style.height = `${Math.max(18, domRect.height + 6)}px`;
        return;
      }
    }

    if (sh.world_box_corners && this.sceneManager) {
      const boxProj = this.sceneManager.projectWorldBoxToViewport(sh.world_box_corners, nearClip);
      if (boxProj) {
        this.projectedBoxEl.classList.remove('hidden');
        this.projectedBoxEl.style.left = `${boxProj.left}px`;
        this.projectedBoxEl.style.top = `${boxProj.top}px`;
        this.projectedBoxEl.style.width = `${Math.max(18, boxProj.right - boxProj.left)}px`;
        this.projectedBoxEl.style.height = `${Math.max(18, boxProj.bottom - boxProj.top)}px`;
        return;
      }
    }

    if (sh.screen_rect_norm) {
      const r = sh.screen_rect_norm;
      this.projectedBoxEl.classList.remove('hidden');
      this.projectedBoxEl.style.left = `${r.x0 * window.innerWidth}px`;
      this.projectedBoxEl.style.top = `${r.y0 * window.innerHeight}px`;
      this.projectedBoxEl.style.width = `${Math.max(18, (r.x1 - r.x0) * window.innerWidth)}px`;
      this.projectedBoxEl.style.height = `${Math.max(18, (r.y1 - r.y0) * window.innerHeight)}px`;
    }
  }

  async selectEntry(entryId, { flyCamera = true, collapseOnMobile = true } = {}) {
    const entry = this.entries.find((e) => e.entry_id === entryId);
    if (!entry) return;

    // Cross-portal navigation if comment belongs to another active page (Prototype vs Wiki vs Catalog)
    const entryNeedsPrototype = SPATIAL_SCENE_IDS.has(entry.scene_id);
    const isPortalRoute = entry.scene_id === 'wiki' || entry.scene_id === 'catalog';
    if (!entryNeedsPrototype && !isPortalRoute) {
      this.selectedEntryId = entryId;
      this._syncPinDomElements();
      this._renderEntriesList();
      if (entry.snapshot_url) {
        this.openLightbox(entry.snapshot_url, `#${entry.seq_num} ${entry.author}: ${entry.comment}`);
      }
      return;
    }
    if (flyCamera) {
      if (entryNeedsPrototype && !this.sceneManager) {
        window.location.href = sceneRouteHref(entry.scene_id, entry.target_ref, entry.entry_id);
        return;
      }
      if (isPortalRoute && this.currentContextSceneId !== entry.scene_id) {
        window.location.href = sceneRouteHref(entry.scene_id, entry.target_ref, entry.entry_id);
        return;
      }
    }

    const centerHint = document.getElementById('center-hint');
    if (centerHint) centerHint.classList.add('hidden');

    this.selectedEntryId = entryId;
    this._syncPinDomElements();
    this._renderEntriesList();

    if (flyCamera) {
      if (this.sceneManager) {
        if (entry.camera_pose) {
          this.sceneManager.flyToCameraPose(entry.scene_id, entry.camera_pose, false);
        } else if (entry.scene_id !== this.sceneManager.currentSceneId) {
          this.sceneManager.transitionTo(entry.scene_id);
        }
      } else if (entry.target_ref && typeof this.options.onNavigateTargetRef === 'function') {
        await Promise.resolve(this.options.onNavigateTargetRef(entry.target_ref));
      }

      if (entry.highlight && entry.highlight.ui_selector) {
        try {
          const domTarget = document.querySelector(entry.highlight.ui_selector);
          if (domTarget && typeof domTarget.scrollIntoView === 'function') {
            domTarget.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        } catch (_) {
          // Ignore invalid selector
        }
      }
    }

    this._updateProjectedPinsAndHighlight();

    const isMobile = window.innerWidth <= 768;
    if (flyCamera && ((collapseOnMobile && isMobile) || !this.drawer.classList.contains('open'))) {
      this.closeDrawer();
      this._showFocusBar(entry);
    }
  }

  clearSelection() {
    this.selectedEntryId = null;
    this.projectedBoxEl.classList.add('hidden');
    this.focusBar.classList.add('hidden');
    this._syncPinDomElements();
    this._renderEntriesList();
  }

  _showFocusBar(entry) {
    const cat = this.config.categories.find((c) => c.category_id === entry.category_id);
    const layerInfo = entry.highlight ? ` · 🎯 ${entry.highlight.layer_label_he}` : '';
    this.focusBar.innerHTML = `
      <div class="focus-bar-top">
        <span><strong>#${entry.seq_num} ${this._escapeHtml(entry.author)}</strong> · ${
          cat ? cat.icon + ' ' + cat.label_he : ''
        }${this._escapeHtml(layerInfo)}</span>
        <span>${SCENE_LABELS_HE[entry.scene_id] || entry.scene_id}</span>
      </div>
      <div class="focus-bar-comment">${this._escapeHtml(entry.comment)}</div>
      <div class="focus-bar-actions">
        <button type="button" id="focus-clear-btn">הסתר סימון</button>
        <button type="button" class="primary" id="focus-back-drawer-btn">💬 חזרה לרשימת המשוב</button>
      </div>
    `;
    this.focusBar.classList.remove('hidden');
    this.focusBar.querySelector('#focus-clear-btn').addEventListener('click', () => {
      this.clearSelection();
    });
    this.focusBar.querySelector('#focus-back-drawer-btn').addEventListener('click', () => {
      this.focusBar.classList.add('hidden');
      this.openDrawer();
    });
  }

  async toggleResolveEntry(entry) {
    const nextStatus = entry.status === 'resolved' ? 'open' : 'resolved';
    try {
      const resp = await fetch(`/api/feedback/${encodeURIComponent(entry.entry_id)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: nextStatus })
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      await this.refreshEntries();
    } catch (err) {
      this._showStatus(`שגיאה בעדכון סטטוס: ${err.message}`, true);
    }
  }

  async deleteEntry(entry) {
    try {
      const resp = await fetch(`/api/feedback/${encodeURIComponent(entry.entry_id)}`, {
        method: 'DELETE'
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      if (this.selectedEntryId === entry.entry_id) {
        this.selectedEntryId = null;
        this.projectedBoxEl.classList.add('hidden');
      }
      await this.refreshEntries();
    } catch (err) {
      this._showStatus(`שגיאה במחיקת הערה: ${err.message}`, true);
    }
  }

  openLightbox(imgUrl, captionText) {
    const img = this.lightbox.querySelector('#beta-lightbox-img');
    const cap = this.lightbox.querySelector('#beta-lightbox-caption');
    img.src = imgUrl;
    cap.textContent = captionText || '';
    this.lightbox.classList.remove('hidden');
  }

  _escapeHtml(str) {
    return String(str || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  _renderEntriesList() {
    if (!this.entriesListEl) return;
    const filtered = this.entries
      .filter(
        (e) =>
          this.sceneFilter === 'all' ||
          e.scene_id === this.sceneFilter ||
          (typeof e.scene_id === 'string' && e.scene_id.startsWith(`${this.sceneFilter}_`))
      )
      .slice()
      .reverse();

    if (filtered.length === 0) {
      this.entriesListEl.innerHTML = `
        <div class="beta-empty-state">
          עדיין אין הערות בתצוגה זו.<br/>
          לחצו על <strong>🎯 סמן והגב</strong> כדי להדגיש כפתור ממשק, פסקה בוויקי, תמונה או מיצב!
        </div>
      `;
      return;
    }

    this.entriesListEl.innerHTML = '';
    for (const entry of filtered) {
      const cat = this.config.categories.find((c) => c.category_id === entry.category_id);
      const isResolved = entry.status === 'resolved';
      const card = document.createElement('article');
      card.dataset.entryId = entry.entry_id;
      card.className = `beta-entry-card ${
        entry.entry_id === this.selectedEntryId ? 'selected' : ''
      } ${isResolved ? 'resolved' : ''}`;

      const iconMark = entry.highlight && entry.highlight.mode === 'ui' ? '🧩' : '🎯';
      const highlightTagHtml = entry.highlight
        ? `<span class="beta-tag highlight-tag">${iconMark} ${this._escapeHtml(
            entry.highlight.layer_label_he
          )}</span>`
        : '';
      const resolvedBadgeHtml = isResolved
        ? `<span class="beta-tag resolved-status-tag" aria-label="סטטוס הערה: טופל">טופל ✓</span>`
        : '';

      const studioModes = (this.config.creative_studio && this.config.creative_studio.modes) || [];
      const studioModeSpec = entry.creation_mode
        ? studioModes.find((m) => m.mode_id === entry.creation_mode)
        : null;
      const creationTagHtml = studioModeSpec
        ? `<span class="beta-tag highlight-tag">${this._escapeHtml(
            studioModeSpec.icon
          )} ${this._escapeHtml(studioModeSpec.label_he)}</span>`
        : '';
      const mediaCountHtml =
        Array.isArray(entry.selected_media_ids) && entry.selected_media_ids.length > 0
          ? `<span class="beta-tag">📎 ${entry.selected_media_ids.length} תמונות</span>`
          : '';

      const snapHtml = entry.snapshot_url
        ? `<img class="beta-card-snap" src="${this._escapeHtml(
            entry.snapshot_url
          )}" alt="צילום סימון #${entry.seq_num}" loading="lazy" />`
        : '';

      card.innerHTML = `
        <div class="beta-card-top">
          <div class="beta-card-author">
            <span class="beta-seq-pill">#${entry.seq_num}</span>
            <span>${this._escapeHtml(entry.author)}</span>
          </div>
          <span class="beta-card-time">${this._escapeHtml(
            formatShortTimestamp(entry.created_at)
          )}</span>
        </div>
        <div class="beta-card-tags">
          ${resolvedBadgeHtml}
          <span class="beta-tag">${SCENE_LABELS_HE[entry.scene_id] || entry.scene_id}</span>
          ${
            cat
              ? `<span class="beta-tag">${cat.icon} ${this._escapeHtml(cat.label_he)}</span>`
              : ''
          }
          ${highlightTagHtml}
          ${creationTagHtml}
          ${mediaCountHtml}
        </div>
        <div class="beta-card-body">
          <div class="beta-card-comment">${this._escapeHtml(entry.comment)}</div>
          ${snapHtml}
        </div>
        <div class="beta-card-actions">
          <button type="button" class="jump-btn">
            ${entry.highlight ? '👁️ הצג סימון בהקשר' : '🧭 עבור להקשר'}
          </button>
          <div style="display:flex; gap:6px;">
            <button
              type="button"
              class="resolve-btn"
              title="${isResolved ? 'סמן הערה זו כפתוחה (לא טופלה)' : 'סמן הערה זו כטופלה'}"
            >
              ${isResolved ? '↺ סמן כפתוח' : '✓ סמן כטופל'}
            </button>
            <button type="button" class="delete-btn" title="מחק הערה">🗑️</button>
          </div>
        </div>
      `;

      card.querySelector('.jump-btn').addEventListener('click', () => {
        this.selectEntry(entry.entry_id, { flyCamera: true, collapseOnMobile: true });
      });
      card.querySelector('.resolve-btn').addEventListener('click', () => {
        this.toggleResolveEntry(entry);
      });
      card.querySelector('.delete-btn').addEventListener('click', () => {
        this.deleteEntry(entry);
      });

      const snapImg = card.querySelector('.beta-card-snap');
      if (snapImg && entry.snapshot_url) {
        snapImg.addEventListener('click', () => {
          this.openLightbox(entry.snapshot_url, `#${entry.seq_num} ${entry.author}: ${entry.comment}`);
        });
      }

      this.entriesListEl.appendChild(card);
    }
  }

  _startPolling() {
    const interval = this.config.limits.poll_interval_ms;
    this._pollTimer = setInterval(() => {
      if (!document.hidden && !this.isPicking) {
        this.refreshEntries();
      }
    }, interval);
  }

  dispose() {
    if (this._pollTimer) {
      clearInterval(this._pollTimer);
      this._pollTimer = null;
    }
    if (this._unsubscribeFrame) {
      this._unsubscribeFrame();
      this._unsubscribeFrame = null;
    }
    if (this._onScrollOrResize) {
      window.removeEventListener('scroll', this._onScrollOrResize);
      window.removeEventListener('resize', this._onScrollOrResize);
    }
    if (this._onEscapeKey) {
      window.removeEventListener('keydown', this._onEscapeKey);
    }
    if (this.pinsLayer) this.pinsLayer.remove();
    if (this.highlightSurface) this.highlightSurface.remove();
    if (this.hoverInspectorEl) this.hoverInspectorEl.remove();
    if (this.highlightBanner) this.highlightBanner.remove();
    if (this.focusBar) this.focusBar.remove();
    if (this.drawer) this.drawer.remove();
    if (this.lightbox) this.lightbox.remove();
  }
}
