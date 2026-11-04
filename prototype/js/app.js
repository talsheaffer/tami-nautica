// prototype/js/app.js
// Main application entry point for Tami-Nautica (תמי-נאוטיקה) — Tamar "Tami" Sheaffer @ 70 2.5D Spatial Prototype
// Manages Exhibition Group navigation, 3D Shelf Portal micro-badges & hover plaques,
// rotating warm Hebrew loading screen, curatorial drawer state, and procedural Web Audio ambient soundscape.

import { SpatialSceneManager, EXHIBITION_GROUPS } from './scene_manager.js';
import { BetaFeedbackOverlay } from './feedback_overlay.js';
import { mountBetaFontChipSelector } from './beta_font.js';
import { mountThemeSelector, mountUiStyleSelector } from './theme.js';

export { EXHIBITION_GROUPS };

export const LOADING_SCREEN_SPEC = Object.freeze({
  brandTitleHe: 'תמי-נאוטיקה',
  brandTitleEn: 'Tami-Nautica',
  messagesHe: Object.freeze([
    'טוען מרחבים...',
    'משחק עם יערה וכרמל...',
    'דואג לעתיד המדינה...',
    'מגייס את גדי לשיפצורים אחרונים...'
  ]),
  rotationIntervalMs: 2200,
  minFirstLineDisplayMs: 700,
  switchHalfFadeMs: 220,
  minHumanDisplayMs: 7400,
  mobileCompactViewportMaxPx: 560,
  mobileMaxFullBreadcrumbs: 4,
  sessionCycleStorageKey: 'tami_loading_cycle_offset'
});

const EXHIBITION_SCENE_SEQUENCE = Object.freeze(
  EXHIBITION_GROUPS.flatMap((group) => group.sceneIds)
);

const PORTAL_SHORT_BADGE_LABELS_HE = Object.freeze({
  cubby_capsules: 'קפסולות נייר',
  cubby_plan_b_v3: 'תמי ויערה',
  cubby_scroll_foot: 'מגילה וכף רגל',
  cubby_wandering_1: 'שיטוט 1',
  cubby_bene_beitcha: 'בנה ביתך',
  cubby_bar_giora: 'בר גיורא',
  cubby_paris_table: 'פאריס · שולחן לבן',
  cubby_wandering_2: 'שיטוט 2',
  cubby_denya_model: 'דגם דניה',
  cubby_marseille: 'מרסיי',
  cubby_fried_house: 'בית פריד',
  yard_railing_portal: 'תמי ואראלה במעקה',
  fried_house_window_wandering_1: 'שיטוט 1 בחלון',
  portal_denya_dining_wandering_2: 'שיטוט 2',
  portal_denya_couch_wandering_2: 'שיטוט 2',
  portal_denya_to_couch: 'לספה והגיטרה',
  portal_denya_back_to_dining: 'לשולחן האוכל',
  portal_tzofen_tree_window: 'חלון אל העץ',
  portal_tzofen_v2_tree_window: 'חלון אל העץ',
  portal_wandering1_scroll_world: 'לתוך המגילה',
  portal_scroll_foot_purim_1966: 'אל פורים 1966',
  portal_bar_giora_vitrine: 'אל הוויטרינה',
  portal_paris_table_tami_1958: 'אל תמי 1958'
});

const SCENE_BREADCRUMB_LABELS_HE = Object.freeze({
  shelf_portal: 'קיר המדפים',
  paper_capsules: 'קפסולות נייר',
  tzofen_brick_table: 'צופן',
  tzofen_brick_table_v2: 'צופן (הפרדה)',
  plan_b_pipe_v3: 'תמי ויערה',
  plan_b_pipe: 'צינור Plan B',
  bene_beitcha: 'בנה ביתך',
  bene_beitcha_scroll_foot: 'מגילה וכף רגל',
  purim_costume_1966: 'תחפושת פורים, 1966',
  vitrine_keys_pen: 'ויטרינת המפתחות',
  humpty_dumpty_booth: 'האמפטי דאמפטי בתא',
  humpty_dumpty_bundle: 'צרור בבד וחבל',
  denya_model: 'דגם דניה',
  denya_living_room: 'סלון דניה',
  denya_couch_guitar: 'ספת דניה',
  wandering_drawing_1: 'שיטוט 1',
  scroll_palm_world: 'עולם המגילה',
  wandering_drawing: 'שיטוט 2',
  fried_yard_installation: 'מיצב בחצר פריד',
  fried_house_railing: 'בית פריד במעקה',
  fried_table_hybrid: 'שולחן בבית פריד',
  herzliya_lawn_carpet: 'הדשא והשטיח',
  herzliya_tree_canopy: 'צמרת העץ',
  bar_giora_bedroom: 'חדר השינה בבר גיורא',
  paris_cite_white_table: 'סטודיו פריז',
  toddler_tami_1958: 'תמי הפעוטה, 1958',
  marseille_mirror: 'מרסיי',
  plan_b_pipe_v1: 'צינור Plan B v1',
  marseille_mirror_v1: 'מרסיי v1',
  herzliya_studio: 'גינת הרצליה v1',
  carmel_wadi: 'ואדי הכרמל v1'
});

const DEFAULT_CENTER_HINT_HE =
  'גרירה לסיבוב מבט · גלגלת / WASD להליכה · Q/E לגובה';
const PORTAL_CENTER_HINT_HE =
  'לחצו על חפצים להסרתם ועל תא פתוח לצלילה פנימה · WASD / Q/E לניווט';
const YARD_PORTAL_CENTER_HINT_HE =
  'לחצו על המיצב להסרתו ולחשיפת תמי ואראלה · לחצו עליהן למעבר לסצנת המעקה';
const FRIED_RAILING_PORTAL_HINT_HE =
  'לחצו על תריס העץ בשמאל הקיר לחשיפת שער אל ׳שיטוט 1׳';
const DENYA_PORTAL_CENTER_HINT_HE =
  'לחצו על הספה או על מסגרת ׳שיטוט 2׳ למעבר בין המרחבים';
const WANDERING1_SCROLL_HINT_HE =
  'לחצו על המגילה שמימין לצלילה לתוך עולם המגילה';
const TZOFEN_PORTAL_HINT_HE =
  'לחצו על מסגרת החלון הלבנה לצלילה אל צמרת העץ בהרצליה';
const SCROLL_FOOT_HINT_HE =
  'לחצו על כף הרגל להסרתה ולהפלת המוט השחור · לחצו על צילום פורים לצלילה פנימה';
const LAWN_CARPET_HINT_HE =
  'לחצו על גליל השטיח שעל הדשא כדי לגלגל ולפרוש את שובל האדמה לרוחב הגינה';
const BEDROOM_VITRINE_HINT_HE =
  'לחצו על ויטרינת המפתחות שעל הקיר לצלילה פנימה עם סיבוב מצלמה תואם';
const PARIS_TABLE_HINT_HE =
  'לחצו על צילום הילדות שעל השולחן הלבן לצלילה אל תמי הפעוטה (1958)';

function escapeHtmlText(raw) {
  return String(raw || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

class SpatialApp {
  constructor() {
    this.canvas = document.getElementById('spatial-canvas');
    this.loadingOverlay = document.getElementById('loading-overlay');
    this.loadingStatusTextEl =
      document.getElementById('loading-status-text') ||
      (this.loadingOverlay ? this.loadingOverlay.querySelector('.loading-text') : null);
    this.loadingSkipHintEl = document.getElementById('loading-skip-hint');
    this._loadingFirstTimer = null;
    this._loadingRotationTimer = null;
    this._loadingSwitchTimer = null;
    this._loadingHoldTimer = null;
    this._loadingDismissed = false;
    this._onSkipLoadingEarly = null;
    this.topBarEl = document.querySelector('.top-bar');
    this.spatialNavBar = document.getElementById('spatial-nav-bar');
    this.spatialBackBtn = document.getElementById('spatial-back-btn');
    this.spatialForwardBtn = document.getElementById('spatial-forward-btn');
    this.spatialBreadcrumbList = document.getElementById('spatial-breadcrumb-list');
    this.spatialOutgoingPortals = document.getElementById('spatial-outgoing-portals');
    this.bottomDockEl = document.querySelector('.bottom-dock');
    this.spatialSidebar = document.getElementById('spatial-sidebar');
    this.sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
    this.sidebarCloseBtn = document.getElementById('sidebar-close-btn');
    this.betaFontSelectorEl = document.getElementById('beta-font-selector');
    this.centerHint = document.getElementById('center-hint');
    this._centerHintTimer = null;
    this.infoDrawer = document.getElementById('info-drawer');
    this.infoToggleBtn = document.getElementById('info-toggle-btn');
    this.closeDrawerBtn = document.getElementById('close-drawer-btn');
    this.fullscreenBtn = document.getElementById('fullscreen-btn');
    this.audioToggleBtn = document.getElementById('audio-toggle');
    this.gyroToggleBtn = document.getElementById('gyro-toggle');
    this.resetPoseBtn = document.getElementById('reset-pose-btn');
    this.restoreOccludersBtn = document.getElementById('restore-occluders-btn');
    this.portalReturnBtn = document.getElementById('portal-return-btn');
    this.portalBadgesLayer = document.getElementById('portal-badges-layer');
    this.portalHoverPlaque = document.getElementById('portal-hover-plaque');
    this.exhibitionPrevBtn = document.getElementById('exhibition-prev-btn');
    this.exhibitionNextBtn = document.getElementById('exhibition-next-btn');
    this.exhibitionTabBtns = document.querySelectorAll('.exhibition-tab-btn');
    this.sceneBtns = document.querySelectorAll('.scene-btn');
    this._portalBadgeEls = new Map();
    this._portalBadgesHostSceneId = null;
    this._activeGroupId = 'portal_hub';
    this._navHistory = [];
    this._navHistoryIndex = -1;
    this._suppressHistoryPush = false;
    this.audioContext = null;
    this.isAudioPlaying = false;

    if (typeof window !== 'undefined') {
      window.__tamiApp = this;
    }

    this._initLoadingScreenRotation();
    mountBetaFontChipSelector(this.betaFontSelectorEl);
    mountThemeSelector(document.getElementById('theme-selector'));
    mountUiStyleSelector(document.getElementById('ui-style-selector'));
    this.initSceneManager();
    this._buildPortalMicroBadges();
    this.updateActiveSceneUI(this.manager.currentSceneId);
    this.feedbackOverlay = new BetaFeedbackOverlay(this.manager);
    this.initUIListeners();
  }

  _shouldHoldLoadingForReading() {
    if (typeof window === 'undefined' || !window.location) return false;
    const params = new URLSearchParams(window.location.search);
    if (params.get('fastLoad') === '1') return false;
    if (params.get('showLoading') === '1') return true;
    const isWebdriver = typeof navigator !== 'undefined' && navigator.webdriver === true;
    const ua = typeof navigator !== 'undefined' ? String(navigator.userAgent || '') : '';
    const isAutomatedHeadless = isWebdriver || ua.includes('HeadlessChrome');
    return !isAutomatedHeadless;
  }

  _buildLoadingMessageSequence() {
    const all = LOADING_SCREEN_SPEC.messagesHe;
    if (all.length <= 1) return [...all];
    const warmTail = all.slice(1);
    let offset = 0;
    try {
      const raw = sessionStorage.getItem(LOADING_SCREEN_SPEC.sessionCycleStorageKey);
      const parsed = raw !== null ? Number.parseInt(raw, 10) : 0;
      if (Number.isFinite(parsed) && parsed >= 0) {
        offset = parsed % warmTail.length;
      }
      sessionStorage.setItem(
        LOADING_SCREEN_SPEC.sessionCycleStorageKey,
        String((offset + 1) % warmTail.length)
      );
    } catch (_err) {
      offset = 0;
    }
    const rotatedTail = warmTail.map((_, i) => warmTail[(i + offset) % warmTail.length]);
    return [all[0], ...rotatedTail];
  }

  _advanceLoadingScreenLine() {
    if (this._loadingDismissed || !this.loadingStatusTextEl) return;
    this._loadingSeqIdx = (this._loadingSeqIdx + 1) % this._loadingSequence.length;
    const nextText = this._loadingSequence[this._loadingSeqIdx];
    this.loadingStatusTextEl.classList.add('is-switching');
    if (this._loadingSwitchTimer) {
      clearTimeout(this._loadingSwitchTimer);
    }
    this._loadingSwitchTimer = setTimeout(() => {
      this._loadingSwitchTimer = null;
      if (this._loadingDismissed || !this.loadingStatusTextEl) return;
      this.loadingStatusTextEl.textContent = nextText;
      this.loadingStatusTextEl.classList.remove('is-switching');
    }, LOADING_SCREEN_SPEC.switchHalfFadeMs);
  }

  _initLoadingScreenRotation() {
    if (!this.loadingOverlay || !this.loadingStatusTextEl) return;
    this._loadingSequence = this._buildLoadingMessageSequence();
    this._loadingSeqIdx = 0;
    this.loadingStatusTextEl.textContent = this._loadingSequence[0];

    const navElapsedMs = typeof performance !== 'undefined' ? performance.now() : 0;
    const firstDelayMs = Math.max(
      LOADING_SCREEN_SPEC.minFirstLineDisplayMs,
      LOADING_SCREEN_SPEC.rotationIntervalMs - navElapsedMs
    );
    this._loadingFirstTimer = setTimeout(() => {
      this._loadingFirstTimer = null;
      this._advanceLoadingScreenLine();
      if (!this._loadingDismissed) {
        this._loadingRotationTimer = setInterval(() => {
          this._advanceLoadingScreenLine();
        }, LOADING_SCREEN_SPEC.rotationIntervalMs);
      }
    }, firstDelayMs);
  }

  _dismissLoadingOverlayNow() {
    if (this._loadingDismissed) return;
    this._loadingDismissed = true;
    if (this._loadingFirstTimer) {
      clearTimeout(this._loadingFirstTimer);
      this._loadingFirstTimer = null;
    }
    if (this._loadingRotationTimer) {
      clearInterval(this._loadingRotationTimer);
      this._loadingRotationTimer = null;
    }
    if (this._loadingSwitchTimer) {
      clearTimeout(this._loadingSwitchTimer);
      this._loadingSwitchTimer = null;
    }
    if (this._loadingHoldTimer) {
      clearTimeout(this._loadingHoldTimer);
      this._loadingHoldTimer = null;
    }
    if (this._onSkipLoadingEarly && this.loadingOverlay) {
      this.loadingOverlay.removeEventListener('click', this._onSkipLoadingEarly);
      this.loadingOverlay.removeEventListener('touchstart', this._onSkipLoadingEarly);
      window.removeEventListener('keydown', this._onSkipLoadingEarly);
      this._onSkipLoadingEarly = null;
    }
    if (this.loadingOverlay) {
      this.loadingOverlay.classList.remove('ready-to-skip');
      this.loadingOverlay.classList.add('hidden');
    }
  }

  _handleLayersReady() {
    if (this._loadingDismissed) return;

    if (this.loadingSkipHintEl) {
      this.loadingSkipHintEl.textContent = 'המרחב מוכן · לחצו בכל מקום כדי להתחיל';
      this.loadingSkipHintEl.classList.remove('hidden');
    }

    if (this.loadingOverlay) {
      this.loadingOverlay.classList.add('ready-to-skip');
      this._onSkipLoadingEarly = (e) => {
        if (e && typeof e.stopPropagation === 'function') {
          e.stopPropagation();
        }
        this._dismissLoadingOverlayNow();
      };
      this.loadingOverlay.addEventListener('click', this._onSkipLoadingEarly, { once: true });
      this.loadingOverlay.addEventListener('touchstart', this._onSkipLoadingEarly, {
        once: true,
        passive: false
      });
      window.addEventListener('keydown', this._onSkipLoadingEarly, { once: true });
    }
  }

  initSceneManager() {
    this.manager = new SpatialSceneManager(this.canvas, {
      onReady: () => {
        this._handleLayersReady();
      },
      onSceneChange: (sceneId) => {
        this.updateActiveSceneUI(sceneId);
      },
      onPortalHover: (_spec, _clientX, _clientY) => {},
      onPortalUnlock: (spec) => {
        this._handlePortalUnlock(spec);
      }
    });

    this.manager.addFrameListener(() => {
      this._updatePortalBadgesProjection();
    });

    window.__SPATIAL_MANAGER__ = this.manager;
    window.__tamiSceneManager = this.manager;
  }

  _findGroupForScene(sceneId) {
    return (
      EXHIBITION_GROUPS.find((group) => group.sceneIds.includes(sceneId)) || EXHIBITION_GROUPS[1]
    );
  }

  _isScenePortalHost(sceneId) {
    if (!this.manager) return false;
    if (typeof this.manager.isPortalHostScene === 'function') {
      return this.manager.isPortalHostScene(sceneId);
    }
    return sceneId === 'shelf_portal';
  }

  _syncBadgeLabelAndLockState(badgeBtn, spec) {
    if (!badgeBtn || !spec || !this.manager) return;
    const isLocked =
      typeof this.manager.isPortalLocked === 'function'
        ? this.manager.isPortalLocked(spec.id)
        : false;
    const isCompactPin = this.manager.currentSceneId === 'shelf_portal';
    badgeBtn.classList.toggle('compact-pin', isCompactPin);
    const pinState = isLocked ? 'locked' : 'open';
    if (badgeBtn.dataset.pinState !== pinState) {
      badgeBtn.dataset.pinState = pinState;
      badgeBtn.innerHTML = '';
      const iconSpan = document.createElement('span');
      iconSpan.className = 'badge-pin-icon';
      iconSpan.setAttribute('aria-hidden', 'true');
      iconSpan.textContent = isLocked ? '🔒' : '↗';
      badgeBtn.appendChild(iconSpan);
    }
    badgeBtn.classList.toggle('locked', isLocked);
    badgeBtn.setAttribute(
      'aria-label',
      isLocked ? `הסר חפץ מסתיר ופתח את ${spec.labelHe}` : `כניסה אל ${spec.labelHe}`
    );
  }

  _handlePortalUnlock(spec) {
    this._syncRestoreOccludersBtn();
    if (!spec) return;
    const isNowLocked =
      this.manager && typeof this.manager.isPortalLocked === 'function'
        ? this.manager.isPortalLocked(spec.id)
        : false;
    if (!isNowLocked && this.centerHint) {
      this.centerHint.classList.add('hidden');
    }
    const badgeEl = this._portalBadgeEls.get(spec.id);
    if (badgeEl) {
      this._syncBadgeLabelAndLockState(badgeEl, spec);
      badgeEl.classList.remove('just-unlocked');
      if (!isNowLocked) {
        void badgeEl.offsetWidth;
        badgeEl.classList.add('just-unlocked');
      }
    }
    if (this.manager) {
      this._renderSpatialBreadcrumbs(this.manager.currentSceneId);
    }
  }

  _syncRestoreOccludersBtn() {
    if (!this.restoreOccludersBtn || !this.manager) return;
    const sceneId = this.manager.currentSceneId;
    const erasedCount =
      typeof this.manager.getErasedPortalOccluderCount === 'function'
        ? this.manager.getErasedPortalOccluderCount(sceneId)
        : 0;
    const showRestore = this._isScenePortalHost(sceneId) && erasedCount > 0;
    this.restoreOccludersBtn.classList.toggle('hidden', !showRestore);
    if (showRestore) {
      let label = 'השב חפצים למדפים';
      if (sceneId === 'fried_yard_installation') {
        label = 'השב את המיצב לפני המעקה';
      } else if (sceneId === 'fried_house_railing') {
        label = 'סגור את תריס העץ בחלון';
      } else if (sceneId === 'bene_beitcha_scroll_foot') {
        label = 'השב את כף הרגל והמוט';
      }
      this.restoreOccludersBtn.textContent = label;
      this.restoreOccludersBtn.setAttribute('aria-label', label);
    }
  }

  _buildPortalMicroBadges() {
    if (!this.portalBadgesLayer || !this.manager) return;
    this.portalBadgesLayer.innerHTML = '';
    this._portalBadgeEls.clear();
    this._portalBadgesHostSceneId = this.manager.currentSceneId;

    const shelves = this.manager.getPortalShelves(this.manager.currentSceneId);
    for (const spec of shelves) {
      if (spec.showMicroBadge === false) continue;
      const badgeBtn = document.createElement('button');
      badgeBtn.type = 'button';
      badgeBtn.className = 'portal-micro-badge';
      badgeBtn.setAttribute('data-portal-id', spec.id);
      badgeBtn.setAttribute('data-target-scene', spec.targetSceneId);
      this._syncBadgeLabelAndLockState(badgeBtn, spec);

      badgeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (!this.manager) return;
        if (
          typeof this.manager.isPortalLocked === 'function' &&
          this.manager.isPortalLocked(spec.id)
        ) {
          this.manager.erasePortalOccluder(spec.id);
        } else {
          this.manager.flyIntoPortalShelf(spec.id);
        }
      });

      badgeBtn.addEventListener('mouseenter', () => {});

      badgeBtn.addEventListener('mouseleave', () => {});

      badgeBtn.addEventListener('focus', () => {});

      badgeBtn.addEventListener('blur', () => {});

      this.portalBadgesLayer.appendChild(badgeBtn);
      this._portalBadgeEls.set(spec.id, badgeBtn);
    }
  }

  _updatePortalBadgesProjection() {
    if (!this.portalBadgesLayer || !this.manager) return;

    const showBadges =
      this._isScenePortalHost(this.manager.currentSceneId) &&
      !this.manager.isTransitioning &&
      !this.manager._portalFlight &&
      !this.manager.interactionCaptureActive;

    this.portalBadgesLayer.classList.toggle('hidden', !showBadges);
    if (!showBadges) return;

    const shelves = this.manager.getPortalShelves(this.manager.currentSceneId);
    const badgeSpecs = shelves.filter((s) => s.showMicroBadge !== false);
    if (
      this._portalBadgesHostSceneId !== this.manager.currentSceneId ||
      badgeSpecs.length !== this._portalBadgeEls.size
    ) {
      this._buildPortalMicroBadges();
    }

    const vw = window.innerWidth || 1440;
    const vh = window.innerHeight || 900;
    let safeMinX = 24;
    let safeMinY = 56;
    let safeMaxY = vh - 24;
    if (this.spatialSidebar && !this.spatialSidebar.classList.contains('collapsed')) {
      const sbRect = this.spatialSidebar.getBoundingClientRect();
      if (sbRect.width > 0) {
        safeMinX = Math.max(24, sbRect.right + 12);
      }
    }
    if (this.topBarEl) {
      const topRect = this.topBarEl.getBoundingClientRect();
      if (topRect.height > 0) {
        safeMinY = Math.max(28, topRect.bottom + 12);
      }
    }
    if (this.spatialNavBar) {
      const navRect = this.spatialNavBar.getBoundingClientRect();
      if (navRect.height > 0) {
        safeMinY = Math.max(safeMinY, navRect.bottom + 12);
      }
    }
    if (this.bottomDockEl) {
      const botRect = this.bottomDockEl.getBoundingClientRect();
      if (botRect.height > 0) {
        safeMaxY = Math.min(vh - 24, botRect.top - 12);
      }
    }

    for (const spec of shelves) {
      const badgeEl = this._portalBadgeEls.get(spec.id);
      if (!badgeEl) continue;

      this._syncBadgeLabelAndLockState(badgeEl, spec);

      const isShelfHub = this.manager.currentSceneId === 'shelf_portal';
      const xOffsetFactor =
        typeof spec.badgeOffsetXFactor === 'number'
          ? spec.badgeOffsetXFactor
          : isShelfHub
            ? 0.34
            : 0.0;
      const yOffsetFactor =
        typeof spec.badgeOffsetYFactor === 'number'
          ? spec.badgeOffsetYFactor
          : isShelfHub
            ? -0.36
            : -0.44;
      const lipWorld = [
        spec.worldX + spec.width * xOffsetFactor,
        spec.worldY + spec.height * yOffsetFactor,
        spec.worldZ
      ];
      const proj = this.manager.projectWorldPointToViewport(lipWorld, 20.0);
      const inSafeViewport =
        proj &&
        proj.visible &&
        proj.clientX >= safeMinX &&
        proj.clientX <= vw - 24 &&
        proj.clientY >= safeMinY &&
        proj.clientY <= safeMaxY;

      if (!inSafeViewport) {
        badgeEl.style.display = 'none';
        continue;
      }

      badgeEl.style.display = 'inline-flex';
      badgeEl.style.left = `${proj.clientX.toFixed(1)}px`;
      badgeEl.style.top = `${proj.clientY.toFixed(1)}px`;
      badgeEl.classList.toggle('hovered', this.manager.hoveredPortalId === spec.id);
    }
  }

  updatePortalHoverPlaque(_spec, _clientX, _clientY) {
    if (this.portalHoverPlaque) {
      this.portalHoverPlaque.classList.add('hidden');
    }
  }

  _setActiveExhibitionGroup(groupId, navigateIfOutside = false) {
    const group = EXHIBITION_GROUPS.find((g) => g.id === groupId);
    if (!group) return;
    this._activeGroupId = group.id;

    this.exhibitionTabBtns.forEach((tabBtn) => {
      const isActiveGroup = tabBtn.getAttribute('data-exhibition-group') === group.id;
      tabBtn.classList.toggle('active', isActiveGroup);
      tabBtn.setAttribute('aria-selected', isActiveGroup ? 'true' : 'false');
      if (isActiveGroup && typeof tabBtn.scrollIntoView === 'function') {
        tabBtn.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
      }
    });

    this.sceneBtns.forEach((btn) => {
      const btnScene = btn.getAttribute('data-scene');
      const inGroup = group.sceneIds.includes(btnScene);
      btn.classList.toggle('in-active-group', inGroup);
    });

    if (navigateIfOutside) {
      const groupHeading = this.spatialSidebar
        ? this.spatialSidebar.querySelector(`[data-group-heading="${group.id}"]`)
        : null;
      if (groupHeading && typeof groupHeading.scrollIntoView === 'function') {
        groupHeading.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
      if (this.manager && !group.sceneIds.includes(this.manager.currentSceneId)) {
        const primarySceneId = group.sceneIds[0];
        if (primarySceneId) {
          this.manager.transitionTo(primarySceneId);
        }
      }
    }
  }

  _stepAdjacentScene(delta) {
    if (!this.manager || EXHIBITION_SCENE_SEQUENCE.length === 0) return;
    const activeScene = this.manager.targetSceneId || this.manager.currentSceneId;
    const idx = EXHIBITION_SCENE_SEQUENCE.indexOf(activeScene);
    const baseIdx = idx >= 0 ? idx : 0;
    const nextIdx =
      (baseIdx + delta + EXHIBITION_SCENE_SEQUENCE.length) % EXHIBITION_SCENE_SEQUENCE.length;
    const nextSceneId = EXHIBITION_SCENE_SEQUENCE[nextIdx];
    if (nextSceneId) {
      this.manager.transitionTo(nextSceneId);
    }
  }

  _getSceneShortLabelHe(sceneId) {
    if (SCENE_BREADCRUMB_LABELS_HE[sceneId]) {
      return SCENE_BREADCRUMB_LABELS_HE[sceneId];
    }
    if (this.manager && this.manager.sceneConfigs && this.manager.sceneConfigs[sceneId]) {
      return this.manager.sceneConfigs[sceneId].titleHe || sceneId;
    }
    return sceneId;
  }

  _getCurrentPortalStackSnapshot() {
    if (this.manager && typeof this.manager.getPortalAncestryStack === 'function') {
      return this.manager.getPortalAncestryStack();
    }
    return [];
  }

  _recordNavigationHistory(sceneId) {
    if (!sceneId) return;
    const portalStack = this._getCurrentPortalStackSnapshot();
    if (this._suppressHistoryPush) {
      this._suppressHistoryPush = false;
      this._syncHistoryButtons();
      return;
    }

    const currentEntry =
      this._navHistoryIndex >= 0 && this._navHistoryIndex < this._navHistory.length
        ? this._navHistory[this._navHistoryIndex]
        : null;
    if (currentEntry && currentEntry.sceneId === sceneId) {
      currentEntry.portalStack = portalStack;
      this._syncHistoryButtons();
      return;
    }

    if (this._navHistoryIndex < this._navHistory.length - 1) {
      this._navHistory = this._navHistory.slice(0, this._navHistoryIndex + 1);
    }
    this._navHistory.push({ sceneId, portalStack });
    if (this._navHistory.length > 64) {
      this._navHistory.shift();
    }
    this._navHistoryIndex = this._navHistory.length - 1;
    this._syncHistoryButtons();
  }

  _syncHistoryButtons() {
    const canBack = this._navHistoryIndex > 0;
    const canForward =
      this._navHistoryIndex >= 0 && this._navHistoryIndex < this._navHistory.length - 1;
    if (this.spatialBackBtn) {
      this.spatialBackBtn.disabled = !canBack;
      if (canBack) {
        const prevLabel = this._getSceneShortLabelHe(
          this._navHistory[this._navHistoryIndex - 1].sceneId
        );
        this.spatialBackBtn.title = `חזרה אל ${prevLabel}`;
      } else {
        this.spatialBackBtn.title = 'חזרה למרחב הקודם בהיסטוריה';
      }
    }
    if (this.spatialForwardBtn) {
      this.spatialForwardBtn.disabled = !canForward;
      if (canForward) {
        const nextLabel = this._getSceneShortLabelHe(
          this._navHistory[this._navHistoryIndex + 1].sceneId
        );
        this.spatialForwardBtn.title = `קדימה אל ${nextLabel}`;
      } else {
        this.spatialForwardBtn.title = 'קדימה למרחב הבא בהיסטוריה';
      }
    }
  }

  navigateHistory(delta) {
    if (!this.manager || this.manager.isTransitioning || this.manager._portalFlight) return;
    const nextIdx = this._navHistoryIndex + delta;
    if (nextIdx < 0 || nextIdx >= this._navHistory.length) return;

    const targetEntry = this._navHistory[nextIdx];
    const currentStack = this._getCurrentPortalStackSnapshot();
    this._navHistoryIndex = nextIdx;
    this._suppressHistoryPush = true;
    this._syncHistoryButtons();

    // Smooth reverse 3D flight when stepping back to immediate portal parent
    if (
      delta === -1 &&
      currentStack.length > 0 &&
      currentStack[currentStack.length - 1].hostSceneId === targetEntry.sceneId &&
      typeof this.manager.returnToPortalHub === 'function'
    ) {
      const ok = this.manager.returnToPortalHub();
      if (ok !== false) return;
    }

    // Smooth forward 3D portal flight when stepping forward into an unlocked child portal
    if (
      delta === 1 &&
      Array.isArray(targetEntry.portalStack) &&
      targetEntry.portalStack.length > 0
    ) {
      const lastHop = targetEntry.portalStack[targetEntry.portalStack.length - 1];
      if (
        lastHop &&
        lastHop.hostSceneId === this.manager.currentSceneId &&
        lastHop.portalId &&
        typeof this.manager.isPortalLocked === 'function' &&
        !this.manager.isPortalLocked(lastHop.portalId)
      ) {
        const ok = this.manager.flyIntoPortalShelf(lastHop.portalId);
        if (ok) return;
      }
    }

    if (typeof this.manager.setPortalAncestryStack === 'function') {
      this.manager.setPortalAncestryStack(targetEntry.portalStack || []);
    }
    this.manager.transitionTo(targetEntry.sceneId, { preservePortalStack: true });
  }

  _appendBreadcrumbItem(
    listEl,
    { label, title = '', onClick = null, isCurrent = false, addSeparator = false }
  ) {
    if (addSeparator) {
      const sepSpan = document.createElement('span');
      sepSpan.className = 'spatial-crumb-sep';
      sepSpan.setAttribute('aria-hidden', 'true');
      sepSpan.textContent = '‹';
      listEl.appendChild(sepSpan);
    }

    const itemWrap = document.createElement('span');
    itemWrap.className = 'spatial-crumb-item';
    itemWrap.setAttribute('role', 'listitem');
    if (isCurrent) {
      const curBtn = document.createElement('button');
      curBtn.type = 'button';
      curBtn.className = 'spatial-crumb-current';
      curBtn.setAttribute('aria-current', 'page');
      curBtn.tabIndex = -1;
      curBtn.textContent = label;
      if (title) curBtn.title = title;
      itemWrap.appendChild(curBtn);
    } else {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'spatial-crumb-btn';
      btn.textContent = label;
      if (title) btn.title = title;
      if (typeof onClick === 'function') {
        btn.addEventListener('click', onClick);
      }
      itemWrap.appendChild(btn);
    }
    listEl.appendChild(itemWrap);
  }

  _populateBreadcrumbElements(items) {
    if (!this.spatialBreadcrumbList) return;
    this.spatialBreadcrumbList.innerHTML = '';
    items.forEach((item, idx) => {
      this._appendBreadcrumbItem(this.spatialBreadcrumbList, {
        ...item,
        addSeparator: idx > 0
      });
    });
  }

  _fitBreadcrumbsToCompactViewport(fullItems) {
    const listEl = this.spatialBreadcrumbList;
    if (!listEl || fullItems.length < 3) return;
    const isCompactMobile =
      typeof window !== 'undefined' &&
      window.innerWidth <= LOADING_SCREEN_SPEC.mobileCompactViewportMaxPx;
    if (!isCompactMobile) return;
    if (listEl.clientWidth <= 0) return;

    if (
      fullItems.length > LOADING_SCREEN_SPEC.mobileMaxFullBreadcrumbs ||
      listEl.scrollWidth > listEl.clientWidth + 2
    ) {
      const collapsedFromIdx = fullItems.length - 3;
      const hiddenLabels = fullItems
        .slice(1, fullItems.length - 2)
        .map((c) => c.label)
        .join(' ‹ ');
      const fourItemView =
        fullItems.length >= 4
          ? [
              fullItems[0],
              {
                label: '…',
                title: hiddenLabels ? `חזרה אל ${hiddenLabels}` : 'מרחבי ביניים',
                onClick: fullItems[collapsedFromIdx].onClick
              },
              fullItems[fullItems.length - 2],
              fullItems[fullItems.length - 1]
            ]
          : fullItems;
      this._populateBreadcrumbElements(fourItemView);

      if (listEl.scrollWidth > listEl.clientWidth + 2 && fullItems.length >= 3) {
        const parentItem = fullItems[fullItems.length - 2];
        const threeItemView = [
          fullItems[0],
          {
            label: '…',
            title: `חזרה אל ${parentItem.label}`,
            onClick: parentItem.onClick
          },
          fullItems[fullItems.length - 1]
        ];
        this._populateBreadcrumbElements(threeItemView);
      }
    }
  }

  _scrollCurrentBreadcrumbIntoView() {
    const listEl = this.spatialBreadcrumbList;
    if (!listEl || typeof listEl.scrollBy !== 'function') return;
    const currentEl = listEl.querySelector('.spatial-crumb-current');
    if (!currentEl) return;
    const listRect = listEl.getBoundingClientRect();
    const curRect = currentEl.getBoundingClientRect();
    if (listRect.width <= 0 || curRect.width <= 0) return;
    const padPx = 8;
    if (curRect.left < listRect.left + padPx) {
      listEl.scrollBy({ left: curRect.left - (listRect.left + padPx), behavior: 'auto' });
    } else if (curRect.right > listRect.right - padPx) {
      listEl.scrollBy({ left: curRect.right - (listRect.right - padPx), behavior: 'auto' });
    }
  }

  _renderSpatialBreadcrumbs(sceneId) {
    if (!this.spatialBreadcrumbList || !this.manager) return;

    const portalStack = this._getCurrentPortalStackSnapshot();
    const matchedGroup = this._findGroupForScene(sceneId);
    const currentLabel = this._getSceneShortLabelHe(sceneId);
    const crumbItems = [];

    if (portalStack.length > 0) {
      if (portalStack[0].hostSceneId !== 'shelf_portal') {
        crumbItems.push({
          label: 'קיר המדפים',
          onClick: () => {
            this.manager.transitionTo('shelf_portal');
          }
        });
      }

      portalStack.forEach((entry, idx) => {
        const hostLabel = this._getSceneShortLabelHe(entry.hostSceneId);
        crumbItems.push({
          label: hostLabel,
          onClick: () => {
            if (typeof this.manager.returnToPortalAncestor === 'function') {
              this.manager.returnToPortalAncestor(idx);
            } else {
              this.manager.transitionTo(entry.hostSceneId);
            }
          }
        });
      });

      crumbItems.push({
        label: currentLabel,
        isCurrent: true
      });
    } else if (sceneId === 'shelf_portal') {
      crumbItems.push({
        label: 'קיר המדפים',
        isCurrent: true
      });
    } else {
      crumbItems.push({
        label: 'קיר המדפים',
        onClick: () => {
          this.manager.transitionTo('shelf_portal');
        }
      });

      if (matchedGroup && matchedGroup.id !== 'portal_hub') {
        const primaryGroupScene = matchedGroup.sceneIds[0];
        const groupLabel = matchedGroup.shortTabHe || matchedGroup.titleHe || matchedGroup.id;
        if (primaryGroupScene && primaryGroupScene !== sceneId && groupLabel !== currentLabel) {
          crumbItems.push({
            label: groupLabel,
            onClick: () => {
              this.manager.transitionTo(primaryGroupScene);
            }
          });
        }
      }

      crumbItems.push({
        label: currentLabel,
        isCurrent: true
      });
    }

    this._populateBreadcrumbElements(crumbItems);
    this._fitBreadcrumbsToCompactViewport(crumbItems);
    this._scrollCurrentBreadcrumbIntoView();
    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(() => {
        this._fitBreadcrumbsToCompactViewport(crumbItems);
        this._scrollCurrentBreadcrumbIntoView();
      });
    }

    // Outgoing portals from current scene + sibling scenes in the same series (#29)
    if (!this.spatialOutgoingPortals) return;
    this.spatialOutgoingPortals.innerHTML = '';

    const outgoingShelves =
      sceneId !== 'shelf_portal' ? this.manager.getPortalShelves(sceneId) : [];
    const outgoingTargetIds = new Set();

    if (outgoingShelves.length > 0) {
      const labelEl = document.createElement('span');
      labelEl.className = 'spatial-portal-label';
      labelEl.textContent = 'שערים מכאן:';
      this.spatialOutgoingPortals.appendChild(labelEl);

      for (const spec of outgoingShelves) {
        outgoingTargetIds.add(spec.targetSceneId);
        const isLocked =
          typeof this.manager.isPortalLocked === 'function'
            ? this.manager.isPortalLocked(spec.id)
            : false;
        const shortLabel =
          spec.shortBadgeHe || PORTAL_SHORT_BADGE_LABELS_HE[spec.id] || spec.labelHe;
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = `spatial-portal-chip${isLocked ? ' locked' : ''}`;
        chip.setAttribute('data-portal-chip', spec.id);
        chip.textContent = isLocked ? `🔒 ${shortLabel}` : `${shortLabel} ↙`;
        chip.title = isLocked
          ? `לחצו להסרת החפץ המסתיר ופתיחת ${spec.labelHe}`
          : `צלילה אל ${spec.labelHe}`;
        chip.addEventListener('click', () => {
          if (
            typeof this.manager.isPortalLocked === 'function' &&
            this.manager.isPortalLocked(spec.id)
          ) {
            this.manager.erasePortalOccluder(spec.id);
          } else {
            this.manager.flyIntoPortalShelf(spec.id);
          }
        });
        this.spatialOutgoingPortals.appendChild(chip);
      }
    }

    // Show sibling sub-scenes when inside a multi-scene series via portal (or inside wandering_drawings)
    const showSiblings =
      matchedGroup &&
      matchedGroup.sceneIds.length > 1 &&
      (portalStack.length > 0 || matchedGroup.id === 'wandering_drawings');
    const siblingIds = showSiblings
      ? matchedGroup.sceneIds.filter((id) => id !== sceneId && !outgoingTargetIds.has(id))
      : [];

    if (siblingIds.length > 0) {
      const sibLabel = document.createElement('span');
      sibLabel.className = 'spatial-portal-label';
      sibLabel.textContent = 'באותה סדרה:';
      this.spatialOutgoingPortals.appendChild(sibLabel);

      for (const sibId of siblingIds) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'spatial-portal-chip sibling-chip';
        chip.setAttribute('data-sibling-scene', sibId);
        chip.textContent = this._getSceneShortLabelHe(sibId);
        chip.addEventListener('click', () => {
          this.manager.transitionTo(sibId);
        });
        this.spatialOutgoingPortals.appendChild(chip);
      }
    }

    this.spatialOutgoingPortals.classList.toggle(
      'hidden',
      this.spatialOutgoingPortals.childElementCount === 0
    );
  }

  _setSidebarOpen(isOpen) {
    if (!this.spatialSidebar) return;
    const open = Boolean(isOpen);
    if (
      !open &&
      document.activeElement &&
      this.spatialSidebar.contains(document.activeElement) &&
      typeof document.activeElement.blur === 'function'
    ) {
      document.activeElement.blur();
    }
    this.spatialSidebar.classList.toggle('collapsed', !open);
    document.body.classList.toggle('spatial-sidebar-open', open);
    if (this.sidebarToggleBtn) {
      this.sidebarToggleBtn.classList.toggle('active', open);
      this.sidebarToggleBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    }
    if (open && typeof window.__sendDiag === 'function') {
      window.__sendDiag('sidebar_opened', { scene: this.manager ? this.manager.currentSceneId : null });
    }
  }

  initUIListeners() {
    // Collapsible Left Sidebar Toggle & Close
    if (this.sidebarToggleBtn && this.spatialSidebar) {
      this.sidebarToggleBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const nextOpen = this.spatialSidebar.classList.contains('collapsed');
        if (nextOpen) {
          this._setInfoDrawerOpen(false);
          if (this.feedbackOverlay) {
            this.feedbackOverlay.closeDrawer();
          }
        }
        this._setSidebarOpen(nextOpen);
        if (typeof this.sidebarToggleBtn.blur === 'function') {
          this.sidebarToggleBtn.blur();
        }
      });
    }
    if (this.sidebarCloseBtn) {
      this.sidebarCloseBtn.addEventListener('click', () => {
        this._setSidebarOpen(false);
      });
    }

    // Spatial History Back / Forward (#29)
    if (this.spatialBackBtn) {
      this.spatialBackBtn.addEventListener('click', () => {
        this.navigateHistory(-1);
        this.spatialBackBtn.blur();
      });
    }
    if (this.spatialForwardBtn) {
      this.spatialForwardBtn.addEventListener('click', () => {
        this.navigateHistory(1);
        this.spatialForwardBtn.blur();
      });
    }

    window.addEventListener('resize', () => {
      if (this.manager && this.manager.currentSceneId) {
        this._renderSpatialBreadcrumbs(this.manager.currentSceneId);
      }
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (this.spatialSidebar && !this.spatialSidebar.classList.contains('collapsed')) {
          this._setSidebarOpen(false);
        }
        if (this.infoDrawer && this.infoDrawer.classList.contains('open')) {
          this._setInfoDrawerOpen(false);
        }
        return;
      }
      if (!e.altKey || e.metaKey || e.ctrlKey) return;
      const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        this.navigateHistory(-1);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        this.navigateHistory(1);
      }
    });

    // Exhibition Group Tabs
    this.exhibitionTabBtns.forEach((tabBtn) => {
      tabBtn.addEventListener('click', () => {
        const groupId = tabBtn.getAttribute('data-exhibition-group');
        if (groupId) {
          this._setActiveExhibitionGroup(groupId, true);
        }
        tabBtn.blur();
      });
    });

    // Prev / Next Exhibition Scene Stepper
    if (this.exhibitionPrevBtn) {
      this.exhibitionPrevBtn.addEventListener('click', () => {
        this._stepAdjacentScene(-1);
        this.exhibitionPrevBtn.blur();
      });
    }
    if (this.exhibitionNextBtn) {
      this.exhibitionNextBtn.addEventListener('click', () => {
        this._stepAdjacentScene(1);
        this.exhibitionNextBtn.blur();
      });
    }

    // Return to 3D Shelf Portal Hub Button
    if (this.portalReturnBtn) {
      this.portalReturnBtn.addEventListener('click', () => {
        if (this.manager) {
          this.manager.returnToPortalHub();
        }
        this.portalReturnBtn.blur();
      });
    }

    // Scene Selector Sub-Pills
    this.sceneBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const sceneId = btn.getAttribute('data-scene');
        if (sceneId && this.manager) {
          this.manager.transitionTo(sceneId);
        }
        btn.blur();
        if (window.innerWidth <= 900) {
          this._setSidebarOpen(false);
        }
      });
    });

    // Curatorial Info Drawer Toggle
    if (this.infoToggleBtn && this.infoDrawer) {
      this.infoToggleBtn.addEventListener('click', () => {
        const nextOpen = !this.infoDrawer.classList.contains('open');
        if (nextOpen) {
          this._setSidebarOpen(false);
          if (this.feedbackOverlay) {
            this.feedbackOverlay.closeDrawer();
          }
        }
        this._setInfoDrawerOpen(nextOpen);
        this.infoToggleBtn.blur();
      });
    }

    if (this.closeDrawerBtn && this.infoDrawer) {
      this.closeDrawerBtn.addEventListener('click', () => {
        this._setInfoDrawerOpen(false);
      });
    }

    // Fullscreen Toggle
    if (this.fullscreenBtn) {
      this.fullscreenBtn.addEventListener('click', () => {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(() => {});
          this.fullscreenBtn.textContent = 'צא ממסך מלא';
        } else {
          document.exitFullscreen().catch(() => {});
          this.fullscreenBtn.textContent = 'מסך מלא';
        }
        this.fullscreenBtn.blur();
      });
    }

    // Gyroscope / DeviceOrientation Toggle
    if (this.gyroToggleBtn) {
      this.gyroToggleBtn.addEventListener('click', () => {
        this.gyroToggleBtn.blur();
        Promise.resolve(this.manager.enableGyroscope()).then((active) => {
          if (
            !active &&
            this.manager &&
            this.manager.gyroPermissionState === 'denied' &&
            this.centerHint
          ) {
            this.centerHint.textContent =
              'יש לאשר גישה לחיישני תנועה בדפדפן כדי להפעיל מצב ג׳יירו';
            this.centerHint.classList.remove('hidden');
          }
        });
      });
    }

    // Web Audio Procedural Garden Wind & Chimes Ambient Soundscape
    if (this.audioToggleBtn) {
      this.audioToggleBtn.addEventListener('click', () => {
        this.toggleAmbientAudio();
        this.audioToggleBtn.blur();
      });
    }

    // Reset 6-DOF Camera Pose
    if (this.resetPoseBtn) {
      this.resetPoseBtn.addEventListener('click', () => {
        if (this.manager && typeof this.manager.resetCameraPose === 'function') {
          this.manager.resetCameraPose();
        }
        this.resetPoseBtn.blur();
      });
    }

    // Restore Erased Shelf Occluder Objects
    if (this.restoreOccludersBtn) {
      this.restoreOccludersBtn.addEventListener('click', () => {
        if (this.manager && typeof this.manager.restoreAllPortalOccluders === 'function') {
          this.manager.restoreAllPortalOccluders();
          this._syncRestoreOccludersBtn();
          for (const spec of this.manager.getPortalShelves()) {
            const badgeEl = this._portalBadgeEls.get(spec.id);
            if (badgeEl) {
              this._syncBadgeLabelAndLockState(badgeEl, spec);
            }
          }
          this._renderSpatialBreadcrumbs(this.manager.currentSceneId);
        }
        this.restoreOccludersBtn.blur();
      });
    }

    // Dismiss drawer or sidebar on clicking outside
    document.addEventListener('click', (e) => {
      if (this.infoDrawer && this.infoDrawer.classList.contains('open')) {
        if (
          !this.infoDrawer.contains(e.target) &&
          (!this.infoToggleBtn || !this.infoToggleBtn.contains(e.target))
        ) {
          this._setInfoDrawerOpen(false);
        }
      }
      if (this.spatialSidebar && !this.spatialSidebar.classList.contains('collapsed')) {
        if (
          !this.spatialSidebar.contains(e.target) &&
          (!this.sidebarToggleBtn || !this.sidebarToggleBtn.contains(e.target))
        ) {
          this._setSidebarOpen(false);
        }
      }
    });
  }

  _setInfoDrawerOpen(isOpen) {
    if (!this.infoDrawer) return;
    const open = Boolean(isOpen);
    if (
      !open &&
      document.activeElement &&
      this.infoDrawer.contains(document.activeElement) &&
      typeof document.activeElement.blur === 'function'
    ) {
      document.activeElement.blur();
    }
    this.infoDrawer.classList.toggle('open', open);
    document.body.classList.toggle('info-drawer-open', open);
  }

  updateActiveSceneUI(sceneId) {
    const matchedGroup = this._findGroupForScene(sceneId);
    if (matchedGroup) {
      this._setActiveExhibitionGroup(matchedGroup.id, false);
    }

    this.sceneBtns.forEach((btn) => {
      const isMatch = btn.getAttribute('data-scene') === sceneId;
      btn.classList.toggle('active', isMatch);
      btn.setAttribute('aria-pressed', isMatch ? 'true' : 'false');
      if (
        isMatch &&
        this.spatialSidebar &&
        !this.spatialSidebar.classList.contains('collapsed') &&
        typeof btn.scrollIntoView === 'function'
      ) {
        btn.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    });

    const returnInfo =
      this.manager && typeof this.manager.getPortalReturnInfo === 'function'
        ? this.manager.getPortalReturnInfo()
        : this.manager && this.manager.enteredViaPortalId && sceneId !== 'shelf_portal'
          ? {
              hostSceneId: 'shelf_portal',
              buttonLabelHe: 'חזרה לקיר המדפים',
              ariaLabelHe: 'חזרה אל קיר המדפים התלת-ממדי'
            }
          : null;
    const showPortalReturn = Boolean(returnInfo);
    if (this.portalReturnBtn) {
      this.portalReturnBtn.classList.toggle('hidden', !showPortalReturn);
      if (returnInfo) {
        this.portalReturnBtn.textContent = returnInfo.buttonLabelHe;
        this.portalReturnBtn.setAttribute('aria-label', returnInfo.ariaLabelHe);
      }
    }
    const originGroup = returnInfo ? this._findGroupForScene(returnInfo.hostSceneId) : null;
    this.exhibitionTabBtns.forEach((tabBtn) => {
      const gId = tabBtn.getAttribute('data-exhibition-group');
      tabBtn.classList.toggle(
        'portal-origin-highlight',
        Boolean(originGroup && gId === originGroup.id)
      );
    });

    if (this.centerHint) {
      if (sceneId === 'shelf_portal') {
        this.centerHint.textContent = PORTAL_CENTER_HINT_HE;
      } else if (sceneId === 'fried_yard_installation') {
        this.centerHint.textContent = YARD_PORTAL_CENTER_HINT_HE;
      } else if (sceneId === 'fried_house_railing') {
        this.centerHint.textContent = FRIED_RAILING_PORTAL_HINT_HE;
      } else if (sceneId === 'denya_living_room' || sceneId === 'denya_couch_guitar') {
        this.centerHint.textContent = DENYA_PORTAL_CENTER_HINT_HE;
      } else if (sceneId === 'wandering_drawing_1') {
        this.centerHint.textContent = WANDERING1_SCROLL_HINT_HE;
      } else if (sceneId === 'tzofen_brick_table' || sceneId === 'tzofen_brick_table_v2') {
        this.centerHint.textContent = TZOFEN_PORTAL_HINT_HE;
      } else if (sceneId === 'bene_beitcha_scroll_foot') {
        this.centerHint.textContent = SCROLL_FOOT_HINT_HE;
      } else if (sceneId === 'herzliya_lawn_carpet') {
        this.centerHint.textContent = LAWN_CARPET_HINT_HE;
      } else if (sceneId === 'bar_giora_bedroom') {
        this.centerHint.textContent = BEDROOM_VITRINE_HINT_HE;
      } else if (sceneId === 'paris_cite_white_table') {
        this.centerHint.textContent = PARIS_TABLE_HINT_HE;
      } else {
        this.centerHint.textContent = DEFAULT_CENTER_HINT_HE;
      }
      this.centerHint.classList.remove('hidden');
      if (this._centerHintTimer) clearTimeout(this._centerHintTimer);
      this._centerHintTimer = setTimeout(() => {
        if (this.centerHint) {
          this.centerHint.classList.add('hidden');
        }
      }, 6500);
    }

    if (!this._isScenePortalHost(sceneId)) {
      this.updatePortalHoverPlaque(null, 0, 0);
    } else {
      this._buildPortalMicroBadges();
    }
    this._syncRestoreOccludersBtn();
    this._recordNavigationHistory(sceneId);
    this._renderSpatialBreadcrumbs(sceneId);

    // Update curatorial drawer highlights across all scene story cards
    document.querySelectorAll('.story-card').forEach((card) => {
      const isActiveStory = card.id === `story-${sceneId}`;
      card.classList.toggle('active-story', isActiveStory);
    });
  }

  toggleAmbientAudio() {
    if (!this.audioContext) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      this.audioContext = new AudioCtx();
      this.initProceduralSound();
    }

    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }

    if (this.isAudioPlaying) {
      this.masterGain.gain.linearRampToValueAtTime(0.001, this.audioContext.currentTime + 0.8);
      this.isAudioPlaying = false;
      this.audioToggleBtn.classList.remove('active');
      this.audioToggleBtn.textContent = '♫ סאונד: מושתק';
    } else {
      this.masterGain.gain.linearRampToValueAtTime(0.12, this.audioContext.currentTime + 0.8);
      this.isAudioPlaying = true;
      this.audioToggleBtn.classList.add('active');
      this.audioToggleBtn.textContent = '♫ סאונד: גן ורוח (פעיל)';
    }
  }

  initProceduralSound() {
    // Generates a soft, atmospheric Mediterranean breeze & warm resonant chord
    // using pure Web Audio API (zero audio file dependencies, 100% offline longevity)
    this.masterGain = this.audioContext.createGain();
    this.masterGain.gain.value = 0.001;
    this.masterGain.connect(this.audioContext.destination);

    // Filtered pink noise simulating breeze through citrus trees and pine wadi
    const bufferSize = this.audioContext.sampleRate * 2;
    const noiseBuffer = this.audioContext.createBuffer(1, bufferSize, this.audioContext.sampleRate);
    const output = noiseBuffer.getChannelData(0);
    let b0 = 0;
    let b1 = 0;
    let b2 = 0;
    for (let i = 0; i < bufferSize; i += 1) {
      const white = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + white * 0.0555179;
      b1 = 0.99332 * b1 + white * 0.0750759;
      b2 = 0.96900 * b2 + white * 0.1538520;
      output[i] = (b0 + b1 + b2) * 0.11;
    }

    const whiteNoise = this.audioContext.createBufferSource();
    whiteNoise.buffer = noiseBuffer;
    whiteNoise.loop = true;

    const noiseFilter = this.audioContext.createBiquadFilter();
    noiseFilter.type = 'lowpass';
    noiseFilter.frequency.value = 420;

    whiteNoise.connect(noiseFilter);
    noiseFilter.connect(this.masterGain);
    whiteNoise.start();

    // Subtle warm resonant harmonics (Ahuva & Michael's garden harmony)
    const freqs = [146.83, 220.0, 329.63]; // D3, A3, E4
    freqs.forEach((f) => {
      const osc = this.audioContext.createOscillator();
      osc.type = 'sine';
      osc.frequency.value = f;

      const oscGain = this.audioContext.createGain();
      oscGain.gain.value = 0.04;

      osc.connect(oscGain);
      oscGain.connect(this.masterGain);
      osc.start();
    });
  }
}

// Bootstrap on DOMContentLoaded (safe when imported in Node unit tests or after readyState interactive)
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', () => {
      new SpatialApp();
    });
  } else {
    new SpatialApp();
  }
}
