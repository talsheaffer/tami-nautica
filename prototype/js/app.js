// prototype/js/app.js
// Main application entry point for Tamar "Tami" Sheaffer 70th Birthday 2.5D Spatial Prototype
// Manages Exhibition Group navigation, 3D Shelf Portal micro-badges & hover plaques,
// curatorial drawer state, and procedural Web Audio ambient soundscape.

import { SpatialSceneManager, EXHIBITION_GROUPS } from './scene_manager.js';
import { BetaFeedbackOverlay } from './feedback_overlay.js';

export { EXHIBITION_GROUPS };

const EXHIBITION_SCENE_SEQUENCE = Object.freeze(
  EXHIBITION_GROUPS.flatMap((group) => group.sceneIds)
);

const PORTAL_SHORT_BADGE_LABELS_HE = Object.freeze({
  cubby_capsules: '🫧 קפסולות נייר',
  cubby_plan_b_v3: '🕳️ תמי ויערה',
  cubby_wandering_1: '✏️ שיטוט 1',
  cubby_bene_beitcha: '🏛️ בנה ביתך',
  cubby_plan_b_pipe: '🕳️ צינור Plan B',
  cubby_wandering_2: '✏️ שיטוט 2',
  cubby_denya_model: '🏠 דגם דניה',
  cubby_marseille: '🪞 מרסיי',
  cubby_fried_house: '🪨 בית פריד',
  yard_railing_portal: '🏡 תמי ואראלה במעקה',
  fried_house_window_wandering_1: '✏️ שיטוט 1 בחלון',
  portal_denya_dining_wandering_2: '✏️ שיטוט 2',
  portal_denya_couch_wandering_2: '✏️ שיטוט 2',
  portal_denya_to_couch: '🎸 לספה והגיטרה',
  portal_denya_back_to_dining: '🍽️ לשולחן האוכל',
  portal_tzofen_tree_window: '🌳 חלון אל העץ',
  portal_wandering1_scroll_world: '🌴 לתוך המגילה'
});

const SCENE_BREADCRUMB_LABELS_HE = Object.freeze({
  shelf_portal: '⊞ קיר המדפים',
  paper_capsules: '🫧 קפסולות נייר',
  tzofen_brick_table: '🧱 צופן',
  plan_b_pipe_v3: '🕳️ תמי ויערה',
  plan_b_pipe: '🕳️ צינור Plan B',
  bene_beitcha: '🏛️ בנה ביתך',
  humpty_dumpty_bundle: '🥚 האמפטי דאמפטי',
  denya_model: '🏠 דגם דניה',
  denya_living_room: '🍽️ סלון דניה',
  denya_couch_guitar: '🎸 ספת דניה',
  wandering_drawing_1: '✏️ שיטוט 1',
  scroll_palm_world: '🌴 עולם המגילה',
  wandering_drawing: '✏️ שיטוט 2',
  fried_yard_installation: '🪨 מיצב בחצר פריד',
  fried_house_railing: '🏡 בית פריד במעקה',
  fried_table_hybrid: '🪑 שולחן בבית פריד',
  herzliya_tree_canopy: '🌳 צמרת העץ',
  marseille_mirror: '🪞 מרסיי',
  plan_b_pipe_v1: '🕳️ צינור Plan B v1',
  marseille_mirror_v1: '🪞 מרסיי v1',
  herzliya_studio: '🌿 גינת הרצליה v1',
  carmel_wadi: '🌲 ואדי הכרמל v1'
});

const DEFAULT_CENTER_HINT_HE =
  'גרור לסיבוב מבט · גלגל / WASD להליכה · Q/E או PgUp/PgDn לעלייה וירידה בגובה';
const PORTAL_CENTER_HINT_HE =
  'לחצו על חפצים להסרתם ועל תא פתוח לצלילה פנימה · WASD להליכה · Q/E או PgUp/PgDn לגובה';
const YARD_PORTAL_CENTER_HINT_HE =
  'לחצו על המיצב לפני המעקה להסרתו ולחשיפת תמי ואראלה · לחצו עליהן למעבר לסצנת המעקה';
const FRIED_RAILING_PORTAL_HINT_HE =
  'לחצו על תריס העץ החום בשמאל הקיר להסרתו ולחשיפת שער אל ׳שיטוט 1׳ · WASD להליכה';
const DENYA_PORTAL_CENTER_HINT_HE =
  'לחצו על הספה או על מסגרת ׳שיטוט 2׳ למעבר בין המרחבים · WASD להליכה · Q/E או PgUp/PgDn לגובה';
const WANDERING1_SCROLL_HINT_HE =
  'ריחפו או לחצו על המגילה שמימין כדי לצלול ברציפות לתוך עולם המגילה · WASD להליכה';
const TZOFEN_PORTAL_HINT_HE =
  'לחצו על מסגרת החלון הלבנה בראש השולחן לצלילה אל צמרת העץ בהרצליה · WASD להליכה';

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
    this.topBarEl = document.querySelector('.top-bar');
    this.spatialNavBar = document.getElementById('spatial-nav-bar');
    this.spatialBackBtn = document.getElementById('spatial-back-btn');
    this.spatialForwardBtn = document.getElementById('spatial-forward-btn');
    this.spatialBreadcrumbList = document.getElementById('spatial-breadcrumb-list');
    this.spatialOutgoingPortals = document.getElementById('spatial-outgoing-portals');
    this.bottomDockEl = document.querySelector('.bottom-dock');
    this.centerHint = document.getElementById('center-hint');
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
    this._activeGroupId = 'binyamin_plan_b';
    this._navHistory = [];
    this._navHistoryIndex = -1;
    this._suppressHistoryPush = false;
    this.audioContext = null;
    this.isAudioPlaying = false;

    this.initSceneManager();
    this._buildPortalMicroBadges();
    this.updateActiveSceneUI(this.manager.currentSceneId);
    this.feedbackOverlay = new BetaFeedbackOverlay(this.manager);
    this.initUIListeners();
  }

  initSceneManager() {
    this.manager = new SpatialSceneManager(this.canvas, {
      onReady: () => {
        if (this.loadingOverlay) {
          this.loadingOverlay.classList.add('hidden');
        }
      },
      onSceneChange: (sceneId) => {
        this.updateActiveSceneUI(sceneId);
      },
      onPortalHover: (spec, clientX, clientY) => {
        this.updatePortalHoverPlaque(spec, clientX, clientY);
      },
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
    const baseLabel =
      spec.shortBadgeHe || PORTAL_SHORT_BADGE_LABELS_HE[spec.id] || spec.labelHe;
    const isLocked =
      typeof this.manager.isPortalLocked === 'function'
        ? this.manager.isPortalLocked(spec.id)
        : false;
    const isCompactPin = this.manager.currentSceneId === 'shelf_portal';
    badgeBtn.classList.toggle('compact-pin', isCompactPin);
    if (isCompactPin) {
      const pinState = `${isLocked ? 'locked' : 'open'}:${baseLabel}`;
      if (badgeBtn.dataset.pinState !== pinState) {
        badgeBtn.dataset.pinState = pinState;
        badgeBtn.innerHTML = '';
        const iconSpan = document.createElement('span');
        iconSpan.className = 'badge-pin-icon';
        iconSpan.setAttribute('aria-hidden', 'true');
        iconSpan.textContent = isLocked ? '🔒' : '↗';
        const labelSpan = document.createElement('span');
        labelSpan.className = 'badge-pin-label';
        labelSpan.textContent = baseLabel;
        badgeBtn.appendChild(iconSpan);
        badgeBtn.appendChild(labelSpan);
      }
    } else {
      delete badgeBtn.dataset.pinState;
      const nextText = isLocked ? `🔒 ${baseLabel}` : baseLabel;
      if (badgeBtn.textContent !== nextText) {
        badgeBtn.textContent = nextText;
      }
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
      const rect = badgeEl.getBoundingClientRect();
      if (rect.width > 0 && !this.portalHoverPlaque.classList.contains('hidden')) {
        this.updatePortalHoverPlaque(spec, rect.left + rect.width * 0.5, rect.top);
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
      let label = '↺ השב חפצים למדפים';
      if (sceneId === 'fried_yard_installation') {
        label = '↺ השב את המיצב לפני המעקה';
      } else if (sceneId === 'fried_house_railing') {
        label = '↺ סגור את תריס העץ בחלון';
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

      badgeBtn.addEventListener('mouseenter', () => {
        const rect = badgeBtn.getBoundingClientRect();
        this.updatePortalHoverPlaque(spec, rect.left + rect.width * 0.5, rect.top);
      });

      badgeBtn.addEventListener('mouseleave', () => {
        this.updatePortalHoverPlaque(null, 0, 0);
      });

      badgeBtn.addEventListener('focus', () => {
        const rect = badgeBtn.getBoundingClientRect();
        this.updatePortalHoverPlaque(spec, rect.left + rect.width * 0.5, rect.top);
      });

      badgeBtn.addEventListener('blur', () => {
        this.updatePortalHoverPlaque(null, 0, 0);
      });

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
    let safeMinY = 56;
    let safeMaxY = vh - 108;
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
        proj.clientX >= 24 &&
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

  updatePortalHoverPlaque(spec, clientX, clientY) {
    if (!this.portalHoverPlaque) return;
    if (
      !spec ||
      !this.manager ||
      !this._isScenePortalHost(this.manager.currentSceneId) ||
      this.manager.isTransitioning ||
      this.manager._portalFlight ||
      this.manager.interactionCaptureActive
    ) {
      this.portalHoverPlaque.classList.add('hidden');
      return;
    }

    const vw = window.innerWidth || 1440;
    const vh = window.innerHeight || 900;
    const clampedX = Math.min(vw - 170, Math.max(170, clientX));
    const clampedY = Math.min(vh - 130, Math.max(92, clientY));
    const isLocked =
      typeof this.manager.isPortalLocked === 'function'
        ? this.manager.isPortalLocked(spec.id)
        : false;
    const occLabel = (spec.occluder && spec.occluder.labelHe) || 'חפץ במדף';
    let ctaHtml = '<span class="plaque-cta">לחצו לצלילה לתוך המרחב ↙</span>';
    if (isLocked) {
      ctaHtml = `<span class="plaque-cta">🔒 חסום ע״י ${escapeHtmlText(occLabel)} · לחצו להסרה ופתיחת המרחב ✨</span>`;
    } else if (spec.id === 'yard_railing_portal') {
      ctaHtml = '<span class="plaque-cta">לחצו למעבר אל תמי ואראלה ליד המעקה ↙</span>';
    } else if (spec.portalMode === 'reverse_transition') {
      ctaHtml = '<span class="plaque-cta">לחצו למעבר אל המבט משולחן האוכל ↩</span>';
    }

    this.portalHoverPlaque.innerHTML =
      `<span class="plaque-title">${escapeHtmlText(spec.labelHe)}</span>` + ctaHtml;
    this.portalHoverPlaque.style.left = `${clampedX.toFixed(1)}px`;
    this.portalHoverPlaque.style.top = `${clampedY.toFixed(1)}px`;
    if (this.centerHint) {
      this.centerHint.classList.add('hidden');
    }
    this.portalHoverPlaque.classList.remove('hidden');
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

    if (navigateIfOutside && this.manager && !group.sceneIds.includes(this.manager.currentSceneId)) {
      const primarySceneId = group.sceneIds[0];
      if (primarySceneId) {
        this.manager.transitionTo(primarySceneId);
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

  _appendBreadcrumbItem(listEl, { label, onClick, isCurrent = false, addSeparator = false }) {
    if (addSeparator) {
      const sepLi = document.createElement('li');
      sepLi.className = 'spatial-crumb-item';
      sepLi.setAttribute('aria-hidden', 'true');
      const sepSpan = document.createElement('span');
      sepSpan.className = 'spatial-crumb-sep';
      sepSpan.textContent = '‹';
      sepLi.appendChild(sepSpan);
      listEl.appendChild(sepLi);
    }

    const li = document.createElement('li');
    li.className = 'spatial-crumb-item';
    if (isCurrent) {
      const curSpan = document.createElement('span');
      curSpan.className = 'spatial-crumb-current';
      curSpan.setAttribute('aria-current', 'page');
      curSpan.textContent = label;
      li.appendChild(curSpan);
    } else {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'spatial-crumb-btn';
      btn.textContent = label;
      btn.addEventListener('click', onClick);
      li.appendChild(btn);
    }
    listEl.appendChild(li);
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
    this.spatialBreadcrumbList.innerHTML = '';

    const portalStack = this._getCurrentPortalStackSnapshot();
    const matchedGroup = this._findGroupForScene(sceneId);
    let hasItems = false;

    if (portalStack.length > 0) {
      if (portalStack[0].hostSceneId !== 'shelf_portal') {
        this._appendBreadcrumbItem(this.spatialBreadcrumbList, {
          label: '⊞ קיר המדפים',
          addSeparator: false,
          onClick: () => {
            this.manager.transitionTo('shelf_portal');
          }
        });
        hasItems = true;
      }

      portalStack.forEach((entry, idx) => {
        const hostLabel = this._getSceneShortLabelHe(entry.hostSceneId);
        this._appendBreadcrumbItem(this.spatialBreadcrumbList, {
          label: hostLabel,
          addSeparator: hasItems,
          onClick: () => {
            if (typeof this.manager.returnToPortalAncestor === 'function') {
              this.manager.returnToPortalAncestor(idx);
            } else {
              this.manager.transitionTo(entry.hostSceneId);
            }
          }
        });
        hasItems = true;
      });

      this._appendBreadcrumbItem(this.spatialBreadcrumbList, {
        label: this._getSceneShortLabelHe(sceneId),
        addSeparator: hasItems,
        isCurrent: true
      });
    } else if (sceneId === 'shelf_portal') {
      this._appendBreadcrumbItem(this.spatialBreadcrumbList, {
        label: '⊞ קיר המדפים · שער התערוכות',
        addSeparator: false,
        isCurrent: true
      });
    } else {
      this._appendBreadcrumbItem(this.spatialBreadcrumbList, {
        label: '⊞ קיר המדפים',
        addSeparator: false,
        onClick: () => {
          this.manager.transitionTo('shelf_portal');
        }
      });
      hasItems = true;

      if (matchedGroup && matchedGroup.id !== 'portal_hub') {
        const primaryGroupScene = matchedGroup.sceneIds[0];
        this._appendBreadcrumbItem(this.spatialBreadcrumbList, {
          label: matchedGroup.shortTabHe || matchedGroup.titleHe || matchedGroup.id,
          addSeparator: true,
          onClick: () => {
            if (primaryGroupScene && primaryGroupScene !== sceneId) {
              this.manager.transitionTo(primaryGroupScene);
            } else {
              this._setActiveExhibitionGroup(matchedGroup.id, false);
            }
          }
        });
      }

      this._appendBreadcrumbItem(this.spatialBreadcrumbList, {
        label: this._getSceneShortLabelHe(sceneId),
        addSeparator: true,
        isCurrent: true
      });
    }

    this._scrollCurrentBreadcrumbIntoView();
    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(() => this._scrollCurrentBreadcrumbIntoView());
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

  initUIListeners() {
    // Spatial History Back / Forward (#29)
    if (this.spatialBackBtn) {
      this.spatialBackBtn.addEventListener('click', () => {
        this.navigateHistory(-1);
      });
    }
    if (this.spatialForwardBtn) {
      this.spatialForwardBtn.addEventListener('click', () => {
        this.navigateHistory(1);
      });
    }

    window.addEventListener('keydown', (e) => {
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
      });
    });

    // Prev / Next Exhibition Scene Stepper
    if (this.exhibitionPrevBtn) {
      this.exhibitionPrevBtn.addEventListener('click', () => {
        this._stepAdjacentScene(-1);
      });
    }
    if (this.exhibitionNextBtn) {
      this.exhibitionNextBtn.addEventListener('click', () => {
        this._stepAdjacentScene(1);
      });
    }

    // Return to 3D Shelf Portal Hub Button
    if (this.portalReturnBtn) {
      this.portalReturnBtn.addEventListener('click', () => {
        if (this.manager) {
          this.manager.returnToPortalHub();
        }
      });
    }

    // Scene Selector Sub-Pills
    this.sceneBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const sceneId = btn.getAttribute('data-scene');
        if (sceneId && this.manager) {
          this.manager.transitionTo(sceneId);
        }
      });
    });

    // Curatorial Info Drawer Toggle
    if (this.infoToggleBtn && this.infoDrawer) {
      this.infoToggleBtn.addEventListener('click', () => {
        const nextOpen = !this.infoDrawer.classList.contains('open');
        if (nextOpen && this.feedbackOverlay) {
          this.feedbackOverlay.closeDrawer();
        }
        this._setInfoDrawerOpen(nextOpen);
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
          this.fullscreenBtn.textContent = '⛶ צא ממסך מלא';
        } else {
          document.exitFullscreen().catch(() => {});
          this.fullscreenBtn.textContent = '⛶ מסך מלא';
        }
      });
    }

    // Gyroscope / DeviceOrientation Toggle
    if (this.gyroToggleBtn) {
      this.gyroToggleBtn.addEventListener('click', () => {
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
      });
    }

    // Reset 6-DOF Camera Pose
    if (this.resetPoseBtn) {
      this.resetPoseBtn.addEventListener('click', () => {
        if (this.manager && typeof this.manager.resetCameraPose === 'function') {
          this.manager.resetCameraPose();
        }
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
      });
    }

    // Dismiss drawer on clicking outside
    document.addEventListener('click', (e) => {
      if (this.infoDrawer && this.infoDrawer.classList.contains('open')) {
        if (
          !this.infoDrawer.contains(e.target) &&
          (!this.infoToggleBtn || !this.infoToggleBtn.contains(e.target))
        ) {
          this._setInfoDrawerOpen(false);
        }
      }
    });
  }

  _setInfoDrawerOpen(isOpen) {
    if (!this.infoDrawer) return;
    this.infoDrawer.classList.toggle('open', Boolean(isOpen));
    document.body.classList.toggle('info-drawer-open', Boolean(isOpen));
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
      if (isMatch && typeof btn.scrollIntoView === 'function') {
        btn.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
      }
    });

    const returnInfo =
      this.manager && typeof this.manager.getPortalReturnInfo === 'function'
        ? this.manager.getPortalReturnInfo()
        : this.manager && this.manager.enteredViaPortalId && sceneId !== 'shelf_portal'
          ? {
              hostSceneId: 'shelf_portal',
              buttonLabelHe: '⊞ חזרה לקיר המדפים',
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
      } else if (sceneId === 'tzofen_brick_table') {
        this.centerHint.textContent = TZOFEN_PORTAL_HINT_HE;
      } else {
        this.centerHint.textContent = DEFAULT_CENTER_HINT_HE;
      }
    }

    if (!this._isScenePortalHost(sceneId)) {
      this.updatePortalHoverPlaque(null, 0, 0);
    } else {
      this._buildPortalMicroBadges();
    }
    this._syncRestoreOccludersBtn();
    this._recordNavigationHistory(sceneId);
    this._renderSpatialBreadcrumbs(sceneId);

    // Update curatorial drawer highlights across all 20 scene story cards
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

// Bootstrap on DOMContentLoaded
window.addEventListener('DOMContentLoaded', () => {
  new SpatialApp();
});
