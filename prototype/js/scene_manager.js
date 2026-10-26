// prototype/js/scene_manager.js
// Three.js 2.5D Layered Depth Image (LDI) + Outpainted 6-DOF Spatial Gallery Renderer
// Honoring Tamar "Tami" Sheaffer's 70th Birthday
// Enforces homographic inner/expanded layer registration, Depth Anything V2 3D room relief,
// decoupled 6-DOF camera navigation (yaw/pitch + forward/backward dolly + X/Y strafe),
// and analytical ray-frustum clamping so zero image borders are ever exposed.

import * as THREE from './vendor/three.module.js';

const SCENE_GEOMETRY_SPEC = Object.freeze({
  baseCameraZ: 400,
  minCameraZ: 120,
  maxCameraZ: 685,
  bgZ: -350,
  midZ: -150,
  fgZ: 0,
  reflectionZ: -340,
  bgBaseWidth: 750,
  bgBaseHeight: 500,
  bgExpansionFactor: 1.80,
  bgSegmentsX: 160,
  bgSegmentsY: 108,
  midBaseWidth: 600,
  midBaseHeight: 400,
  fgBaseWidth: 500,
  fgBaseHeight: 333.3333,
  roomDepthReliefAmplitude: 92.0,
  roomPinMarginUv: 0.045,
  minInnerScale: 1.875,
  frustumInnerCoverageFactor: 0.85,
  frustumEdgeSafetyRatio: 0.985,
  desktopFovDeg: 45,
  portraitFovDeg: 56,
  maxStrafeX: 250,
  maxStrafeY: 120,
  maxYawDeg: 34,
  maxPitchDeg: 24,
  touchYawSpanDeg: 36,
  touchPitchSpanDeg: 26,
  mouseYawSpanDeg: 40,
  mousePitchSpanDeg: 28,
  pinchDollyScale: 1.45,
  touchStrafeScale: 0.65,
  mouseStrafeScale: 0.55,
  wheelDollyScale: 0.42,
  trackpadPinchDollyScale: 2.2,
  keyMoveSpeedPerSec: 270,
  hoverAmbientShiftX: 14,
  hoverAmbientShiftY: 8,
  hoverAmbientYawDeg: 2.2,
  hoverAmbientPitchDeg: 1.3,
  pointerExcursionShiftX: 65,
  pointerExcursionShiftY: 34,
  pointerExcursionYawDeg: 9.0,
  pointerExcursionPitchDeg: 5.0,
  gyroTiltSensitivityDeg: 18,
  gyroYawSensitivityDeg: 22,
  gyroMaxYawDeg: 22,
  gyroMaxPitchDeg: 16
});

const ROOM_DEPTH_VERTEX_SHADER = `
uniform sampler2D uDepthMap;
uniform float uDepthAmplitude;
uniform float uDistBg;
uniform float uPinMarginUv;

varying vec2 vUv;

void main() {
    vUv = uv;
    float rawDepth = texture2D(uDepthMap, uv).r;

    // Smoothly pin outer UV perimeter of the 2700x1800 outpainted room to Z = bgZ
    // so analytical frustum ray intersections on Z = bgZ remain exact.
    float pinX = smoothstep(0.0, uPinMarginUv, uv.x) * smoothstep(0.0, uPinMarginUv, 1.0 - uv.x);
    float pinY = smoothstep(0.0, uPinMarginUv, uv.y) * smoothstep(0.0, uPinMarginUv, 1.0 - uv.y);
    float zDisp = rawDepth * uDepthAmplitude * pinX * pinY;

    // Ray-projective compensation: zero rest-pose distortion at optical center (0, 0, baseCameraZ),
    // revealing authentic 3D room perspective as soon as camera pose translates, dollies, or rotates.
    float rayScale = (uDistBg - zDisp) / uDistBg;
    vec3 displaced = vec3(position.x * rayScale, position.y * rayScale, zDisp);

    gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}
`;

const ROOM_DEPTH_FRAGMENT_SHADER = `
precision highp float;

uniform sampler2D uColorMap;
varying vec2 vUv;

void main() {
    gl_FragColor = texture2D(uColorMap, vUv);
}
`;

const TRANSITION_VERTEX_SHADER = `
varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}
`;

const TRANSITION_FRAGMENT_SHADER = `
precision highp float;

uniform sampler2D uTexCurrent;
uniform sampler2D uTexNext;
uniform float uProgress;
uniform float uAspect;
uniform vec2 uPointer;

varying vec2 vUv;

#define PI 3.14159265359

void main() {
    vec2 uv = vUv;
    vec2 centeredUv = (uv - 0.5) * vec2(uAspect, 1.0);
    float dist = length(centeredUv);
    float envelope = sin(uProgress * PI);

    // Wave 1: Concentric ripple expanding outward
    float ripple = sin(dist * 18.0 - uProgress * 12.0) * 0.04 * envelope;

    // Wave 2: Directional warp echoing curved metal sheets in Tami's "בנה ביתך"
    float directionalWarp = cos(uv.y * 10.0 + uv.x * 6.0 + uPointer.x * 2.0) * 0.03 * envelope;

    // Combined displacement vector
    vec2 displacement = normalize(centeredUv + vec2(0.001)) * ripple + vec2(directionalWarp, -directionalWarp);

    vec2 uvCurrent = uv + displacement * (1.0 - uProgress);
    vec2 uvNext    = uv - displacement * uProgress;

    vec4 colCurrent = texture2D(uTexCurrent, uvCurrent);
    vec4 colNext    = texture2D(uTexNext, uvNext);

    // Specular silver reflection flash with M4 Reviewer 2 Finding 2 fix:
    // max(0.0, 1.0 - dist) avoids negative specular color in ultrawide corners
    float specularFlash = smoothstep(0.4, 0.5, uProgress) * (1.0 - smoothstep(0.5, 0.6, uProgress));
    vec3 flashColor = vec3(0.92, 0.94, 0.98) * specularFlash * 0.35 * max(0.0, 1.0 - dist);

    float blendFactor = smoothstep(0.3, 0.7, uProgress);
    vec4 finalColor = mix(colCurrent, colNext, blendFactor) + vec4(flashColor, 0.0);

    gl_FragColor = finalColor;
}
`;

function createSoftRadialDustTexture() {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const half = size / 2;
  const grad = ctx.createRadialGradient(half, half, 0, half, half, half);
  grad.addColorStop(0.0, 'rgba(255, 255, 255, 0.95)');
  grad.addColorStop(0.35, 'rgba(255, 250, 240, 0.45)');
  grad.addColorStop(0.75, 'rgba(255, 250, 240, 0.08)');
  grad.addColorStop(1.0, 'rgba(255, 250, 240, 0.0)');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

/**
 * Tests whether all 4 viewport corner rays from (posX, posY, posZ) rotated by
 * Euler(pitchRad, yawRad, 0, 'YXZ') intersect Z = bgZ inside [-maxHalfW..maxHalfW] x [-maxHalfH..maxHalfH].
 */
function doFrustumCornersFitOnBackground(
  posX,
  posY,
  posZ,
  yawRad,
  pitchRad,
  tanHalfFov,
  aspect,
  bgZ,
  maxHalfW,
  maxHalfH
) {
  const cosY = Math.cos(yawRad);
  const sinY = Math.sin(yawRad);
  const cosP = Math.cos(pitchRad);
  const sinP = Math.sin(pitchRad);
  const distZ = posZ - bgZ;
  if (distZ <= 1.0) return false;

  const tanX = tanHalfFov * aspect;
  const tanY = tanHalfFov;

  for (let sx = -1; sx <= 1; sx += 2) {
    for (let sy = -1; sy <= 1; sy += 2) {
      const vx = sx * tanX;
      const vy = sy * tanY;
      // Local ray is (vx, vy, -1); apply R_x(pitchRad) then R_y(yawRad)
      const y1 = vy * cosP + sinP;
      const z1 = vy * sinP - cosP;
      const rx = vx * cosY + z1 * sinY;
      const ry = y1;
      const rz = -vx * sinY + z1 * cosY;

      if (rz >= -0.15) {
        return false;
      }
      const t = distZ / (-rz);
      const ix = posX + t * rx;
      const iy = posY + t * ry;
      if (Math.abs(ix) > maxHalfW || Math.abs(iy) > maxHalfH) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Clamps a 6-DOF camera pose (x, y, z, yawDeg, pitchDeg) so that:
 * 1. posZ stays within [minCameraZ, effectiveMaxCameraZ] where centered view fits inside layer0.
 * 2. Off-axis translation & rotation scale smoothly via binary search whenever any viewport
 *    corner ray would otherwise cross the expanded 2700x1800 room boundary.
 */
function clampPoseToRoomFrustum(rawPose, fovDeg, aspect, layer0Scale, spec) {
  const halfFovRad = THREE.MathUtils.degToRad(fovDeg * 0.5);
  const tanHalfFov = Math.tan(halfFovRad);
  const maxHalfW = 0.5 * spec.bgBaseWidth * layer0Scale * spec.frustumEdgeSafetyRatio;
  const maxHalfH = 0.5 * spec.bgBaseHeight * layer0Scale * spec.frustumEdgeSafetyRatio;

  const maxCenteredDistW = (maxHalfW * 0.94) / Math.max(0.05, tanHalfFov * aspect);
  const maxCenteredDistH = (maxHalfH * 0.94) / Math.max(0.05, tanHalfFov);
  const effectiveMaxZ = Math.max(
    spec.minCameraZ,
    Math.min(spec.maxCameraZ, spec.bgZ + Math.min(maxCenteredDistW, maxCenteredDistH))
  );

  const clampedZ = THREE.MathUtils.clamp(rawPose.z, spec.minCameraZ, effectiveMaxZ);
  const candX = THREE.MathUtils.clamp(rawPose.x, -spec.maxStrafeX, spec.maxStrafeX);
  const candY = THREE.MathUtils.clamp(rawPose.y, -spec.maxStrafeY, spec.maxStrafeY);
  const candYawDeg = THREE.MathUtils.clamp(rawPose.yawDeg, -spec.maxYawDeg, spec.maxYawDeg);
  const candPitchDeg = THREE.MathUtils.clamp(rawPose.pitchDeg, -spec.maxPitchDeg, spec.maxPitchDeg);

  const candYawRad = THREE.MathUtils.degToRad(candYawDeg);
  const candPitchRad = THREE.MathUtils.degToRad(candPitchDeg);

  if (
    doFrustumCornersFitOnBackground(
      candX,
      candY,
      clampedZ,
      candYawRad,
      candPitchRad,
      tanHalfFov,
      aspect,
      spec.bgZ,
      maxHalfW,
      maxHalfH
    )
  ) {
    return {
      x: candX,
      y: candY,
      z: clampedZ,
      yawDeg: candYawDeg,
      pitchDeg: candPitchDeg
    };
  }

  let lo = 0.0;
  let hi = 1.0;
  for (let iter = 0; iter < 8; iter += 1) {
    const mid = 0.5 * (lo + hi);
    if (
      doFrustumCornersFitOnBackground(
        candX * mid,
        candY * mid,
        clampedZ,
        candYawRad * mid,
        candPitchRad * mid,
        tanHalfFov,
        aspect,
        spec.bgZ,
        maxHalfW,
        maxHalfH
      )
    ) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  return {
    x: candX * lo,
    y: candY * lo,
    z: clampedZ,
    yawDeg: candYawDeg * lo,
    pitchDeg: candPitchDeg * lo
  };
}

export class SpatialSceneManager {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.onSceneChange = options.onSceneChange || (() => {});
    this.onReady = options.onReady || (() => {});

    const assetVer = 'v=neural4';
    this.sceneConfigs = {
      herzliya: {
        id: 'herzliya',
        title: "חצר הילדות בהרצליה ובית המלאכה",
        bgTexture: `assets/scenes/scene1_herzliya_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene1_herzliya_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene1_herzliya_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene1_herzliya_fg.png?${assetVer}`,
        particleColor: 0xEAD8BE,
        particleCount: 85,
        hasReflection: false
      },
      denya: {
        id: 'denya',
        title: "הסטודיו בדניה והמבנה המעוות — 'בנה ביתך'",
        bgTexture: `assets/scenes/scene2_denya_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene2_denya_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene2_denya_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene2_denya_fg.png?${assetVer}`,
        reflectionTexture: `assets/scenes/scene2_reflections.png?${assetVer}`,
        particleColor: 0xB8C7CF,
        particleCount: 90,
        hasReflection: true
      }
    };

    this.currentSceneId = 'herzliya';
    this.targetSceneId = null;

    // Additive normalized pointer excursion (kept for subtle hover & automated test harness)
    this.pointer = { x: 0, y: 0, targetX: 0, targetY: 0 };

    // Decoupled 6-DOF camera pose state (world X, Y, Z + Euler YXZ yawDeg, pitchDeg)
    this.baseCameraZ = SCENE_GEOMETRY_SPEC.baseCameraZ;
    this.poseTarget = {
      x: 0,
      y: 0,
      z: this.baseCameraZ,
      yawDeg: 0,
      pitchDeg: 0
    };
    this.poseCurrent = {
      x: 0,
      y: 0,
      z: this.baseCameraZ,
      yawDeg: 0,
      pitchDeg: 0
    };

    this._hoverNorm = { x: 0, y: 0, targetX: 0, targetY: 0 };
    this._innerScale = SCENE_GEOMETRY_SPEC.minInnerScale;
    this._layer0Scale = SCENE_GEOMETRY_SPEC.minInnerScale * SCENE_GEOMETRY_SPEC.bgExpansionFactor;

    this.isTransitioning = false;
    this.transitionStartTime = 0;
    this.transitionDuration = 1600;
    this.transitionProgress = 0.0;
    this.isPaused = false;
    this.gyroActive = false;
    this.animId = null;
    this.layersReady = false;
    this._lastFrameTimeMs = performance.now();

    this._loadedTextures = 0;
    this._totalTextures = 9; // 2x (bg, depth, mid, fg) + 1x reflection
    this._dustSpriteTexture = null;

    // Touch & mouse interaction state
    this._isTouching = false;
    this._touchMode = 'none'; // 'look' | 'pinch_strafe'
    this._touchStartX = 0;
    this._touchStartY = 0;
    this._touchBaseYawDeg = 0;
    this._touchBasePitchDeg = 0;
    this._touchMaxTravelPx = 0;
    this._touchStartTimeMs = 0;
    this._lastSingleTapMs = 0;
    this._prevPinchDist = 0;
    this._prevPinchMidX = 0;
    this._prevPinchMidY = 0;

    this._isMouseDragging = false;
    this._mouseDragMode = 'look'; // 'look' | 'strafe'
    this._mouseDownX = 0;
    this._mouseDownY = 0;
    this._mouseBaseYawDeg = 0;
    this._mouseBasePitchDeg = 0;
    this._mouseBasePosX = 0;
    this._mouseBasePosY = 0;

    this._keysHeld = new Set();

    // Gyroscope relative state
    this._gyroOrigin = null;
    this._gyroYawDeg = 0;
    this._gyroPitchDeg = 0;
    this._gyroEventsCount = 0;

    this._lastDiagTouchTs = 0;
    this._lastDiagGyroTs = 0;

    const hideCenterHint = () => {
      const hint = document.getElementById('center-hint');
      if (hint && !hint.classList.contains('hidden')) {
        hint.classList.add('hidden');
      }
    };

    const isInteractiveTarget = (target) => {
      if (!target || typeof target.closest !== 'function') return false;
      return !!(
        target.closest('button') ||
        target.closest('#info-drawer') ||
        target.closest('.scene-pills') ||
        target.closest('.aux-controls')
      );
    };

    this._applyDollyAlongGaze = (forwardDelta) => {
      const effectiveYawRad = THREE.MathUtils.degToRad(this.poseTarget.yawDeg + this._gyroYawDeg);
      // Forward gaze in Three.js XZ plane is (-sin(yaw), -cos(yaw))
      const nextRaw = {
        x: this.poseTarget.x - Math.sin(effectiveYawRad) * forwardDelta,
        y: this.poseTarget.y,
        z: this.poseTarget.z - Math.cos(effectiveYawRad) * forwardDelta,
        yawDeg: this.poseTarget.yawDeg,
        pitchDeg: this.poseTarget.pitchDeg
      };
      const clamped = this._clampPose(nextRaw);
      this.poseTarget.x = clamped.x;
      this.poseTarget.y = clamped.y;
      this.poseTarget.z = clamped.z;
      this.poseTarget.yawDeg = clamped.yawDeg;
      this.poseTarget.pitchDeg = clamped.pitchDeg;
    };

    // Desktop Mouse Drag (Rotate / Strafe) + Subtle Ambient Hover
    this._onMouseDown = (e) => {
      if (isInteractiveTarget(e.target)) return;
      if (e.button !== 0 && e.button !== 2) return;
      this._isMouseDragging = true;
      this._mouseDragMode = (e.button === 2 || e.shiftKey) ? 'strafe' : 'look';
      this._mouseDownX = e.clientX;
      this._mouseDownY = e.clientY;
      this._mouseBaseYawDeg = this.poseTarget.yawDeg;
      this._mouseBasePitchDeg = this.poseTarget.pitchDeg;
      this._mouseBasePosX = this.poseTarget.x;
      this._mouseBasePosY = this.poseTarget.y;
      if (this.canvas) {
        this.canvas.style.cursor = 'grabbing';
      }
      hideCenterHint();
    };

    this._onMouseMove = (e) => {
      if (this._isMouseDragging) {
        const dx = e.clientX - this._mouseDownX;
        const dy = e.clientY - this._mouseDownY;
        if (this._mouseDragMode === 'strafe') {
          const nextClamped = this._clampPose({
            x: this._mouseBasePosX - dx * SCENE_GEOMETRY_SPEC.mouseStrafeScale,
            y: this._mouseBasePosY + dy * SCENE_GEOMETRY_SPEC.mouseStrafeScale,
            z: this.poseTarget.z,
            yawDeg: this.poseTarget.yawDeg,
            pitchDeg: this.poseTarget.pitchDeg
          });
          this.poseTarget.x = nextClamped.x;
          this.poseTarget.y = nextClamped.y;
        } else {
          const spanX = Math.max(180, this.width * 0.45);
          const spanY = Math.max(160, this.height * 0.42);
          const nextClamped = this._clampPose({
            x: this.poseTarget.x,
            y: this.poseTarget.y,
            z: this.poseTarget.z,
            yawDeg: this._mouseBaseYawDeg + (dx / spanX) * SCENE_GEOMETRY_SPEC.mouseYawSpanDeg,
            pitchDeg: this._mouseBasePitchDeg + (dy / spanY) * SCENE_GEOMETRY_SPEC.mousePitchSpanDeg
          });
          this.poseTarget.yawDeg = nextClamped.yawDeg;
          this.poseTarget.pitchDeg = nextClamped.pitchDeg;
        }
      } else if (!this._isTouching) {
        this._hoverNorm.targetX = (e.clientX / Math.max(1, this.width)) * 2 - 1;
        this._hoverNorm.targetY = -(e.clientY / Math.max(1, this.height)) * 2 + 1;
      }
    };

    this._onMouseUp = () => {
      if (!this._isMouseDragging) return;
      this._isMouseDragging = false;
      if (this.canvas) {
        this.canvas.style.cursor = 'grab';
      }
    };

    this._onContextMenu = (e) => {
      if (!isInteractiveTarget(e.target)) {
        e.preventDefault();
      }
    };

    // Desktop Scroll Wheel / Trackpad Pinch -> Forward & Backward Dolly
    this._onWheel = (e) => {
      if (isInteractiveTarget(e.target)) return;
      if (e.cancelable) e.preventDefault();
      const scale = e.ctrlKey
        ? SCENE_GEOMETRY_SPEC.trackpadPinchDollyScale
        : SCENE_GEOMETRY_SPEC.wheelDollyScale;
      const forwardDelta = -e.deltaY * scale;
      this._applyDollyAlongGaze(forwardDelta);
      hideCenterHint();
    };

    // Mobile Multi-Touch: 1-Finger Look Rotation, 2-Finger Pinch Dolly + Midpoint Strafe, Double-Tap Reset
    const initSingleFingerLook = (touch) => {
      this._touchMode = 'look';
      this._touchStartX = touch.clientX;
      this._touchStartY = touch.clientY;
      this._touchBaseYawDeg = this.poseTarget.yawDeg;
      this._touchBasePitchDeg = this.poseTarget.pitchDeg;
      this._touchMaxTravelPx = 0;
      this._touchStartTimeMs = performance.now();
    };

    const initTwoFingerPinchStrafe = (t0, t1) => {
      this._touchMode = 'pinch_strafe';
      const dx = t1.clientX - t0.clientX;
      const dy = t1.clientY - t0.clientY;
      this._prevPinchDist = Math.hypot(dx, dy);
      this._prevPinchMidX = 0.5 * (t0.clientX + t1.clientX);
      this._prevPinchMidY = 0.5 * (t0.clientY + t1.clientY);
      this._touchMaxTravelPx = 999;
    };

    this._onTouchStart = (e) => {
      if (isInteractiveTarget(e.target)) return;
      this._isTouching = true;
      this._hoverNorm.targetX = 0;
      this._hoverNorm.targetY = 0;

      if (e.touches.length === 1) {
        initSingleFingerLook(e.touches[0]);
      } else if (e.touches.length >= 2) {
        initTwoFingerPinchStrafe(e.touches[0], e.touches[1]);
      }
    };

    this._onTouchMove = (e) => {
      if (!this._isTouching) return;
      if (e.cancelable) e.preventDefault();

      if (e.touches.length === 1 && this._touchMode === 'look') {
        const dx = e.touches[0].clientX - this._touchStartX;
        const dy = e.touches[0].clientY - this._touchStartY;
        this._touchMaxTravelPx = Math.max(this._touchMaxTravelPx, Math.hypot(dx, dy));

        const spanX = Math.max(120, this.width * 0.42);
        const spanY = Math.max(120, this.height * 0.38);
        const nextClamped = this._clampPose({
          x: this.poseTarget.x,
          y: this.poseTarget.y,
          z: this.poseTarget.z,
          yawDeg: this._touchBaseYawDeg + (dx / spanX) * SCENE_GEOMETRY_SPEC.touchYawSpanDeg,
          pitchDeg: this._touchBasePitchDeg + (dy / spanY) * SCENE_GEOMETRY_SPEC.touchPitchSpanDeg
        });
        this.poseTarget.yawDeg = nextClamped.yawDeg;
        this.poseTarget.pitchDeg = nextClamped.pitchDeg;
        hideCenterHint();

        const now = Date.now();
        if (window.__sendDiag && now - this._lastDiagTouchTs > 1500) {
          this._lastDiagTouchTs = now;
          window.__sendDiag('touch_look', {
            yaw: Number(this.poseTarget.yawDeg.toFixed(1)),
            pitch: Number(this.poseTarget.pitchDeg.toFixed(1)),
            z: Math.round(this.poseTarget.z)
          });
        }
      } else if (e.touches.length >= 2) {
        if (this._touchMode !== 'pinch_strafe') {
          initTwoFingerPinchStrafe(e.touches[0], e.touches[1]);
          return;
        }
        const t0 = e.touches[0];
        const t1 = e.touches[1];
        const dist = Math.hypot(t1.clientX - t0.clientX, t1.clientY - t0.clientY);
        const midX = 0.5 * (t0.clientX + t1.clientX);
        const midY = 0.5 * (t0.clientY + t1.clientY);

        const distDelta = dist - this._prevPinchDist;
        const midDx = midX - this._prevPinchMidX;
        const midDy = midY - this._prevPinchMidY;

        this._prevPinchDist = dist;
        this._prevPinchMidX = midX;
        this._prevPinchMidY = midY;

        const effectiveYawRad = THREE.MathUtils.degToRad(this.poseTarget.yawDeg + this._gyroYawDeg);
        const dollyStep = distDelta * SCENE_GEOMETRY_SPEC.pinchDollyScale;
        const strafeRight = -midDx * SCENE_GEOMETRY_SPEC.touchStrafeScale;
        const strafeUp = midDy * SCENE_GEOMETRY_SPEC.touchStrafeScale;

        const nextClamped = this._clampPose({
          x:
            this.poseTarget.x -
            Math.sin(effectiveYawRad) * dollyStep +
            Math.cos(effectiveYawRad) * strafeRight,
          y: this.poseTarget.y + strafeUp,
          z:
            this.poseTarget.z -
            Math.cos(effectiveYawRad) * dollyStep -
            Math.sin(effectiveYawRad) * strafeRight,
          yawDeg: this.poseTarget.yawDeg,
          pitchDeg: this.poseTarget.pitchDeg
        });
        this.poseTarget.x = nextClamped.x;
        this.poseTarget.y = nextClamped.y;
        this.poseTarget.z = nextClamped.z;
        this.poseTarget.yawDeg = nextClamped.yawDeg;
        this.poseTarget.pitchDeg = nextClamped.pitchDeg;
        hideCenterHint();

        const now = Date.now();
        if (window.__sendDiag && now - this._lastDiagTouchTs > 1500) {
          this._lastDiagTouchTs = now;
          window.__sendDiag('touch_dolly_strafe', {
            x: Math.round(this.poseTarget.x),
            y: Math.round(this.poseTarget.y),
            z: Math.round(this.poseTarget.z)
          });
        }
      }
    };

    this._onTouchEnd = (e) => {
      if (!this._isTouching) return;

      if (e.touches && e.touches.length === 1) {
        // Smoothly transition from 2-finger pinch back to 1-finger look without jumping
        initSingleFingerLook(e.touches[0]);
        return;
      }

      if (!e.touches || e.touches.length === 0) {
        const nowPerf = performance.now();
        if (
          this._touchMode === 'look' &&
          this._touchMaxTravelPx < 16 &&
          nowPerf - this._touchStartTimeMs < 260
        ) {
          if (nowPerf - this._lastSingleTapMs < 300) {
            this.resetCameraPose();
            this._lastSingleTapMs = 0;
          } else {
            this._lastSingleTapMs = nowPerf;
          }
        }
        this._isTouching = false;
        this._touchMode = 'none';
        // Re-zero relative gyro origin so resting hand posture after gesture becomes neutral
        this._gyroOrigin = null;
        this._gyroYawDeg = 0;
        this._gyroPitchDeg = 0;
      }
    };

    this._onDeviceOrientation = (e) => {
      if (!this.gyroActive || this._isTouching) return;
      if (e.gamma === null && e.beta === null && e.alpha === null) return;

      const alpha = typeof e.alpha === 'number' ? e.alpha : 0;
      const beta = typeof e.beta === 'number' ? e.beta : 0;
      const gamma = typeof e.gamma === 'number' ? e.gamma : 0;

      this._gyroEventsCount += 1;
      if (!this._gyroOrigin) {
        this._gyroOrigin = { alpha, beta, gamma };
      }

      let dAlpha = alpha - this._gyroOrigin.alpha;
      if (dAlpha > 180) dAlpha -= 360;
      if (dAlpha < -180) dAlpha += 360;
      const dGamma = gamma - this._gyroOrigin.gamma;
      const dBeta = beta - this._gyroOrigin.beta;

      const normYaw = THREE.MathUtils.clamp(
        (dGamma / SCENE_GEOMETRY_SPEC.gyroTiltSensitivityDeg) -
          (dAlpha / SCENE_GEOMETRY_SPEC.gyroYawSensitivityDeg),
        -1,
        1
      );
      const normPitch = THREE.MathUtils.clamp(
        -(dBeta / SCENE_GEOMETRY_SPEC.gyroTiltSensitivityDeg),
        -1,
        1
      );

      this._gyroYawDeg = -normYaw * SCENE_GEOMETRY_SPEC.gyroMaxYawDeg;
      this._gyroPitchDeg = normPitch * SCENE_GEOMETRY_SPEC.gyroMaxPitchDeg;

      if (Math.abs(normYaw) > 0.08 || Math.abs(normPitch) > 0.08) {
        hideCenterHint();
      }

      const now = Date.now();
      if (window.__sendDiag && now - this._lastDiagGyroTs > 2000) {
        this._lastDiagGyroTs = now;
        window.__sendDiag('gyro_sample', {
          count: this._gyroEventsCount,
          yaw: Number(this._gyroYawDeg.toFixed(1)),
          pitch: Number(this._gyroPitchDeg.toFixed(1))
        });
      }
    };

    this._onResize = () => {
      this.width = window.innerWidth;
      this.height = window.innerHeight;
      this.camera.aspect = this.width / this.height;
      this.renderer.setSize(this.width, this.height);
      if (this.rtCurrent) this.rtCurrent.setSize(this.width, this.height);
      if (this.rtNext) this.rtNext.setSize(this.width, this.height);
      if (this.transitionMaterial && this.transitionMaterial.uniforms) {
        this.transitionMaterial.uniforms.uAspect.value = this.camera.aspect;
      }
      this.applyDynamicWidescreenScaling();
    };

    this._onKeyDown = (e) => {
      if (e.key === '1') this.transitionTo('herzliya');
      if (e.key === '2') this.transitionTo('denya');
      if (e.key === ' ') {
        e.preventDefault();
        this.isPaused = !this.isPaused;
      }
      const lower = e.key.toLowerCase();
      if (lower === 'r') {
        this.resetCameraPose();
      }
      if (lower === 'f') {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(() => {});
        } else {
          document.exitFullscreen().catch(() => {});
        }
      }
      if (
        ['w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(
          lower
        )
      ) {
        this._keysHeld.add(lower);
        hideCenterHint();
      }
    };

    this._onKeyUp = (e) => {
      this._keysHeld.delete(e.key.toLowerCase());
    };

    this._onWindowBlur = () => {
      this._keysHeld.clear();
      this._isMouseDragging = false;
    };

    window.__PROTOTYPE_STATE__ = {
      currentScene: this.currentSceneId,
      cameraPosition: { x: 0, y: 0, z: this.baseCameraZ },
      cameraRotationDeg: { yaw: 0, pitch: 0 },
      layersReady: false,
      isTransitioning: false,
      innerLayerScale: this._innerScale,
      layer0Scale: this._layer0Scale,
      bgExpansionFactor: SCENE_GEOMETRY_SPEC.bgExpansionFactor,
      fps: 60,
      aspect: this.width / this.height
    };
    window.__tamiSceneManager = this;

    this.initRenderer();
    this.initScenes();
    this.initTransitionPipeline();
    this.bindEvents();
    this.animate();
  }

  _clampPose(rawPose) {
    return clampPoseToRoomFrustum(
      rawPose,
      this.camera ? this.camera.fov : SCENE_GEOMETRY_SPEC.desktopFovDeg,
      this.camera ? this.camera.aspect : this.width / Math.max(1, this.height),
      this._layer0Scale,
      SCENE_GEOMETRY_SPEC
    );
  }

  setCameraPose(partialPose = {}, immediate = false) {
    const merged = {
      x: typeof partialPose.x === 'number' ? partialPose.x : this.poseTarget.x,
      y: typeof partialPose.y === 'number' ? partialPose.y : this.poseTarget.y,
      z: typeof partialPose.z === 'number' ? partialPose.z : this.poseTarget.z,
      yawDeg: typeof partialPose.yawDeg === 'number' ? partialPose.yawDeg : this.poseTarget.yawDeg,
      pitchDeg:
        typeof partialPose.pitchDeg === 'number' ? partialPose.pitchDeg : this.poseTarget.pitchDeg
    };
    const clamped = this._clampPose(merged);
    this.poseTarget.x = clamped.x;
    this.poseTarget.y = clamped.y;
    this.poseTarget.z = clamped.z;
    this.poseTarget.yawDeg = clamped.yawDeg;
    this.poseTarget.pitchDeg = clamped.pitchDeg;

    if (immediate) {
      this.poseCurrent.x = clamped.x;
      this.poseCurrent.y = clamped.y;
      this.poseCurrent.z = clamped.z;
      this.poseCurrent.yawDeg = clamped.yawDeg;
      this.poseCurrent.pitchDeg = clamped.pitchDeg;
      this._hoverNorm.x = 0;
      this._hoverNorm.y = 0;
      this._hoverNorm.targetX = 0;
      this._hoverNorm.targetY = 0;
      this.pointer.x = this.pointer.targetX;
      this.pointer.y = this.pointer.targetY;
      this._syncCameraTransform();
    }
    return { ...clamped };
  }

  resetCameraPose(immediate = false) {
    this._gyroOrigin = null;
    this._gyroYawDeg = 0;
    this._gyroPitchDeg = 0;
    this.pointer.targetX = 0;
    this.pointer.targetY = 0;
    this._hoverNorm.targetX = 0;
    this._hoverNorm.targetY = 0;
    if (immediate) {
      this.pointer.x = 0;
      this.pointer.y = 0;
    }
    if (window.__sendDiag) {
      window.__sendDiag('pose_reset', { scene: this.currentSceneId });
    }
    return this.setCameraPose(
      { x: 0, y: 0, z: this.baseCameraZ, yawDeg: 0, pitchDeg: 0 },
      immediate
    );
  }

  initRenderer() {
    let glContext = null;
    try {
      glContext = this.canvas.getContext('webgl2') || this.canvas.getContext('webgl');
    } catch (e) {
      console.warn('WebGL context query warning:', e);
    }

    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      context: glContext || undefined,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance'
    });
    this.renderer.setClearColor(0x121314, 1.0);
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    const aspect = this.width / this.height;
    const initialFov =
      aspect < 1.0 ? SCENE_GEOMETRY_SPEC.portraitFovDeg : SCENE_GEOMETRY_SPEC.desktopFovDeg;
    this.camera = new THREE.PerspectiveCamera(initialFov, aspect, 1, 3000);
    this.camera.rotation.order = 'YXZ';
    this.camera.position.set(0, 0, this.baseCameraZ);
    this.camera.rotation.set(0, 0, 0);

    if (this.canvas) {
      this.canvas.style.cursor = 'grab';
    }

    window.__PROTOTYPE_STATE__.webglVersion = this.renderer.capabilities.isWebGL2
      ? 'webgl2'
      : 'webgl';

    if (window.__sendDiag) {
      const gl = this.renderer.getContext();
      let gpu = 'unknown';
      try {
        const dbg = gl.getExtension('WEBGL_debug_renderer_info');
        if (dbg) gpu = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
      } catch (e) {}
      window.__sendDiag('webgl_init', {
        version: window.__PROTOTYPE_STATE__.webglVersion,
        maxTex: this.renderer.capabilities.maxTextureSize,
        gpu: gpu
      });
    }
  }

  _updateLoadingText(text) {
    const el = document.querySelector('#loading-overlay .loading-text');
    if (el) el.textContent = text;
  }

  _loadTrackedTexture(url) {
    return this.textureLoader.load(
      url,
      (tex) => {
        this._loadedTextures += 1;
        this._updateLoadingText(
          `טוען מרחבים רב-שכבתיים... (${this._loadedTextures}/${this._totalTextures})`
        );
        if (window.__sendDiag) {
          window.__sendDiag('texture_loaded', {
            url: url,
            loaded: this._loadedTextures,
            total: this._totalTextures,
            w: tex.image ? tex.image.width : 0,
            h: tex.image ? tex.image.height : 0
          });
        }
        if (this._loadedTextures >= this._totalTextures) {
          this.checkAllLayersReady();
        }
      },
      undefined,
      (err) => {
        const msg = `Failed to load required scene texture: ${url}`;
        console.error(msg, err);
        if (window.__sendDiag) {
          window.__sendDiag('texture_error', {
            url: url,
            err: String(err && (err.message || err.type || err))
          });
        }
      }
    );
  }

  initScenes() {
    this.textureLoader = new THREE.TextureLoader();
    this._dustSpriteTexture = createSoftRadialDustTexture();
    this.sceneObjects = {};

    const distBg = this.baseCameraZ - SCENE_GEOMETRY_SPEC.bgZ;

    for (const [key, cfg] of Object.entries(this.sceneConfigs)) {
      const scene = new THREE.Scene();
      const group = new THREE.Group();
      scene.add(group);

      // Layer 0: Subdivided 2700x1800 Outpainted + LaMa-Inpainted Room Mesh with Depth Anything V2 Relief
      const bgGeo = new THREE.PlaneGeometry(
        SCENE_GEOMETRY_SPEC.bgBaseWidth,
        SCENE_GEOMETRY_SPEC.bgBaseHeight,
        SCENE_GEOMETRY_SPEC.bgSegmentsX,
        SCENE_GEOMETRY_SPEC.bgSegmentsY
      );
      const bgTex = this._loadTrackedTexture(cfg.bgTexture);
      const depthTex = this._loadTrackedTexture(cfg.depthTexture);
      bgTex.minFilter = THREE.LinearFilter;
      bgTex.magFilter = THREE.LinearFilter;
      depthTex.minFilter = THREE.LinearFilter;
      depthTex.magFilter = THREE.LinearFilter;

      const bgMat = new THREE.ShaderMaterial({
        vertexShader: ROOM_DEPTH_VERTEX_SHADER,
        fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
        uniforms: {
          uColorMap: { value: bgTex },
          uDepthMap: { value: depthTex },
          uDepthAmplitude: { value: SCENE_GEOMETRY_SPEC.roomDepthReliefAmplitude },
          uDistBg: { value: distBg },
          uPinMarginUv: { value: SCENE_GEOMETRY_SPEC.roomPinMarginUv }
        },
        depthWrite: true,
        depthTest: true
      });
      const layer0 = new THREE.Mesh(bgGeo, bgMat);
      layer0.position.set(0, 0, SCENE_GEOMETRY_SPEC.bgZ);
      group.add(layer0);

      // Layer 1: Neural-Segmented Midground Sculptures (1500x1000 inner frame at Z = -150)
      const midGeo = new THREE.PlaneGeometry(
        SCENE_GEOMETRY_SPEC.midBaseWidth,
        SCENE_GEOMETRY_SPEC.midBaseHeight
      );
      const midTex = this._loadTrackedTexture(cfg.midTexture);
      midTex.minFilter = THREE.LinearFilter;
      const midMat = new THREE.MeshBasicMaterial({
        map: midTex,
        transparent: true,
        depthWrite: false
      });
      const layer1 = new THREE.Mesh(midGeo, midMat);
      layer1.position.set(0, 0, SCENE_GEOMETRY_SPEC.midZ);
      layer1.renderOrder = 1;
      group.add(layer1);

      // Layer 2: Neural-Segmented Foreground Sculptures (1500x1000 inner frame at Z = 0)
      const fgGeo = new THREE.PlaneGeometry(
        SCENE_GEOMETRY_SPEC.fgBaseWidth,
        SCENE_GEOMETRY_SPEC.fgBaseHeight
      );
      const fgTex = this._loadTrackedTexture(cfg.fgTexture);
      fgTex.minFilter = THREE.LinearFilter;
      const fgMat = new THREE.MeshBasicMaterial({
        map: fgTex,
        transparent: true,
        depthWrite: false
      });
      const layer2 = new THREE.Mesh(fgGeo, fgMat);
      layer2.position.set(0, 0, SCENE_GEOMETRY_SPEC.fgZ);
      layer2.renderOrder = 2;
      group.add(layer2);

      // Layer 2.5: Bounded Architectural Mirror Aperture Reflections (Z = -340)
      let reflectionMesh = null;
      if (cfg.hasReflection && cfg.reflectionTexture) {
        const refGeo = new THREE.PlaneGeometry(
          SCENE_GEOMETRY_SPEC.fgBaseWidth,
          SCENE_GEOMETRY_SPEC.fgBaseHeight
        );
        const refTex = this._loadTrackedTexture(cfg.reflectionTexture);
        refTex.minFilter = THREE.LinearFilter;
        const refMat = new THREE.MeshBasicMaterial({
          map: refTex,
          transparent: true,
          opacity: 0.55,
          depthWrite: false,
          blending: THREE.NormalBlending
        });
        reflectionMesh = new THREE.Mesh(refGeo, refMat);
        reflectionMesh.position.set(0, 0, SCENE_GEOMETRY_SPEC.reflectionZ);
        reflectionMesh.renderOrder = 3;
        group.add(reflectionMesh);
      }

      // Layer 3: Soft Radial Dust Motes spanning the expanded 3D gallery volume
      const particleGeo = new THREE.BufferGeometry();
      const count = cfg.particleCount;
      const positions = new Float32Array(count * 3);
      for (let i = 0; i < count * 3; i += 3) {
        positions[i] = (Math.random() - 0.5) * 920;
        positions[i + 1] = (Math.random() - 0.5) * 560;
        positions[i + 2] = (Math.random() - 0.5) * 320 + 110;
      }
      particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const particleMat = new THREE.PointsMaterial({
        size: 6.5,
        sizeAttenuation: true,
        map: this._dustSpriteTexture,
        color: cfg.particleColor,
        transparent: true,
        opacity: 0.28,
        depthWrite: false,
        blending: THREE.AdditiveBlending
      });
      const layer3 = new THREE.Points(particleGeo, particleMat);
      layer3.renderOrder = 4;
      group.add(layer3);

      this.sceneObjects[key] = {
        scene,
        group,
        layer0,
        layer1,
        layer2,
        layer3,
        reflectionMesh
      };
    }

    this.applyDynamicWidescreenScaling();
  }

  applyDynamicWidescreenScaling() {
    const aspect = this.camera.aspect;
    const isPortrait = aspect < 1.0;

    this.camera.fov = isPortrait
      ? SCENE_GEOMETRY_SPEC.portraitFovDeg
      : SCENE_GEOMETRY_SPEC.desktopFovDeg;
    this.camera.updateProjectionMatrix();

    const distBg = this.baseCameraZ - SCENE_GEOMETRY_SPEC.bgZ;
    const distMid = this.baseCameraZ - SCENE_GEOMETRY_SPEC.midZ;
    const distFg = this.baseCameraZ - SCENE_GEOMETRY_SPEC.fgZ;
    const distRef = this.baseCameraZ - SCENE_GEOMETRY_SPEC.reflectionZ;

    const halfFovRad = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const visibleHeightAtBg = 2 * distBg * Math.tan(halfFovRad);
    const visibleWidthAtBg = visibleHeightAtBg * aspect;

    // Inner 1500x1000 scale ensures wide viewports stay generously framed, while Layer 0 scales by 1.80x
    // around the exact same optical center so bore-sight alignment with Mid/Fg/Reflection is 1:1 exact.
    const reqInnerScaleX =
      (visibleWidthAtBg * SCENE_GEOMETRY_SPEC.frustumInnerCoverageFactor) /
      SCENE_GEOMETRY_SPEC.bgBaseWidth;
    const reqInnerScaleY =
      (visibleHeightAtBg * SCENE_GEOMETRY_SPEC.frustumInnerCoverageFactor) /
      SCENE_GEOMETRY_SPEC.bgBaseHeight;
    const innerScale = Math.max(SCENE_GEOMETRY_SPEC.minInnerScale, reqInnerScaleX, reqInnerScaleY);
    const layer0Scale = innerScale * SCENE_GEOMETRY_SPEC.bgExpansionFactor;

    this._innerScale = innerScale;
    this._layer0Scale = layer0Scale;

    const innerWorldWidthAtBg = SCENE_GEOMETRY_SPEC.bgBaseWidth * innerScale;
    const midScale = (innerWorldWidthAtBg * (distMid / distBg)) / SCENE_GEOMETRY_SPEC.midBaseWidth;
    const fgScale = (innerWorldWidthAtBg * (distFg / distBg)) / SCENE_GEOMETRY_SPEC.fgBaseWidth;
    const refScale = (innerWorldWidthAtBg * (distRef / distBg)) / SCENE_GEOMETRY_SPEC.fgBaseWidth;

    for (const key in this.sceneObjects) {
      const obj = this.sceneObjects[key];
      if (!obj) continue;
      if (obj.layer0) obj.layer0.scale.set(layer0Scale, layer0Scale, 1.0);
      if (obj.layer1) obj.layer1.scale.set(midScale, midScale, 1.0);
      if (obj.layer2) obj.layer2.scale.set(fgScale, fgScale, 1.0);
      if (obj.reflectionMesh) obj.reflectionMesh.scale.set(refScale, refScale, 1.0);
    }

    // Re-clamp active camera pose to updated viewport frustum
    const reclamped = this._clampPose(this.poseTarget);
    this.poseTarget.x = reclamped.x;
    this.poseTarget.y = reclamped.y;
    this.poseTarget.z = reclamped.z;
    this.poseTarget.yawDeg = reclamped.yawDeg;
    this.poseTarget.pitchDeg = reclamped.pitchDeg;

    window.__PROTOTYPE_STATE__.innerLayerScale = innerScale;
    window.__PROTOTYPE_STATE__.layer0Scale = layer0Scale;
    window.__PROTOTYPE_STATE__.aspect = aspect;
  }

  checkAllLayersReady() {
    if (this.layersReady) return;
    this.layersReady = true;
    window.__PROTOTYPE_STATE__.layersReady = true;
    if (window.__sendDiag) {
      window.__sendDiag('layers_ready', {
        scene: this.currentSceneId,
        fov: this.camera.fov,
        layer0Scale: Number(window.__PROTOTYPE_STATE__.layer0Scale.toFixed(3)),
        aspect: Number(this.camera.aspect.toFixed(3))
      });
    }
    this.onReady();
  }

  initTransitionPipeline() {
    this.rtCurrent = new THREE.WebGLRenderTarget(this.width, this.height);
    this.rtNext = new THREE.WebGLRenderTarget(this.width, this.height);

    this.transitionCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.transitionScene = new THREE.Scene();

    this.transitionMaterial = new THREE.ShaderMaterial({
      vertexShader: TRANSITION_VERTEX_SHADER,
      fragmentShader: TRANSITION_FRAGMENT_SHADER,
      uniforms: {
        uTexCurrent: { value: null },
        uTexNext: { value: null },
        uProgress: { value: 0.0 },
        uAspect: { value: this.width / this.height },
        uPointer: { value: new THREE.Vector2(0, 0) }
      },
      depthWrite: false,
      depthTest: false
    });

    const quadGeo = new THREE.PlaneGeometry(2, 2);
    this.transitionQuad = new THREE.Mesh(quadGeo, this.transitionMaterial);
    this.transitionScene.add(this.transitionQuad);
  }

  bindEvents() {
    window.addEventListener('mousedown', this._onMouseDown);
    window.addEventListener('mousemove', this._onMouseMove, { passive: true });
    window.addEventListener('mouseup', this._onMouseUp);
    window.addEventListener('contextmenu', this._onContextMenu);
    window.addEventListener('wheel', this._onWheel, { passive: false });
    window.addEventListener('touchstart', this._onTouchStart, { passive: true });
    window.addEventListener('touchmove', this._onTouchMove, { passive: false });
    window.addEventListener('touchend', this._onTouchEnd, { passive: true });
    window.addEventListener('touchcancel', this._onTouchEnd, { passive: true });
    window.addEventListener('deviceorientation', this._onDeviceOrientation, { passive: true });
    window.addEventListener('resize', this._onResize);
    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
    window.addEventListener('blur', this._onWindowBlur);
  }

  switchScene(nextSceneId) {
    return this.transitionTo(nextSceneId);
  }

  transitionTo(nextSceneId) {
    if (!nextSceneId) return false;
    if (this.currentSceneId === nextSceneId) return false;
    if (this.isTransitioning) return false;
    if (!this.sceneConfigs[nextSceneId]) return false;

    this.isTransitioning = true;
    this.targetSceneId = nextSceneId;
    this.transitionStartTime = performance.now();
    this.transitionProgress = 0.0;

    if (window.__sendDiag) {
      window.__sendDiag('scene_transition', { from: this.currentSceneId, to: nextSceneId });
    }

    window.__PROTOTYPE_STATE__.isTransitioning = true;
    this.onSceneChange(nextSceneId);
    return true;
  }

  enableGyroscope(callback) {
    const activateAndMonitor = (active) => {
      this._gyroOrigin = null;
      this._gyroYawDeg = 0;
      this._gyroPitchDeg = 0;
      if (active) {
        const beforeCount = this._gyroEventsCount;
        setTimeout(() => {
          if (this.gyroActive && this._gyroEventsCount === beforeCount) {
            if (window.__sendDiag) {
              window.__sendDiag('gyro_no_events', {
                msg: 'No deviceorientation events received within 1500ms'
              });
            }
          }
        }, 1500);
      }
      if (window.__sendDiag) {
        window.__sendDiag('gyro_toggled', { active: active });
      }
    };

    const updateUI = (active) => {
      const gyroBtn = document.getElementById('gyro-toggle');
      if (gyroBtn) {
        if (active) {
          gyroBtn.classList.add('active');
          gyroBtn.textContent = '✓ ג׳יירו פעיל';
        } else {
          gyroBtn.classList.remove('active');
          gyroBtn.textContent = '⟲ מצב הטיה: עכבר / ג׳יירו';
        }
      }
      activateAndMonitor(active);
      if (typeof callback === 'function') callback(active);
    };

    if (
      typeof DeviceOrientationEvent !== 'undefined' &&
      typeof DeviceOrientationEvent.requestPermission === 'function'
    ) {
      return DeviceOrientationEvent.requestPermission()
        .then((permissionState) => {
          const granted = permissionState === 'granted';
          this.gyroActive = granted;
          updateUI(granted);
          return granted;
        })
        .catch((err) => {
          console.error('DeviceOrientation permission error:', err);
          this.gyroActive = false;
          updateUI(false);
          return false;
        });
    } else {
      this.gyroActive = !this.gyroActive;
      updateUI(this.gyroActive);
      return this.gyroActive;
    }
  }

  _stepKeyboardNavigation(dtSec) {
    if (this._keysHeld.size === 0) return;

    const moveDist = SCENE_GEOMETRY_SPEC.keyMoveSpeedPerSec * dtSec;
    let forward = 0;
    let right = 0;
    let up = 0;

    if (this._keysHeld.has('w') || this._keysHeld.has('arrowup')) forward += moveDist;
    if (this._keysHeld.has('s') || this._keysHeld.has('arrowdown')) forward -= moveDist;
    if (this._keysHeld.has('d') || this._keysHeld.has('arrowright')) right += moveDist;
    if (this._keysHeld.has('a') || this._keysHeld.has('arrowleft')) right -= moveDist;
    if (this._keysHeld.has('e')) up += moveDist;
    if (this._keysHeld.has('q')) up -= moveDist;

    if (forward === 0 && right === 0 && up === 0) return;

    const yawRad = THREE.MathUtils.degToRad(this.poseTarget.yawDeg + this._gyroYawDeg);
    const nextClamped = this._clampPose({
      x: this.poseTarget.x - Math.sin(yawRad) * forward + Math.cos(yawRad) * right,
      y: this.poseTarget.y + up,
      z: this.poseTarget.z - Math.cos(yawRad) * forward - Math.sin(yawRad) * right,
      yawDeg: this.poseTarget.yawDeg,
      pitchDeg: this.poseTarget.pitchDeg
    });
    this.poseTarget.x = nextClamped.x;
    this.poseTarget.y = nextClamped.y;
    this.poseTarget.z = nextClamped.z;
  }

  _syncCameraTransform() {
    // Combine damped 6-DOF pose + gyro + subtle hover + programmatic pointer excursion,
    // then run the analytical ray-frustum guard so the final rendered camera pose NEVER exposes borders.
    const gyroYaw = this.gyroActive ? this._gyroYawDeg : 0;
    const gyroPitch = this.gyroActive ? this._gyroPitchDeg : 0;

    const combinedPose = this._clampPose({
      x:
        this.poseCurrent.x +
        this._hoverNorm.x * SCENE_GEOMETRY_SPEC.hoverAmbientShiftX +
        this.pointer.x * SCENE_GEOMETRY_SPEC.pointerExcursionShiftX,
      y:
        this.poseCurrent.y +
        this._hoverNorm.y * SCENE_GEOMETRY_SPEC.hoverAmbientShiftY +
        this.pointer.y * SCENE_GEOMETRY_SPEC.pointerExcursionShiftY,
      z: this.poseCurrent.z,
      yawDeg:
        this.poseCurrent.yawDeg +
        gyroYaw -
        this._hoverNorm.x * SCENE_GEOMETRY_SPEC.hoverAmbientYawDeg -
        this.pointer.x * SCENE_GEOMETRY_SPEC.pointerExcursionYawDeg,
      pitchDeg:
        this.poseCurrent.pitchDeg +
        gyroPitch +
        this._hoverNorm.y * SCENE_GEOMETRY_SPEC.hoverAmbientPitchDeg +
        this.pointer.y * SCENE_GEOMETRY_SPEC.pointerExcursionPitchDeg
    });

    this.camera.position.set(combinedPose.x, combinedPose.y, combinedPose.z);
    this.camera.rotation.set(
      THREE.MathUtils.degToRad(combinedPose.pitchDeg),
      THREE.MathUtils.degToRad(combinedPose.yawDeg),
      0
    );
    this.camera.updateMatrixWorld();

    window.__PROTOTYPE_STATE__.cameraPosition.x = this.camera.position.x;
    window.__PROTOTYPE_STATE__.cameraPosition.y = this.camera.position.y;
    window.__PROTOTYPE_STATE__.cameraPosition.z = this.camera.position.z;
    window.__PROTOTYPE_STATE__.cameraRotationDeg.yaw = combinedPose.yawDeg;
    window.__PROTOTYPE_STATE__.cameraRotationDeg.pitch = combinedPose.pitchDeg;
  }

  animate() {
    this.animId = requestAnimationFrame(() => this.animate());

    const nowMs = performance.now();
    const dtSec = Math.min(0.05, Math.max(0.001, (nowMs - this._lastFrameTimeMs) * 0.001));
    this._lastFrameTimeMs = nowMs;

    this._stepKeyboardNavigation(dtSec);

    const activeDrag = this._isTouching || this._isMouseDragging;
    const poseDamping = activeDrag ? 0.22 : 0.11;
    const pointerDamping = 0.09;

    this.pointer.x += (this.pointer.targetX - this.pointer.x) * pointerDamping;
    this.pointer.y += (this.pointer.targetY - this.pointer.y) * pointerDamping;
    this._hoverNorm.x += (this._hoverNorm.targetX - this._hoverNorm.x) * 0.07;
    this._hoverNorm.y += (this._hoverNorm.targetY - this._hoverNorm.y) * 0.07;

    this.poseCurrent.x += (this.poseTarget.x - this.poseCurrent.x) * poseDamping;
    this.poseCurrent.y += (this.poseTarget.y - this.poseCurrent.y) * poseDamping;
    this.poseCurrent.z += (this.poseTarget.z - this.poseCurrent.z) * poseDamping;
    this.poseCurrent.yawDeg += (this.poseTarget.yawDeg - this.poseCurrent.yawDeg) * poseDamping;
    this.poseCurrent.pitchDeg +=
      (this.poseTarget.pitchDeg - this.poseCurrent.pitchDeg) * poseDamping;

    this._syncCameraTransform();

    const denyaObj = this.sceneObjects.denya;
    if (denyaObj && denyaObj.reflectionMesh) {
      const viewExcursion = Math.min(
        1.0,
        Math.abs(window.__PROTOTYPE_STATE__.cameraRotationDeg.yaw) / 20.0 +
          Math.abs(this.camera.position.x) / 140.0
      );
      denyaObj.reflectionMesh.material.opacity = 0.35 + 0.45 * viewExcursion;
    }

    if (!this.isPaused) {
      for (const key in this.sceneObjects) {
        const particles = this.sceneObjects[key].layer3;
        if (particles) {
          particles.rotation.y += 0.0004;
          particles.rotation.x += 0.0002;
        }
      }
    }

    const normYaw = THREE.MathUtils.clamp(
      window.__PROTOTYPE_STATE__.cameraRotationDeg.yaw / SCENE_GEOMETRY_SPEC.maxYawDeg,
      -1,
      1
    );
    const normPitch = THREE.MathUtils.clamp(
      window.__PROTOTYPE_STATE__.cameraRotationDeg.pitch / SCENE_GEOMETRY_SPEC.maxPitchDeg,
      -1,
      1
    );

    if (this.isTransitioning) {
      if (!this.targetSceneId || !this.sceneObjects[this.targetSceneId]) {
        this.isTransitioning = false;
        this.targetSceneId = null;
        window.__PROTOTYPE_STATE__.isTransitioning = false;
        return;
      }

      const elapsed = nowMs - this.transitionStartTime;
      this.transitionProgress = Math.min(1.0, elapsed / this.transitionDuration);

      this.renderer.setRenderTarget(this.rtCurrent);
      this.renderer.render(this.sceneObjects[this.currentSceneId].scene, this.camera);

      this.renderer.setRenderTarget(this.rtNext);
      this.renderer.render(this.sceneObjects[this.targetSceneId].scene, this.camera);

      this.renderer.setRenderTarget(null);
      this.transitionMaterial.uniforms.uTexCurrent.value = this.rtCurrent.texture;
      this.transitionMaterial.uniforms.uTexNext.value = this.rtNext.texture;
      this.transitionMaterial.uniforms.uProgress.value = this.transitionProgress;
      this.transitionMaterial.uniforms.uPointer.value.set(normYaw, normPitch);
      this.renderer.render(this.transitionScene, this.transitionCamera);

      if (this.transitionProgress >= 1.0) {
        this.isTransitioning = false;
        this.currentSceneId = this.targetSceneId;
        this.targetSceneId = null;
        window.__PROTOTYPE_STATE__.currentScene = this.currentSceneId;
        window.__PROTOTYPE_STATE__.isTransitioning = false;
      }
    } else {
      this.renderer.setRenderTarget(null);
      this.renderer.render(this.sceneObjects[this.currentSceneId].scene, this.camera);
    }
  }

  dispose() {
    if (this.animId !== null) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }

    window.removeEventListener('mousedown', this._onMouseDown);
    window.removeEventListener('mousemove', this._onMouseMove);
    window.removeEventListener('mouseup', this._onMouseUp);
    window.removeEventListener('contextmenu', this._onContextMenu);
    window.removeEventListener('wheel', this._onWheel);
    window.removeEventListener('touchstart', this._onTouchStart);
    window.removeEventListener('touchmove', this._onTouchMove);
    window.removeEventListener('touchend', this._onTouchEnd);
    window.removeEventListener('touchcancel', this._onTouchEnd);
    window.removeEventListener('deviceorientation', this._onDeviceOrientation);
    window.removeEventListener('resize', this._onResize);
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    window.removeEventListener('blur', this._onWindowBlur);

    for (const key in this.sceneObjects) {
      const obj = this.sceneObjects[key];
      if (obj && obj.scene) {
        obj.scene.traverse((child) => {
          if (child.isMesh || child.isPoints) {
            if (child.geometry) {
              child.geometry.dispose();
            }
            if (child.material) {
              const materials = Array.isArray(child.material) ? child.material : [child.material];
              materials.forEach((mat) => {
                if (mat.map) mat.map.dispose();
                if (mat.uniforms) {
                  Object.values(mat.uniforms).forEach((u) => {
                    if (u && u.value && typeof u.value.dispose === 'function') {
                      u.value.dispose();
                    }
                  });
                }
                mat.dispose();
              });
            }
          }
        });
        while (obj.scene.children.length > 0) {
          obj.scene.remove(obj.scene.children[0]);
        }
      }
    }
    this.sceneObjects = {};

    if (this._dustSpriteTexture) {
      this._dustSpriteTexture.dispose();
      this._dustSpriteTexture = null;
    }

    if (this.transitionQuad) {
      if (this.transitionQuad.geometry) this.transitionQuad.geometry.dispose();
      if (this.transitionMaterial) this.transitionMaterial.dispose();
      if (this.transitionScene) {
        this.transitionScene.remove(this.transitionQuad);
      }
      this.transitionQuad = null;
    }

    if (this.rtCurrent) {
      this.rtCurrent.dispose();
      this.rtCurrent = null;
    }
    if (this.rtNext) {
      this.rtNext.dispose();
      this.rtNext = null;
    }

    if (this.renderer) {
      this.renderer.dispose();
      if (this.renderer.forceContextLoss) {
        this.renderer.forceContextLoss();
      }
    }
  }
}
