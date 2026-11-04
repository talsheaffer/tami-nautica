# 3D Portal Flight & Scene Transition Principles

Canonical mathematical and engineering laws governing every camera flight and scene transition in `/Users/talsheaffer/code/tami` (`prototype/js/scene_manager.js`, `prototype/js/app.js`, `prototype/scene_config.py`, and `prototype/assets/scenes/scene_manifest.json`).

In the spatial archive, moving between spaces is **physical 3D camera flight**, never a decorative 2D video effect. Every transition MUST satisfy five mandatory invariants:

1. **Pure 3D Camera Flight (Zero Post-Process Distortion Effects)**
2. **$C^2$ Camera Kinematics & Exact Boundary Pose Registration ($\Delta\text{Pose}_{t=0} = 0$, $\Delta\text{Pose}_{t=1} = 0$)**
3. **Chromatic & Radiometric Continuity (Smooth Hue Adaptation, Zero Black Void Fade)**
4. **Eager Advance Bundle & GPU Prewarming (Zero Mid-Flight Loading or Shader Stalls)**
5. **Bijective Scene State Isomorphism Across Standalone & Embedded Diorama Views**

---

## 1. The Five Mandatory Flight Transition Laws

### Law 1 — Pure 3D Camera Flight (Zero Post-Process Distortion Effects)
- **No Distortion or Flash Effects**: Runtime scene transitions MUST NOT apply synthetic post-process effects such as `ripple`, `directionalWarp`, UV displacement waves, or `specularFlash` highlights. There is **just flight**.
- **Universal Portal Flight Routing**: Whenever the visitor transitions between two scenes connected by a portal (forward into a target scene or reverse back to its host scene)—whether triggered by clicking the 3D aperture, clicking the return button, clicking breadcrumbs, or selecting the scene in the exhibition navigation chrome—the engine MUST execute real 3D camera flight through that portal rather than falling back to a 2D wipe.
- **Pure Optical Blending When Required**: On non-diorama (`portalMode: 'crossfade'`) portals or non-adjacent multi-hop jumps where an optical blend bridges two scene renders, `TRANSITION_FRAGMENT_SHADER` MUST perform pure linear/smoothstep radiometric interpolation (`mix(colCurrent, colNext, blendFactor)`) paired with continuous 3D camera glide, with zero UV warping and zero additive flash.

---

### Law 2 — $C^2$ Camera Kinematics & Exact Boundary Pose Registration
- **$C^2$ Trajectory**: Flight progress $t \in [0, 1]$ maps through quintic smoothstep
  $$e(t) = 6t^5 - 15t^4 + 10t^3, \qquad e'(0)=e'(1)=0, \quad e''(0)=e''(1)=0$$
  so linear velocity, angular velocity, and acceleration are zero at takeoff ($t = 0$) and landing ($t = 1$).
- **Exact Takeoff ($t = 0$)**: Forward flight begins from the exact live 6-DOF camera pose $(x, y, z, \text{yaw}, \text{pitch}, \text{roll})$ (folding any active gyroscope offset cleanly into pose). Reverse diorama flight maps the visitor's live child-scene pose $(x_c, y_c, z_c, \text{yaw}_c, \text{pitch}_c)$ into the host cubby's coordinate frame via similarity scale $\lambda_p$:
  $$\mathbf{c}_{\text{from}} = \mathbf{c}_{\text{fit}} + \lambda_p R_z(\theta_{\text{roll}})(\mathbf{c}_{\text{child}} - \mathbf{c}_{\text{child,rest}})$$
- **Exact Rotated-Frustum Landing (`effectiveFitZ` at $t = 1$)**: For a portal rectangle of world size $(W, H)$ on $Z = Z_p$ rotated by $\theta_p$ (`rotationDeg`), and target arrival orientation $R_{\text{cam}} = R_{\text{YXZ}}(\text{pitch}_{\text{rest}}, \text{yaw}_{\text{rest}}, \text{roll}_{\text{flight}})$ (including mobile `portraitInitialYawDeg`), let $\mathbf{r}_c^{\text{world}} = R_{\text{cam}}(\pm\tan(\tfrac{\text{fov}}{2})\cdot\text{aspect},\; \pm\tan(\tfrac{\text{fov}}{2}),\; -1)$ be the four camera frustum corner rays and $(u_c, v_c)$ their slopes in the portal's local axes per unit $-\Delta Z$. The arrival distance MUST satisfy:
  $$d_{\text{eff}} = \eta_{\text{fit}} \cdot \min_{c \in \{1..4\}} \min\left(\frac{W / 2}{|u_c|},\; \frac{H / 2}{|v_c|}\right), \qquad \eta_{\text{fit}} \in (0.99, 1.0]$$
  so that at $t = 1$ ($\mathbf{c}_{\text{fit}} = (X_p, Y_p, Z_p + d_{\text{eff}})$), **100% of the viewport frustum lies strictly inside the portal aperture**.
- **Scale-Invariant Diorama Registration**: Embedding `dioramaGroup` with $\lambda_p = \frac{d_{\text{eff}} + D_{\text{recess}}}{z_{\text{rest}} - z_{\text{front}}}$ and $\mathbf{p}_{\text{dio}} = \mathbf{c}_{\text{fit}} - \lambda_p R_z(\theta_{\text{roll}})\mathbf{c}_{\text{rest}}$ cancels $\lambda_p$ identically under perspective projection at $\mathbf{c}_{\text{fit}}$, guaranteeing sub-pixel 1:1 registration with the standalone target scene at $t = 1$.
- **Continuous Frustum Clipping (Zero Mid-Flight Unclip Snaps)**: All 4 portal frustum clipping planes MUST remain continuously active across the entire $t \in [0, 1]$ trajectory. Toggling clipping planes off at arbitrary mid-flight thresholds (such as $t = 0.88$ forward or $t = 0.16$ reverse) is strictly forbidden because it causes a 1-frame peripheral pop whenever $d(t) > d_{\text{eff}}$.
- **Zero Post-Landing Zoom Jumps on Crossfade Portals**: Direct (`!embedDiorama`) portal flights MUST NOT inject artificial `transitionGlideInOffsetZ` dolly offsets onto the target scene during optical cross-dissolve; both views glide continuously without scale jumps.

---

### Law 3 — Chromatic & Radiometric Continuity (Smooth Hue Adaptation)
- **Zero Fade-to-Black on Host Architecture**: Opaque host room/wall plates (`hostObj.layer0`) MUST remain at `uFadeToClear = 0.0` throughout every portal flight—never darkened toward clear color (`0x121314`). Surrounding walls and adjacent portal cubbies glide naturally off the viewport edges via true 3D perspective motion.
- **Proximity-Driven Rim Dissolve**: Only foreground/midground rim elements sitting in front of the aperture (`layer1`, `layer2`, `extraLayers`, `frameLine`, host `layer3` dust) dissolve smoothly as the physical aperture viewport-fill ratio
  $$\rho(Z_{\text{cam}}) = \frac{d_{\text{eff}}}{\max(d_{\text{eff}},\; Z_{\text{cam}} - Z_p)} \in (0, 1]$$
  approaches $1.0$ ($\rho \in [\rho_{\text{rimStart}}, 1.0]$), making forward and reverse rim dissolves 100% time-symmetric in 3D space.
- **Smooth $C^2$ Hue / White-Balance Adaptation (`uTint`)**: When an embedded portal cubby sits inside a host lighting environment with a distinct chromatic cast (e.g., warm cream tungsten shelf lighting vs. cool neutral museum daylight inside `bene_beitcha_scroll_foot`), `dioramaGroup` materials apply a multiplicative `uTint` (`dioramaRestTintRgb`, all channels $\in (0, 1]$ so highlights never clip) that interpolates smoothly via $e(t)$ between `dioramaRestTintRgb` (at cubby rest) and neutral `[1.0, 1.0, 1.0]` (at target scene landing), ensuring $\Delta\text{RGB}_{t=0} = 0$ and $\Delta\text{RGB}_{t=1} = 0$ in both directions.

---

### Law 4 — Eager Advance Bundle & GPU Prewarming
- **Everything Loaded & Prewarmed Before Takeoff**: All scene layer textures (`bg`, `depth`, `mid`, `midDepth`, `fg`, `fgDepth`, `reflection`, `extraLayers` + depths), portal `occluder` + `occluderDepth` textures, portal `veil` textures, child portal textures, and outgoing portal target scenes (both `embedDiorama: true` and `embedDiorama: false`) MUST be decoded in advance via URL-shared `THREE.Texture` instances, uploaded to GPU memory (`renderer.initTexture`), and shader-compiled (`renderer.compile`) for both clipped and unclipped states before flight begins.
- **Idempotent Manifest Sync**: Background `scene_manifest.json` synchronization (`updatePortalShelves`) MUST compare canonical structural portal definitions (excluding viewport-dependent derived coordinates `worldX`, `worldY`, `width`, `height`, `fitZ`, `effectiveFitZ`) so active portal shelves are never disposed and rebuilt mid-session.

---

### Law 5 — Bijective Scene State Isomorphism
Every mutable visual or interactive state of a scene $S$ MUST be represented identically in standalone `sceneObjects[S]` and in every `dioramaGroup` embedding $S$:
- **Rigid-Body Transform Completeness (`companionFall` & Dynamic Layers)**: Synchronizing `extraMeshes` inside `dioramaGroup` MUST copy **all** rigid-body and material degrees of freedom from `sourceExtra.mesh`: `position`, **`rotation` (3D Euler/quaternion)**, `scale`, `visible`, `uDistBg`, `uOpticalCenterLocalX`, `uOpticalCenterLocalY`, `uClipUvMinX`, `uClipUvMaxX`, and `uOpacity`. Never sync `position` without `rotation` (which previously caused `scroll_black_rod` in `bene_beitcha_scroll_foot` to snap vertical at $0^\circ$ while sitting at its dropped $-95^\circ$ position!).
- **Child Portal Depth Relief & Continuous Animation State**: Child portal occluders with depth maps (such as the sculpted ceramic foot `scene17_scroll_foot_occluder` in `bene_beitcha_scroll_foot`) MUST use the exact same `ROOM_DEPTH_VERTEX_SHADER` + `depthTexture` + `depthAmplitude` displacement inside `dioramaGroup` as in the standalone scene, synchronized continuously with `occluderAnimFactor` (not binary `isErased` booleans).
- **Strict Two-Pass Transform Ordering & Per-Tick Live Sync**:
  1. **Pass 1 (Host State)**: Update all scenes' own layer geometries, `carpetUnrollRig` UV clips/translations, and portal `occluderMesh` / `companionFall` / `veilMesh` transforms first.
  2. **Pass 2 (Diorama Mirror Sync)**: Mirror those updated host states into every `dioramaGroup` (`_syncDioramaMirrorsForScene`), and invoke `_syncDioramaMirrorsForScene` on every live animation step (`_applyOccluderInstanceTransform`, `_applyCarpetUnrollTransform`).
- **Active-Ancestor Occluder Lock Suppression**: Auto-regeneration timers (`_scheduleOccluderRegen`) MUST NEVER re-lock a portal while the visitor is currently inside its target scene or any descendant scene in `_portalOriginStack`; timers defer cleanly until after the visitor returns to the host scene.

---

## 2. Automated Verification Gate
Every change to `prototype/js/scene_manager.js`, `prototype/js/app.js`, or portal configurations MUST pass `python3 -m unittest tests/test_flight_principles.py` (verifying zero shader distortion uniforms/code, continuous clipping & zero black wall fade across $t \in [0, 1]$, smooth `uTint` continuity, advance bundle prewarming, and 1:1 `position` + `rotation` state isomorphism across forward and reverse flights).
