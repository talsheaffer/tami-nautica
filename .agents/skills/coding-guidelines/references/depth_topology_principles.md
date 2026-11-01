# 2.5D Spatial Depth & Multi-Surface Topology Principles

Canonical mathematical and engineering laws governing every 2.5D spatial scene in `/Users/talsheaffer/code/tami` (`prototype/vision_primitives.py`, `prototype/scene_config.py`, scene builders, and `prototype/js/scene_manager.js`).

Every boundary pixel of a segmented 2.5D object belongs to **one of two physical regimes**:
1. **Contact Boundary ($\partial\Omega_{\text{contact}}$)** — where the object physically touches a supporting/receiving surface (floor footprint, tabletop contact, wall bolt, sofa seat). Here world depth **MUST be $C^0$-continuous** ($\Delta Z_{\text{world}} = 0$).
2. **Free Occlusion Silhouette ($\partial\Omega_{\text{free}}$)** — where the object stands in front of a more distant background across open space. Here world depth **MUST be discontinuous** ($C^{-1}$ jump across separate meshes, with zero rubber-sheet triangle stretching and zero background depth bleed).

---

## 1. Unified World-Space Coordinate Equation

In `prototype/js/scene_manager.js` (`ROOM_DEPTH_VERTEX_SHADER` + `updatePlaneScaleForViewport`), let $(0, y_0, z_0) = (0, \text{boreY}, \text{boreZ})$ be the rest-pose optical center (`boreSightAtRest: true`).

For any depth-displaced mesh $k$ placed at base plane $Z_k$ with displacement amplitude $A_k$, normalized depth texture $d_k(u, v) \in [0, 1]$, and UV perimeter pin margin $p_k$:

$$Z_{\text{world}}^{(k)}(u, v) = Z_k + A_k \cdot d_k(u, v) \cdot \text{pin}_{p_k}(u, v)$$

Because `ROOM_DEPTH_VERTEX_SHADER` scales $(X, Y - y_0)$ by $\text{rayScale}_k = \frac{(z_0 - Z_k) - z_{\text{disp}}^{(k)}}{z_0 - Z_k}$ while `updatePlaneScaleForViewport` scales mesh $k$ by $\text{exRatio}_k = \frac{z_0 - Z_k}{z_0 - Z_{\text{bg}}}$, the product $\text{exRatio}_k \cdot \text{rayScale}_k = \frac{z_0 - Z_{\text{world}}^{(k)}}{z_0 - Z_{\text{bg}}}$ depends **strictly on $Z_{\text{world}}^{(k)}$**—independently of how $(Z_k, A_k)$ is parameterized!

Consequently, two layers $a$ and $b$ remain locked together in 3D from **every** camera viewpoint (translation, dolly, yaw, pitch) at pixel $(x, y)$ **if and only if**:

$$Z_{\text{world}}^{(a)}(x, y) = Z_{\text{world}}^{(b)}(x, y)$$

---

## 2. The Five Mandatory Depth Topology Laws

### Law 1 — Contact Continuity ($C^0$ World-Z Anchoring at Supporting Surfaces)
Whenever an object on `layer1` (`mid`), `layer2` (`fg`), or an `extraLayer` physically touches a receiving surface (floor, steps, wall, ceiling, tabletop, couch):
- **Never place a grounded or wall-attached object on an un-anchored flat plane** whose constant $Z_{\text{plane}}$ differs from the receiving surface's $Z_{\text{world}}^{\text{host}}(x, y)$ at the contact locus (which makes floor objects float/skate, as seen in un-anchored `bene_beitcha`, or makes diagonal elements shear off walls, as seen in a flat `fried_house_railing` handrail).
- **Multi-anchor objects spanning depth** (e.g., a stair handrail bolted to both foreground steps and a recessed house wall, or a table/sofa receding in perspective) **MUST use a depth-displaced mesh** (`depthTexture` + `depthAmplitude`, `pinMarginUv = 0.0`), never a constant-$Z$ billboard.
- **Harmonic Contact-Anchored Depth Solver**: Inside each connected object component $\Omega_i$ with non-empty contact locus $\partial\Omega_{i,\text{contact}}$, compute a smooth harmonic offset field $\delta(x, y)$ ($\nabla^2 \delta = 0$ on $\Omega_i$) satisfying:
  - **Dirichlet condition on $\partial\Omega_{i,\text{contact}}$**: $Z_{\text{world}}^{\text{obj}}(x, y) = Z_{\text{world}}^{\text{host}}(x, y)$ (exact attachment to the floor/wall/table depth at the contact anchor).
  - **Neumann free-boundary condition on $\partial\Omega_{i,\text{free}}$**: $\frac{\partial \delta}{\partial n} = 0$, preserving internal 3D sculptural relief from Depth Anything V2 (`A_internal * (d_DA2 - d_DA2_anchor)`) above/forward of the contact anchor.

### Law 2 — Free-Silhouette Discontinuity ($C^{-1}$ Occlusion Split & Zero Rubber-Sheeting)
Across every non-contacting occlusion edge $\partial\Omega_{\text{free}}$ where a foreground object occludes a background surface across air:
- **Separate Surfaces, Never One Stretched Heightfield**: Foreground occluders with large depth jumps relative to the wall/yard behind them (e.g., concrete pipes in `plan_b_pipe*`, railings, foreground sculptures, chairs, people) MUST live on their own alpha-matted or alpha-cutout layer rather than sharing a single continuous heightfield mesh with the distant background.
- **Outward EDT Depth Extrapolation**: Within every depth-displaced foreground/midground/extra layer, extrapolate the interior object depth $d_{\text{obj}}$ **outward** across transparent pixels (`alpha == 0`) via nearest-neighbor Euclidean Distance Transform (`scipy.ndimage.distance_transform_edt`) and set `pinMarginUv = 0.0`. Never leave transparent pixels at `depth = 0` adjacent to opaque pixels (`depth > 0`), which causes bilinear filtering and straddling mesh triangles to bend/deform object rims toward $Z_k$.

### Law 3 — Ghost-Free Occluded Background Depth (`layer0`)
- `layer0`'s room depth map (`*_depth.png`) represents the **empty room / yard / wall behind foreground cutouts** (`*_bg_inpainted.*`), NOT the foreground cutouts that were removed!
- Whenever foreground/midground objects are segmented out and LaMa-inpainted on `layer0`, `layer0`'s depth map MUST either be estimated on the LaMa-inpainted background plate or harmonically inpainted ($\nabla^2 d_{\text{room}} = 0$ with boundary values from the surrounding room surface outside the dilated cutout mask) so no "ghost depth mounds" warp the floor or wall behind segmented objects.

### Law 4 — Strict Depth Monotonicity & Intentional Suspension
- At every pixel $(x, y)$ where foreground layer $f$ occludes background layer $b$, world Z must satisfy $Z_{\text{world}}^{(f)}(x, y) \ge Z_{\text{world}}^{(b)}(x, y)$ (up to sub-unit numerical tolerance at exact contact loci where $Z_{\text{world}}^{(f)} = Z_{\text{world}}^{(b)}$; render order / depth test resolves coplanar ties cleanly in favor of the attached object).
- **Intentionally suspended/floating artworks** (e.g., hanging paper capsules in `paper_capsules`, floating islands under the Vieux-Port mirror in `marseille_mirror`, floating cardboard Denya house in `denya_model`, floating graphite islands in `wandering_drawing*`) have $\partial\Omega_{\text{contact}} = \emptyset$ (except where suspension threads meet a ceiling/pipe), so they legitimately sit at $Z_{\text{world}}^{\text{obj}} > Z_{\text{world}}^{\text{bg}}$—whereas **grounded people/furniture/walls in those same scenes** (e.g., family standing on the Marseille quai pavement) MUST still obey Law 1 at their ground contact!

### Law 5 — Universal Bore-Sight Registration & Automated Invariant Gate
- Every multi-layer scene MUST enable `boreSightAtRest: true` so rest-pose optical rays align across desktop and mobile viewports without lateral/vertical parallax drift at rest.
- Every scene's contact-anchored layers and silhouette-extrapolated depth maps MUST be declared in SSOT `@dataclass` configuration (`scene_config.py` / scene builder configs -> `scene_manifest.json`) and verified automatically by `tests/test_depth_principles.py` (`|ΔZ_world|` at contact anchors and zero-cliff rim checks on displaced cutout layers).
