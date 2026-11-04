// prototype/js/scene_manager.js
// Three.js 2.5D Layered Depth Image (LDI) + Outpainted 6-DOF Spatial Gallery Renderer
// Honoring Tamar "Tami" Sheaffer's 70th Birthday
// Enforces homographic inner/expanded layer registration, Depth Anything V2 3D room relief,
// decoupled 6-DOF camera navigation (yaw/pitch + forward/backward dolly + X/Y strafe),
// and analytical ray-frustum clamping so zero image borders are ever exposed.

import * as THREE from './vendor/three.module.js';

export const EXHIBITION_GROUPS = Object.freeze([
  Object.freeze({
    id: 'portal_hub',
    titleHe: 'שער התערוכות (קיר המדפים)',
    shortTabHe: 'קיר המדפים',
    sceneIds: Object.freeze(['shelf_portal'])
  }),
  Object.freeze({
    id: 'binyamin_plan_b',
    titleHe: 'גלריה בנימין · אולי שם יהיה יותר טוב / Plan B',
    shortTabHe: 'גלריה בנימין · Plan B',
    sceneIds: Object.freeze([
      'paper_capsules',
      'tzofen_brick_table',
      'plan_b_pipe_v3'
    ])
  }),
  Object.freeze({
    id: 'bene_beitcha_home',
    titleHe: 'בנה ביתך ודגם דניה · בית וארכיטקטורה',
    shortTabHe: 'בנה ביתך ודניה',
    sceneIds: Object.freeze([
      'bene_beitcha',
      'bene_beitcha_scroll_foot',
      'purim_costume_1966',
      'vitrine_keys_pen',
      'humpty_dumpty_booth',
      'denya_model',
      'denya_living_room',
      'denya_couch_guitar'
    ])
  }),
  Object.freeze({
    id: 'wandering_drawings',
    titleHe: 'שיטוט ברישום · רישומים על נייר',
    shortTabHe: 'שיטוט ברישום',
    sceneIds: Object.freeze([
      'wandering_drawing_1',
      'scroll_palm_world',
      'wandering_drawing'
    ])
  }),
  Object.freeze({
    id: 'family_places',
    titleHe: 'בתים, ילדות ומרסיי · משפחה וזיכרון',
    shortTabHe: 'בתים, ילדות ומרסיי',
    sceneIds: Object.freeze([
      'fried_yard_installation',
      'fried_house_railing',
      'fried_table_hybrid',
      'herzliya_lawn_carpet',
      'herzliya_tree_canopy',
      'bar_giora_bedroom',
      'paris_cite_white_table',
      'toddler_tami_1958',
      'marseille_mirror'
    ])
  }),
  Object.freeze({
    id: 'archive_versions',
    titleHe: 'ארכיון גרסאות',
    shortTabHe: 'ארכיון',
    sceneIds: Object.freeze([
      'tzofen_brick_table_v2',
      'plan_b_pipe',
      'humpty_dumpty_bundle',
      'plan_b_pipe_v1',
      'marseille_mirror_v1'
    ])
  })
]);

const DEFAULT_SHELF_PORTAL_SHELVES = Object.freeze([
  Object.freeze({
    id: 'cubby_scroll_foot',
    targetSceneId: 'bene_beitcha_scroll_foot',
    labelHe: 'מגילה וכף רגל · ׳בנה ביתך׳',
    innerBoxPx: Object.freeze([459.73, 56.52, 551.16, 161.88]),
    worldZ: -284.0,
    recessDepth: 160.0,
    occluder: Object.freeze({
      texture: 'assets/scenes/scene9_occluder_cubby_wandering_1.png',
      innerBoxPx: Object.freeze([462.0, 105.46, 548.89, 159.61]),
      labelHe: 'שברי פורצלן ואבן כורכר',
      coverageRatio: 0.3337,
      locksPortal: true
    })
  }),
  Object.freeze({
    id: 'cubby_capsules',
    targetSceneId: 'paper_capsules',
    labelHe: 'קפסולות נייר · אולי שם יהיה יותר טוב',
    innerBoxPx: Object.freeze([763.55, 55.37, 857.86, 161.88]),
    worldZ: -284.0,
    recessDepth: 160.0,
    occluder: Object.freeze({
      texture: 'assets/scenes/scene9_occluder_cubby_capsules.png',
      innerBoxPx: Object.freeze([765.82, 113.52, 855.59, 159.61]),
      labelHe: 'לוח בטון ופקעת חוטים צבעונית',
      coverageRatio: 0.2524,
      locksPortal: true
    })
  }),
  Object.freeze({
    id: 'cubby_wandering_2',
    targetSceneId: 'wandering_drawing',
    labelHe: 'שיטוט 2 · איים מרחפים',
    innerBoxPx: Object.freeze([531.08, 168.86, 625.39, 267.88]),
    worldZ: -284.0,
    recessDepth: 160.0,
    occluder: Object.freeze({
      texture: 'assets/scenes/scene9_occluder_cubby_wandering_2.png',
      innerBoxPx: Object.freeze([535.65, 228.74, 623.12, 263.88]),
      labelHe: 'קפלי פורצלן לבנים',
      coverageRatio: 0.2552,
      locksPortal: true
    })
  }),
  Object.freeze({
    id: 'cubby_bene_beitcha',
    targetSceneId: 'bene_beitcha',
    labelHe: 'בנה ביתך · מוזיאון רמת גן',
    innerBoxPx: Object.freeze([835.48, 168.28, 929.79, 267.88]),
    worldZ: -284.0,
    recessDepth: 160.0,
    occluder: Object.freeze({
      texture: 'assets/scenes/scene9_occluder_cubby_bene_beitcha.png',
      innerBoxPx: Object.freeze([837.75, 225.28, 918.89, 262.15]),
      labelHe: 'אבני חול וכורכר חמות',
      coverageRatio: 0.277,
      locksPortal: true
    })
  }),
  Object.freeze({
    id: 'cubby_fried_house',
    targetSceneId: 'fried_house_railing',
    labelHe: 'בית פריד · כרית ואבן על הרמפה',
    innerBoxPx: Object.freeze([611.06, 275.43, 703.65, 367.54]),
    worldZ: -284.0,
    recessDepth: 160.0,
    occluder: Object.freeze({
      texture: 'assets/scenes/scene9_occluder_cubby_fried_house.png',
      innerBoxPx: Object.freeze([620.24, 307.66, 697.35, 364.7]),
      labelHe: 'אבן זהובה מונחת על חלוק אבן',
      coverageRatio: 0.3571,
      locksPortal: true
    })
  }),
  Object.freeze({
    id: 'cubby_denya_model',
    targetSceneId: 'denya_model',
    labelHe: 'דגם בית דניה · קרטון ועץ',
    innerBoxPx: Object.freeze([762.4, 274.86, 857.86, 367.54]),
    worldZ: -284.0,
    recessDepth: 160.0,
    occluder: Object.freeze({
      texture: 'assets/scenes/scene9_occluder_cubby_denya_model.png',
      innerBoxPx: Object.freeze([762.95, 275.4, 857.32, 364.7]),
      labelHe: 'שמיכת לבד אפורה מקופלת',
      coverageRatio: 1.0,
      locksPortal: true
    })
  }),
  Object.freeze({
    id: 'cubby_bar_giora',
    targetSceneId: 'bar_giora_bedroom',
    labelHe: 'חדר שינה בבר גיורא · ויטרינה ומנורת מפרש',
    innerBoxPx: Object.freeze([838.36, 431.55, 930.94, 521.36]),
    worldZ: -284.0,
    recessDepth: 160.0
  }),
  Object.freeze({
    id: 'cubby_plan_b_v3',
    targetSceneId: 'plan_b_pipe_v3',
    labelHe: 'Plan B · תמי ויערה באור הדיפוזי',
    innerBoxPx: Object.freeze([716.94, 532.37, 902.75, 642.91]),
    worldZ: -284.0,
    recessDepth: 160.0,
    occluder: Object.freeze({
      texture: 'assets/scenes/scene9_occluder_cubby_plan_b_v3.png',
      innerBoxPx: Object.freeze([745.68, 583.61, 869.98, 642.95]),
      labelHe: 'פרח לוטוס ורוד ואריג מפוספס',
      coverageRatio: 0.2704,
      locksPortal: true
    })
  }),
  Object.freeze({
    id: 'cubby_marseille',
    targetSceneId: 'marseille_mirror',
    labelHe: 'מרסיי · מתחת לתקרת המראה',
    innerBoxPx: Object.freeze([716.94, 721.33, 902.75, 836.48]),
    worldZ: -284.0,
    recessDepth: 160.0,
    occluder: Object.freeze({
      texture: 'assets/scenes/scene9_occluder_cubby_marseille.png',
      innerBoxPx: Object.freeze([752.01, 723.02, 862.5, 824.99]),
      labelHe: 'כלי טרקוטה ולוחות בטון על לבד אדום',
      coverageRatio: 0.4869,
      locksPortal: true
    })
  }),
  Object.freeze({
    id: 'cubby_paris_table',
    targetSceneId: 'paris_cite_white_table',
    labelHe: 'פאריס · שולחן לבן בסיטה',
    innerBoxPx: Object.freeze([612.21, 732.85, 704.8, 836.48]),
    worldZ: -284.0,
    recessDepth: 160.0
  })
]);

const DEFAULT_YARD_PORTAL_SHELVES = Object.freeze([
  Object.freeze({
    id: 'yard_railing_portal',
    targetSceneId: 'fried_house_railing',
    labelHe: 'תמי שפר ואראלה סדן על המעקה הירוק בבית פריד',
    shortBadgeHe: 'תמי ואראלה במעקה',
    showMicroBadge: false,
    innerBoxPx: Object.freeze([515.0, 468.0, 705.0, 788.0]),
    worldZ: -306.04,
    recessDepth: 0.0,
    embedDiorama: false,
    showFrame: false,
    occluder: Object.freeze({
      texture: 'assets/scenes/scene_railing_occluder_installation.png',
      depthTexture: 'assets/scenes/scene_railing_occluder_installation_depth.png',
      depthAmplitude: 64.0,
      pinMarginUv: 0.0,
      segmentsX: 192,
      segmentsY: 128,
      worldZ: -302.0,
      renderOrder: 6,
      fullInnerCanvas: true,
      alphaHitThreshold: 24,
      innerBoxPx: Object.freeze([436.0, 94.0, 792.0, 930.0]),
      labelHe: 'המיצב ׳מרחב לא מוגן 7׳',
      coverageRatio: 0.9809,
      locksPortal: true
    })
  })
]);

const DEFAULT_DENYA_LIVING_PORTAL_SHELVES = Object.freeze([
  Object.freeze({
    id: 'portal_denya_dining_wandering_2',
    targetSceneId: 'wandering_drawing',
    portalMode: 'aperture',
    labelHe: '׳שיטוט 2׳ על קיר הסלון · איים מרחפים',
    shortBadgeHe: 'שיטוט 2',
    innerBoxPx: Object.freeze([626.26, 221.46, 685.93, 258.65]),
    worldZ: -379.39,
    recessDepth: 96.0,
    embedDiorama: true,
    showFrame: true,
    restFrameOpacity: 0.0,
    frameRenderOrder: 6.5,
    badgeOffsetYFactor: -0.68
  }),
  Object.freeze({
    id: 'portal_denya_to_couch',
    targetSceneId: 'denya_couch_guitar',
    portalMode: 'crossfade',
    labelHe: 'ספת הסלון בדניה · טל, יערה וכרמל עם הגיטרה',
    shortBadgeHe: 'לספה והגיטרה',
    innerBoxPx: Object.freeze([435.0, 266.0, 815.0, 422.0]),
    worldZ: -378.4,
    recessDepth: 0.0,
    embedDiorama: false,
    showFrame: false,
    restFrameOpacity: 0.0,
    frameRenderOrder: 7.5,
    fadeStartProgress: 0.24,
    badgeOffsetYFactor: -0.35
  })
]);

const DEFAULT_DENYA_COUCH_PORTAL_SHELVES = Object.freeze([
  Object.freeze({
    id: 'portal_denya_couch_wandering_2',
    targetSceneId: 'wandering_drawing',
    portalMode: 'aperture',
    labelHe: '׳שיטוט 2׳ על קיר הסלון · איים מרחפים',
    shortBadgeHe: 'שיטוט 2',
    innerBoxPx: Object.freeze([623.0, 163.0, 749.0, 237.0]),
    worldZ: -341.6,
    recessDepth: 96.0,
    embedDiorama: true,
    showFrame: true,
    restFrameOpacity: 0.0,
    frameRenderOrder: 6.5,
    badgeOffsetYFactor: -0.68
  }),
  Object.freeze({
    id: 'portal_denya_back_to_dining',
    targetSceneId: 'denya_living_room',
    portalMode: 'reverse_transition',
    labelHe: 'מבט משולחן האוכל · תמי, גדי והמשפחה',
    shortBadgeHe: 'לשולחן האוכל',
    innerBoxPx: Object.freeze([540.0, 580.0, 760.0, 648.0]),
    worldZ: -281.8,
    recessDepth: 0.0,
    embedDiorama: false,
    showFrame: false,
    restFrameOpacity: 0.0,
    frameRenderOrder: 9.5,
    badgeOffsetYFactor: 0.0
  })
]);

const DEFAULT_FRIED_HOUSE_PORTAL_SHELVES = Object.freeze([
  Object.freeze({
    id: 'fried_house_window_wandering_1',
    targetSceneId: 'wandering_drawing_1',
    portalMode: 'aperture',
    labelHe: 'חלון העץ בבית פריד · שער אל ׳שיטוט ברישום 1׳',
    shortBadgeHe: 'שיטוט 1 בחלון',
    innerBoxPx: Object.freeze([-203.0, 219.0, 125.0, 446.0]),
    worldZ: -221.06,
    recessDepth: 118.0,
    embedDiorama: true,
    showFrame: true,
    restFrameOpacity: 0.0,
    frameRenderOrder: 10.7,
    occluder: Object.freeze({
      texture: 'assets/scenes/scene10_fried_house_window_shutter_crop.png',
      innerBoxPx: Object.freeze([-226.0, 181.0, 142.0, 459.0]),
      worldZ: -220.71,
      renderOrder: 10.8,
      fullInnerCanvas: false,
      alphaHitThreshold: 24,
      labelHe: 'תריס העץ החום בחלון בית פריד',
      coverageRatio: 0.99,
      locksPortal: true
    })
  })
]);

const DEFAULT_WANDERING1_PORTAL_SHELVES = Object.freeze([
  Object.freeze({
    id: 'portal_wandering1_scroll_world',
    targetSceneId: 'scroll_palm_world',
    portalMode: 'aperture',
    labelHe: 'עולם הדקל והמוטות בתוך המגילה · ׳מגילה וכף רגל׳',
    shortBadgeHe: 'לתוך המגילה',
    innerBoxPx: Object.freeze([1070.0, 408.0, 1258.0, 734.0]),
    worldZ: -80.0,
    recessDepth: 135.0,
    embedDiorama: true,
    showFrame: false,
    restFrameOpacity: 0.0,
    frameRenderOrder: 6.2,
    veil: Object.freeze({
      texture: 'assets/scenes/scene7_wandering1_scroll_veil.png',
      worldZ: -80.0,
      renderOrder: 6.5,
      fullInnerCanvas: true,
      restOpacity: 1.0,
      hoverOpacity: 0.45,
      dissolveEndProgress: 0.58
    })
  })
]);

const DEFAULT_TZOFEN_PORTAL_SHELVES = Object.freeze([
  Object.freeze({
    id: 'portal_tzofen_tree_window',
    targetSceneId: 'herzliya_tree_canopy',
    portalMode: 'aperture',
    labelHe: 'מסגרת החלון הלבנה ב׳צופן׳ · שער אל צמרת העץ בהרצליה',
    shortBadgeHe: 'חלון אל העץ',
    innerBoxPx: Object.freeze([975.0, 146.0, 1126.0, 250.0]),
    worldZ: -277.01,
    recessDepth: 115.0,
    embedDiorama: true,
    showFrame: false,
    restFrameOpacity: 0.0,
    frameRenderOrder: 5.8
  })
]);

const DEFAULT_TZOFEN_V2_PORTAL_SHELVES = Object.freeze([
  Object.freeze({
    id: 'portal_tzofen_v2_tree_window',
    targetSceneId: 'herzliya_tree_canopy',
    portalMode: 'aperture',
    labelHe: 'מסגרת החלון הלבנה ב׳צופן׳ (ארכיון) · שער אל צמרת העץ בהרצליה',
    shortBadgeHe: 'חלון אל העץ',
    innerBoxPx: Object.freeze([975.0, 146.0, 1126.0, 250.0]),
    worldZ: -253.09,
    recessDepth: 115.0,
    embedDiorama: true,
    showFrame: false,
    restFrameOpacity: 0.0,
    frameRenderOrder: 5.8
  })
]);

const DEFAULT_SCROLL_FOOT_PORTAL_SHELVES = Object.freeze([
  Object.freeze({
    id: 'portal_scroll_foot_purim_1966',
    targetSceneId: 'purim_costume_1966',
    labelHe: 'תמי ועדה מחופשות בפורים (~1966) · לחצו על התצלום להיכנס לגינה',
    shortBadgeHe: 'אל פורים 1966',
    showMicroBadge: false,
    innerBoxPx: Object.freeze([706.0, 774.0, 883.0, 939.0]),
    worldZ: -274.29,
    recessDepth: 0.0,
    embedDiorama: true,
    keepHostBackdropOpaque: true,
    outerWallFadeRangeT: Object.freeze([0.78, 0.96]),
    showFrame: false,
    restFrameOpacity: 0.0,
    frameRenderOrder: 6.8,
    // Purim print lies tilted on the scroll floor: embed Scene 18 on its real surface (flat at
    // rest, full relief at landing) so the camera swerves to meet it head-on (pure 3D flight).
    wallMount: Object.freeze({
      hostLayerId: 'pinned_purim_photo',
      pointPairsInnerPx: Object.freeze([
        [[346.8, 100.0], [737.282, 833.823]],
        [[750.0, 100.0], [777.323, 823.427]],
        [[1153.2, 100.0], [817.089, 813.101]],
        [[346.8, 500.0], [750.763, 867.543]],
        [[750.0, 500.0], [791.562, 856.819]],
        [[1153.2, 500.0], [832.076, 846.169]],
        [[346.8, 900.0], [764.799, 902.652]],
        [[750.0, 900.0], [806.385, 891.582]],
        [[1153.2, 900.0], [847.676, 880.591]]
      ]),
      hostWallClearanceWorld: 0.08,
      flatAtRest: true,
      cavityInnerBoxPx: Object.freeze([246.0, 0.0, 1254.0, 1000.0]),
      cavityFeatherPx: 4.0,
      dioramaRenderOrders: Object.freeze({ bg: 6.55, mid: 6.6, fg: 6.65, dust: 6.7 })
    }),
    occluder: Object.freeze({
      texture: 'assets/scenes/scene17_scroll_foot_occluder.png',
      depthTexture: 'assets/scenes/scene17_scroll_foot_occluder_depth.png',
      worldZ: -350.0,
      depthAmplitude: 165.0,
      pinMarginUv: 0.0,
      segmentsX: 192,
      segmentsY: 128,
      renderOrder: 7.8,
      fullInnerCanvas: true,
      alphaHitThreshold: 24,
      innerBoxPx: Object.freeze([794.0, 757.0, 845.0, 818.0]),
      labelHe: 'כף הרגל המפוסלת המצמידה את פינת התצלום',
      coverageRatio: 0.6416,
      locksPortal: true,
      companionFall: Object.freeze({
        extraLayerId: 'scroll_black_rod',
        pivotInnerPx: Object.freeze([836.8, 791.0]),
        pivotWorldZ: -293.706,
        fallAngleDeg: -95.0,
        shiftWorldX: 0.0,
        shiftWorldY: 0.0
      })
    })
  })
]);

const DEFAULT_BAR_GIORA_PORTAL_SHELVES = Object.freeze([
  Object.freeze({
    id: 'portal_bar_giora_vitrine',
    targetSceneId: 'vitrine_keys_pen',
    labelHe: 'תיבת הוויטרינה מ׳בנה ביתך׳ · לחצו להתקרבות וסיבוב אל תוך הוויטרינה',
    shortBadgeHe: 'אל הוויטרינה',
    showMicroBadge: false,
    innerBoxPx: Object.freeze([496.0, 365.0, 614.0, 483.0]),
    worldZ: -319.01,
    recessDepth: 0.0,
    embedDiorama: true,
    keepHostBackdropOpaque: true,
    outerWallFadeRangeT: Object.freeze([0.78, 0.96]),
    showFrame: false,
    restFrameOpacity: 0.0,
    frameRenderOrder: 5.8,
    rotationDeg: -130.0,
    flightRollDeg: -130.0,
    wallMount: Object.freeze({
      hostCenterInnerPx: Object.freeze([555.0, 424.0]),
      hostBoxSidePx: 118.0,
      hostWallZ: -319.01,
      hostWallSlopeDzDx: -0.5944,
      hostWallSlopeDzDy: 0.00154,
      hostWallClearanceWorld: 0.08,
      hostRefInnerScale: 2.12,
      targetBoxCenterInnerPx: Object.freeze([749.5, 504.5]),
      targetBoxSidePx: 864.0,
      targetPlasterDepthNorm: 0.203922,
      cavityInnerBoxPx: Object.freeze([352.0, 120.0, 1148.0, 895.0]),
      cavityFeatherPx: 4.0,
      inPlaneRollDeg: -130.0,
      dioramaRenderOrders: Object.freeze({ bg: 2.0, mid: 3.0, fg: 4.0, dust: 4.5 })
    })
  })
]);

const DEFAULT_PARIS_TABLE_PORTAL_SHELVES = Object.freeze([
  Object.freeze({
    id: 'portal_paris_table_tami_1958',
    targetSceneId: 'toddler_tami_1958',
    portalMode: 'crossfade',
    labelHe: 'תמי הפעוטה (1958) · לחצו על התצלום שעל השולחן הלבן להתקרבות',
    shortBadgeHe: 'אל תמי 1958',
    showMicroBadge: false,
    innerBoxPx: Object.freeze([420.0, 353.0, 667.0, 390.0]),
    worldZ: -312.27,
    recessDepth: 0.0,
    embedDiorama: false,
    showFrame: false,
    restFrameOpacity: 0.0,
    frameRenderOrder: 6.8
  })
]);

export const SCENE_GEOMETRY_SPEC = Object.freeze({
  baseCameraZ: 400,
  minCameraZ: 120,
  maxCameraZ: 685,
  cameraNearPlane: 1,
  cameraFarPlane: 6000,
  bgZ: -350,
  midZ: -150,
  fgZ: 0,
  reflectionZ: -340,
  particlesRenderOrder: 18,
  nearPassFadeStartDist: 8.0,
  nearPassFadeFullDist: 58.0,
  tunnelArrivalClearanceZ: 62.0,
  tunnelMinSpanZ: -80.0,
  bgBaseWidth: 750,
  bgBaseHeight: 500,
  bgExpansionFactor: 1.80,
  bgSegmentsX: 160,
  bgSegmentsY: 108,
  midBaseWidth: 600,
  midBaseHeight: 400,
  fgBaseWidth: 500,
  fgBaseHeight: 333.3333,
  innerFrameWidthPx: 1500,
  innerFrameHeightPx: 1000,
  roomDepthReliefAmplitude: 92.0,
  roomPinMarginUv: 0.045,
  depthEdgePinMarginUv: 0.015,
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
  keyElevationSpeedPerSec: 180,
  hoverAmbientShiftX: 14,
  hoverAmbientShiftY: 8,
  hoverAmbientYawDeg: 2.2,
  hoverAmbientPitchDeg: 1.3,
  pointerExcursionShiftX: 65,
  pointerExcursionShiftY: 34,
  pointerExcursionYawDeg: 9.0,
  pointerExcursionPitchDeg: 5.0,
  dampingReferenceHz: 60.0,
  dampingMaxDtSec: 0.25,
  poseDragDamping60Hz: 0.22,
  poseIdleDamping60Hz: 0.11,
  pointerDamping60Hz: 0.09,
  hoverDamping60Hz: 0.07,
  gyroGain: 1.25,
  gyroMaxYawDeg: 24,
  gyroMaxPitchDeg: 18,
  gyroSmoothingRate: 14.0,
  gyroRecenterThresholdDeg: 26.0,
  gyroParallaxShiftX: 24.0,
  gyroParallaxShiftY: 15.0,
  gyroHintDismissYawDeg: 1.8,
  gyroHintDismissPitchDeg: 1.5,
  gyroNoEventsTimeoutMs: 1500,
  transitionDurationMs: 820,
  transitionGlideOutDollyZ: 85.0,
  transitionGlideInOffsetZ: 95.0,
  portalDefaultWorldZ: -8.0,
  portalDefaultRecessDepth: 160.0,
  portalOccluderWorldZ: 0.5,
  portalOccluderLiftZ: 12.0,
  portalOccluderLiftScale: 1.06,
  portalOccluderDissolveMs: 420,
  portalOccluderRegenDelayMs: 60000,
  portalFlightDurationMs: 1150,
  portalFadeStartProgress: 0.75,
  portalApertureFitSafetyRatio: 0.996,
  portalRimDissolveFillStart: 0.82,
  shelfCubbyRestTintRgb: Object.freeze([1.0, 0.978, 0.952]),
  neutralTintRgb: Object.freeze([1.0, 1.0, 1.0]),
  backgroundWarmInitialDelayMs: 50,
  backgroundWarmStepDelayMs: 35,
  occluderActiveVisitorDeferMs: 5000,
  portalFrameBevelOffsetZ: -0.6,
  portalBaseRenderOrder: 2,
  shelfFgRenderOrder: 15,
  portalDioramaSegmentsX: 48,
  portalDioramaSegmentsY: 32,
  portalDustRenderOrder: 13.5,
  portalOccluderDefaultAlphaHit: 24,
  touchSyntheticMouseGuardMs: 2500,
  directPortalFlightDurationMs: 1380,
  directPortalFitScale: 1.15,
  directPortalMinFitDistance: 76.0,
  directPortalBlendStartT: 0.66,
  directPortalReverseBlendEndT: 0.34,
  tunnelMinStrafeScale: 0.42,
  clearColorHex: 0x121314,
  shelfPortalShelves: DEFAULT_SHELF_PORTAL_SHELVES,
  yardPortalShelves: DEFAULT_YARD_PORTAL_SHELVES,
  denyaLivingPortalShelves: DEFAULT_DENYA_LIVING_PORTAL_SHELVES,
  denyaCouchPortalShelves: DEFAULT_DENYA_COUCH_PORTAL_SHELVES,
  friedHousePortalShelves: DEFAULT_FRIED_HOUSE_PORTAL_SHELVES,
  wandering1PortalShelves: DEFAULT_WANDERING1_PORTAL_SHELVES,
  tzofenPortalShelves: DEFAULT_TZOFEN_PORTAL_SHELVES,
  tzofenV2PortalShelves: DEFAULT_TZOFEN_V2_PORTAL_SHELVES,
  scrollFootPortalShelves: DEFAULT_SCROLL_FOOT_PORTAL_SHELVES,
  barGioraPortalShelves: DEFAULT_BAR_GIORA_PORTAL_SHELVES,
  parisTablePortalShelves: DEFAULT_PARIS_TABLE_PORTAL_SHELVES
});

/**
 * Desktop keyboard navigation bindings keyed by layout-independent KeyboardEvent.code
 * (so Hebrew / non-QWERTY layouts work identically). Elevation (up/down) is clamped by each
 * scene's maxStrafeY + the analytical frustum guard in _clampPose.
 */
export const KEYBOARD_NAV_BINDINGS = Object.freeze({
  forward: Object.freeze(['KeyW', 'ArrowUp']),
  backward: Object.freeze(['KeyS', 'ArrowDown']),
  right: Object.freeze(['KeyD', 'ArrowRight']),
  left: Object.freeze(['KeyA', 'ArrowLeft']),
  up: Object.freeze(['KeyE', 'PageUp']),
  down: Object.freeze(['KeyQ', 'PageDown'])
});
const KEYBOARD_NAV_CODES = new Set(Object.values(KEYBOARD_NAV_BINDINGS).flat());

const Z_AXIS_UNIT = new THREE.Vector3(0, 0, 1);
const Q_DEVICE_MINUS_HALF_PI_X = new THREE.Quaternion(-Math.sqrt(0.5), 0, 0, Math.sqrt(0.5));

/**
 * Returns the current screen orientation angle in degrees (0, 90, -90, 180)
 * using standard ScreenOrientation API with fallback for legacy iOS Safari.
 */
export function getScreenOrientationAngleDeg() {
  if (
    typeof window !== 'undefined' &&
    window.screen &&
    window.screen.orientation &&
    typeof window.screen.orientation.angle === 'number' &&
    Number.isFinite(window.screen.orientation.angle)
  ) {
    return window.screen.orientation.angle;
  }
  if (
    typeof window !== 'undefined' &&
    typeof window.orientation === 'number' &&
    Number.isFinite(window.orientation)
  ) {
    return window.orientation;
  }
  return 0;
}

/**
 * Converts W3C DeviceOrientation intrinsic Z-X'-Y'' Euler angles (alpha, beta, gamma)
 * plus screen orientation angle (0, 90, -90, 180) into a unit 3D camera quaternion
 * free of gimbal-lock discontinuities across upright vertical posture (beta = 90 deg).
 */
export function computeDeviceQuaternion(
  alphaDeg,
  betaDeg,
  gammaDeg,
  screenAngleDeg = 0,
  targetQuat = null
) {
  const out = targetQuat || new THREE.Quaternion();
  const euler = new THREE.Euler(
    THREE.MathUtils.degToRad(betaDeg),
    THREE.MathUtils.degToRad(alphaDeg),
    THREE.MathUtils.degToRad(-gammaDeg),
    'YXZ'
  );
  out.setFromEuler(euler);
  out.multiply(Q_DEVICE_MINUS_HALF_PI_X);
  const qScreen = new THREE.Quaternion().setFromAxisAngle(
    Z_AXIS_UNIT,
    -THREE.MathUtils.degToRad(screenAngleDeg || 0)
  );
  out.multiply(qScreen);
  return out;
}

/**
 * Computes relative camera yawDeg and pitchDeg from W3C DeviceOrientation Euler angles
 * relative to a stored neutral reference quaternion.
 * - Tilting the phone UP (increasing beta, e.g. 60 -> 85 -> 95 deg) yields POSITIVE pitchDeg (+Y look).
 * - Crossing upright vertical (beta = 90 deg) has zero Euler gimbal-lock jump.
 * - Gaze excursion (hypot of yaw/pitch, ignoring pure in-plane Z wrist roll) softly slerps
 *   referenceQuat past gyroRecenterThresholdDeg so body turns never dead-zone.
 */
export function computeGyroAnglesFromDeviceEuler(
  alphaDeg,
  betaDeg,
  gammaDeg,
  screenAngleDeg = 0,
  referenceQuat = null,
  rigSpec = SCENE_GEOMETRY_SPEC
) {
  const currentQuat = computeDeviceQuaternion(alphaDeg, betaDeg, gammaDeg, screenAngleDeg);
  if (!referenceQuat) {
    return {
      yawDeg: 0.0,
      pitchDeg: 0.0,
      rawYawDeg: 0.0,
      rawPitchDeg: 0.0,
      excursionDeg: 0.0,
      currentQuat,
      referenceQuat: currentQuat.clone()
    };
  }

  let updatedRef = referenceQuat.clone();
  const rawDelta = updatedRef.clone().invert().multiply(currentQuat);
  const rawForward = new THREE.Vector3(0, 0, -1).applyQuaternion(rawDelta);
  const preYawDeg = THREE.MathUtils.radToDeg(Math.atan2(-rawForward.x, -rawForward.z));
  const prePitchDeg = THREE.MathUtils.radToDeg(
    Math.asin(THREE.MathUtils.clamp(rawForward.y, -1.0, 1.0))
  );
  const rawExcursionDeg = Math.hypot(preYawDeg, prePitchDeg);
  const scaledExcursion = rawExcursionDeg * rigSpec.gyroGain;
  if (scaledExcursion > rigSpec.gyroRecenterThresholdDeg && scaledExcursion > 1e-4) {
    const slerpT = (scaledExcursion - rigSpec.gyroRecenterThresholdDeg) / scaledExcursion;
    updatedRef.slerp(currentQuat, THREE.MathUtils.clamp(slerpT, 0.0, 1.0));
  }

  const deltaQuat = updatedRef.clone().invert().multiply(currentQuat);
  const forwardVec = new THREE.Vector3(0, 0, -1).applyQuaternion(deltaQuat);
  const rawYawDeg = THREE.MathUtils.radToDeg(Math.atan2(-forwardVec.x, -forwardVec.z));
  const rawPitchDeg = THREE.MathUtils.radToDeg(
    Math.asin(THREE.MathUtils.clamp(forwardVec.y, -1.0, 1.0))
  );

  const yawDeg = THREE.MathUtils.clamp(
    rawYawDeg * rigSpec.gyroGain,
    -rigSpec.gyroMaxYawDeg,
    rigSpec.gyroMaxYawDeg
  );
  const pitchDeg = THREE.MathUtils.clamp(
    rawPitchDeg * rigSpec.gyroGain,
    -rigSpec.gyroMaxPitchDeg,
    rigSpec.gyroMaxPitchDeg
  );

  return {
    yawDeg,
    pitchDeg,
    rawYawDeg,
    rawPitchDeg,
    excursionDeg: rawExcursionDeg,
    currentQuat,
    referenceQuat: updatedRef
  };
}

/**
 * Computes exact world-space portal cubby coordinates and perspective fitZ distance
 * from either 1500x1000 innerBoxPx [x0, y0, x1, y1] or explicit world dimensions.
 */
export function computePortalWorldGeometry(
  shelfSpec,
  innerScale,
  boreY,
  boreZ,
  sceneBgZ,
  cameraFovDeg,
  cameraAspect
) {
  const worldZ =
    typeof shelfSpec.worldZ === 'number'
      ? shelfSpec.worldZ
      : SCENE_GEOMETRY_SPEC.portalDefaultWorldZ;
  const recessDepth =
    typeof shelfSpec.recessDepth === 'number'
      ? shelfSpec.recessDepth
      : SCENE_GEOMETRY_SPEC.portalDefaultRecessDepth;

  const boreDistBg = Math.max(1.0, boreZ - sceneBgZ);
  const planeRatio = (boreZ - worldZ) / boreDistBg;
  const planeWorldW = SCENE_GEOMETRY_SPEC.bgBaseWidth * innerScale * planeRatio;
  const planeWorldH = SCENE_GEOMETRY_SPEC.bgBaseHeight * innerScale * planeRatio;
  const planeCenterY = boreY * (1.0 - planeRatio);

  let worldX = typeof shelfSpec.worldX === 'number' ? shelfSpec.worldX : 0.0;
  let worldY = typeof shelfSpec.worldY === 'number' ? shelfSpec.worldY : 0.0;
  let width = typeof shelfSpec.width === 'number' ? shelfSpec.width : 110.0;
  let height = typeof shelfSpec.height === 'number' ? shelfSpec.height : 52.0;

  if (Array.isArray(shelfSpec.innerBoxPx) && shelfSpec.innerBoxPx.length === 4) {
    const [x0, y0, x1, y1] = shelfSpec.innerBoxPx;
    const frameW = SCENE_GEOMETRY_SPEC.innerFrameWidthPx;
    const frameH = SCENE_GEOMETRY_SPEC.innerFrameHeightPx;
    const normCx = (x0 + x1) * 0.5 / frameW - 0.5;
    const normCy = 0.5 - (y0 + y1) * 0.5 / frameH;
    worldX = normCx * planeWorldW;
    worldY = planeCenterY + normCy * planeWorldH;
    width = (Math.max(4.0, x1 - x0) / frameW) * planeWorldW;
    height = (Math.max(4.0, y1 - y0) / frameH) * planeWorldH;
  } else {
    const scaleMult = innerScale / SCENE_GEOMETRY_SPEC.minInnerScale;
    worldX *= scaleMult;
    worldY = planeCenterY + worldY * scaleMult;
    width *= scaleMult;
    height *= scaleMult;
  }

  const tanHalfFov = Math.max(0.05, Math.tan(THREE.MathUtils.degToRad(cameraFovDeg * 0.5)));
  const safeAspect = Math.max(0.1, cameraAspect);
  const heightFitZ = (height * 0.5) / tanHalfFov;
  const widthFitZ = (width * 0.5) / (tanHalfFov * safeAspect);
  const fitZ = Math.max(SCENE_GEOMETRY_SPEC.cameraNearPlane + 2.0, Math.min(heightFitZ, widthFitZ));

  return {
    worldX,
    worldY,
    worldZ,
    width,
    height,
    recessDepth,
    heightFitZ,
    widthFitZ,
    fitZ
  };
}

/**
 * Computes the exact maximum arrival distance d = effectiveFitZ from Z = worldZ at which
 * all 4 camera frustum corners (accounting for arrival yawDeg, pitchDeg, rollDeg and portal
 * rotationDeg) lie strictly inside the portal rectangle [-width/2, +width/2] x [-height/2, +height/2].
 */
export function computePortalEffectiveFitZ(
  geom,
  cameraFovDeg,
  cameraAspect,
  arrivalYawDeg = 0.0,
  arrivalPitchDeg = 0.0,
  arrivalRollDeg = 0.0,
  portalRotationDeg = 0.0,
  safetyRatio = SCENE_GEOMETRY_SPEC.portalApertureFitSafetyRatio
) {
  const halfW = Math.max(2.0, (geom.width || 110.0) * 0.5);
  const halfH = Math.max(2.0, (geom.height || 52.0) * 0.5);
  const tanHalfFov = Math.max(0.05, Math.tan(THREE.MathUtils.degToRad(cameraFovDeg * 0.5)));
  const safeAspect = Math.max(0.1, cameraAspect);
  const tanX = tanHalfFov * safeAspect;
  const tanY = tanHalfFov;

  const camQuat = new THREE.Quaternion().setFromEuler(
    new THREE.Euler(
      THREE.MathUtils.degToRad(arrivalPitchDeg || 0.0),
      THREE.MathUtils.degToRad(arrivalYawDeg || 0.0),
      THREE.MathUtils.degToRad(arrivalRollDeg || 0.0),
      'YXZ'
    )
  );
  const portalRad = THREE.MathUtils.degToRad(portalRotationDeg || 0.0);
  const cosP = Math.cos(portalRad);
  const sinP = Math.sin(portalRad);

  let maxAllowedDist = Infinity;
  const signs = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1]
  ];
  const rayVec = new THREE.Vector3();
  for (const [sx, sy] of signs) {
    rayVec.set(sx * tanX, sy * tanY, -1.0).applyQuaternion(camQuat);
    const negRz = Math.max(0.05, -rayVec.z);
    const slopeWorldX = rayVec.x / negRz;
    const slopeWorldY = rayVec.y / negRz;
    const slopeLocalU = Math.abs(slopeWorldX * cosP + slopeWorldY * sinP);
    const slopeLocalV = Math.abs(-slopeWorldX * sinP + slopeWorldY * cosP);
    if (slopeLocalU > 1e-6) {
      maxAllowedDist = Math.min(maxAllowedDist, halfW / slopeLocalU);
    }
    if (slopeLocalV > 1e-6) {
      maxAllowedDist = Math.min(maxAllowedDist, halfH / slopeLocalV);
    }
  }
  const fallbackFitZ = typeof geom.fitZ === 'number' ? geom.fitZ : 80.0;
  const rawFit = Number.isFinite(maxAllowedDist) ? maxAllowedDist : fallbackFitZ;
  return Math.max(SCENE_GEOMETRY_SPEC.cameraNearPlane + 2.0, rawFit * safetyRatio);
}

const ROOM_DEPTH_VERTEX_SHADER = `
uniform sampler2D uDepthMap;
uniform float uDepthAmplitude;
uniform float uDistBg;
uniform float uPinMarginUv;
uniform float uOpticalCenterLocalX;
uniform float uOpticalCenterLocalY;

varying vec2 vUv;
#include <clipping_planes_pars_vertex>

void main() {
    vUv = uv;
    float rawDepth = texture2D(uDepthMap, uv).r;

    // Smoothly pin outer UV perimeter of the 2700x1800 outpainted room to Z = bgZ
    // so analytical frustum ray intersections on Z = bgZ remain exact.
    // Guard uPinMarginUv == 0.0 because GLSL ES 3.0 smoothstep(0.0, 0.0, x) is undefined.
    float pinX = (uPinMarginUv > 1e-5)
        ? smoothstep(0.0, uPinMarginUv, uv.x) * smoothstep(0.0, uPinMarginUv, 1.0 - uv.x)
        : 1.0;
    float pinY = (uPinMarginUv > 1e-5)
        ? smoothstep(0.0, uPinMarginUv, uv.y) * smoothstep(0.0, uPinMarginUv, 1.0 - uv.y)
        : 1.0;
    float zDisp = rawDepth * uDepthAmplitude * pinX * pinY;

    // Ray-projective compensation: zero rest-pose distortion at scene optical center (0, y0, z0),
    // revealing authentic 3D room perspective as soon as camera pose translates, dollies, or rotates.
    float rayScale = (uDistBg - zDisp) / uDistBg;
    vec3 displaced = vec3(
        uOpticalCenterLocalX + (position.x - uOpticalCenterLocalX) * rayScale,
        uOpticalCenterLocalY + (position.y - uOpticalCenterLocalY) * rayScale,
        zDisp
    );

    vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    #include <clipping_planes_vertex>
}
`;

const ROOM_DEPTH_FRAGMENT_SHADER = `
precision highp float;

uniform sampler2D uColorMap;
uniform float uOpacity;
uniform float uAlphaCutout;
uniform vec3 uTint;
uniform vec3 uClearColor;
uniform float uFadeToClear;
uniform float uClipUvMinX;
uniform float uClipUvMaxX;
uniform vec4 uCavityUvRect;
uniform vec2 uCavityFeatherUv;
uniform float uOuterWallOpacity;
varying vec2 vUv;
#include <clipping_planes_pars_fragment>

void main() {
    #include <clipping_planes_fragment>
    if (vUv.x < uClipUvMinX || vUv.x > uClipUvMaxX) { discard; }
    vec4 texel = texture2D(uColorMap, vUv);
    if (uAlphaCutout > 0.0 && texel.a < uAlphaCutout) { discard; }
    vec3 tintedRgb = clamp(texel.rgb * uTint, 0.0, 1.0);
    vec3 rgb = (uFadeToClear > 0.0) ? mix(tintedRgb, uClearColor, uFadeToClear) : tintedRgb;
    float outAlpha = (uAlphaCutout > 0.0 || uFadeToClear > 0.0) ? 1.0 : (texel.a * uOpacity);
    if (uOuterWallOpacity < 0.9999) {
        float cavX = smoothstep(uCavityUvRect.x - uCavityFeatherUv.x, uCavityUvRect.x + uCavityFeatherUv.x, vUv.x)
                   * (1.0 - smoothstep(uCavityUvRect.z - uCavityFeatherUv.x, uCavityUvRect.z + uCavityFeatherUv.x, vUv.x));
        float cavY = smoothstep(uCavityUvRect.y - uCavityFeatherUv.y, uCavityUvRect.y + uCavityFeatherUv.y, vUv.y)
                   * (1.0 - smoothstep(uCavityUvRect.w - uCavityFeatherUv.y, uCavityUvRect.w + uCavityFeatherUv.y, vUv.y));
        float cavityMask = cavX * cavY;
        float borderFade = smoothstep(0.0, 0.06, vUv.x) * smoothstep(0.0, 0.06, 1.0 - vUv.x)
                         * smoothstep(0.0, 0.06, vUv.y) * smoothstep(0.0, 0.06, 1.0 - vUv.y);
        float effBorderFade = mix(borderFade, 1.0, uOuterWallOpacity);
        outAlpha *= mix(uOuterWallOpacity * effBorderFade, 1.0, cavityMask);
        if (outAlpha <= 0.0005) { discard; }
    }
    gl_FragColor = vec4(rgb, outAlpha);
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

varying vec2 vUv;

void main() {
    vec4 colCurrent = texture2D(uTexCurrent, vUv);
    vec4 colNext    = texture2D(uTexNext, vUv);
    float blendFactor = smoothstep(0.0, 1.0, clamp(uProgress, 0.0, 1.0));
    gl_FragColor = mix(colCurrent, colNext, blendFactor);
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
 * Euler(pitchRad, yawRad, 0, 'YXZ') intersect Z = bgZ inside [-maxHalfW..maxHalfW] x [-maxHalfH..maxHalfH],
 * and (when fgBounds is active) stay within any edge-bleeding foreground subject bounds.
 */
export function doFrustumCornersFitOnBackground(
  posX,
  posY,
  posZ,
  yawRad,
  pitchRad,
  tanHalfFov,
  aspect,
  bgZ,
  maxHalfW,
  maxHalfH,
  fgBounds = null,
  rollRad = 0.0
) {
  const cosY = Math.cos(yawRad);
  const sinY = Math.sin(yawRad);
  const cosP = Math.cos(pitchRad);
  const sinP = Math.sin(pitchRad);
  const cosR = Math.cos(rollRad);
  const sinR = Math.sin(rollRad);
  const distZ = posZ - bgZ;
  if (distZ <= 1.0) return false;

  const checkFg =
    fgBounds !== null &&
    typeof fgBounds === 'object' &&
    posZ > fgBounds.guardMinZ &&
    posZ - fgBounds.z > 1.0;
  const distFg = checkFg ? posZ - fgBounds.z : 0.0;

  const tanX = tanHalfFov * aspect;
  const tanY = tanHalfFov;

  for (let sx = -1; sx <= 1; sx += 2) {
    for (let sy = -1; sy <= 1; sy += 2) {
      const vx0 = sx * tanX;
      const vy0 = sy * tanY;
      // Local ray is (vx0, vy0, -1); apply R_z(rollRad), then R_x(pitchRad), then R_y(yawRad)
      const vx = vx0 * cosR - vy0 * sinR;
      const vy = vx0 * sinR + vy0 * cosR;
      const y1 = vy * cosP + sinP;
      const z1 = vy * sinP - cosP;
      const rx = vx * cosY + z1 * sinY;
      const ry = y1;
      const rz = -vx * sinY + z1 * cosY;

      if (rz >= -0.15) {
        return false;
      }
      const invNegRz = 1.0 / (-rz);
      const t = distZ * invNegRz;
      const ix = posX + t * rx;
      const iy = posY + t * ry;
      if (Math.abs(ix) > maxHalfW || Math.abs(iy) > maxHalfH) {
        return false;
      }
      if (checkFg) {
        const tFg = distFg * invNegRz;
        const fx = posX + tFg * rx;
        const fy = posY + tFg * ry;
        if (fgBounds.minX !== null && fx < fgBounds.minX) return false;
        if (fgBounds.maxX !== null && fx > fgBounds.maxX) return false;
        if (fgBounds.minY !== null && fy < fgBounds.minY) return false;
        if (fgBounds.maxY !== null && fy > fgBounds.maxY) return false;
      }
    }
  }
  return true;
}

/**
 * Clamps a 6-DOF camera pose (x, y, z, yawDeg, pitchDeg) so that:
 * 1. posZ stays within [minCameraZ, effectiveMaxCameraZ] where centered view fits inside layer0
 *    (and any edge-bleeding foreground bounds).
 * 2. Off-axis translation & rotation scale smoothly via binary search whenever any viewport
 *    corner ray would otherwise cross the expanded 2700x1800 room boundary.
 */
export function clampPoseToRoomFrustum(rawPose, fovDeg, aspect, layer0Scale, spec) {
  const halfFovRad = THREE.MathUtils.degToRad(fovDeg * 0.5);
  const tanHalfFov = Math.tan(halfFovRad);
  const tanX = Math.max(0.05, tanHalfFov * aspect);
  const tanY = Math.max(0.05, tanHalfFov);
  const maxHalfW = 0.5 * spec.bgBaseWidth * layer0Scale * spec.frustumEdgeSafetyRatio;
  const maxHalfH = 0.5 * spec.bgBaseHeight * layer0Scale * spec.frustumEdgeSafetyRatio;
  const restX = typeof spec.restCameraX === 'number' ? spec.restCameraX : 0.0;
  const restY = typeof spec.restCameraY === 'number' ? spec.restCameraY : 0.0;
  const rollDeg = typeof rawPose.rollDeg === 'number' ? rawPose.rollDeg : 0.0;
  const rollRad = THREE.MathUtils.degToRad(rollDeg);
  const fgBounds = spec.fgBleedBounds || null;

  const maxCenteredDistW = (maxHalfW * 0.95) / tanX;
  const maxCenteredDistH = (maxHalfH * 0.95) / tanY;
  let effectiveMaxZ = Math.min(
    spec.maxCameraZ,
    spec.bgZ + Math.min(maxCenteredDistW, maxCenteredDistH)
  );
  if (fgBounds) {
    if (fgBounds.minX !== null && fgBounds.maxX !== null) {
      effectiveMaxZ = Math.min(
        effectiveMaxZ,
        fgBounds.z + ((fgBounds.maxX - fgBounds.minX) * 0.98) / (2.0 * tanX)
      );
    } else if (fgBounds.minX !== null) {
      effectiveMaxZ = Math.min(
        effectiveMaxZ,
        ((maxHalfW * 0.95 - fgBounds.minX) / tanX + fgBounds.z + spec.bgZ) * 0.5
      );
    } else if (fgBounds.maxX !== null) {
      effectiveMaxZ = Math.min(
        effectiveMaxZ,
        ((fgBounds.maxX + maxHalfW * 0.95) / tanX + fgBounds.z + spec.bgZ) * 0.5
      );
    }
    if (fgBounds.minY !== null && fgBounds.maxY !== null) {
      effectiveMaxZ = Math.min(
        effectiveMaxZ,
        fgBounds.z + ((fgBounds.maxY - fgBounds.minY) * 0.98) / (2.0 * tanY)
      );
    } else if (fgBounds.minY !== null) {
      effectiveMaxZ = Math.min(
        effectiveMaxZ,
        ((maxHalfH * 0.95 - fgBounds.minY) / tanY + fgBounds.z + spec.bgZ) * 0.5
      );
    } else if (fgBounds.maxY !== null) {
      effectiveMaxZ = Math.min(
        effectiveMaxZ,
        ((fgBounds.maxY + maxHalfH * 0.95) / tanY + fgBounds.z + spec.bgZ) * 0.5
      );
    }
  }
  effectiveMaxZ = Math.max(spec.minCameraZ, effectiveMaxZ);

  const clampedZ = THREE.MathUtils.clamp(rawPose.z, spec.minCameraZ, effectiveMaxZ);
  const distBgAtZ = Math.max(1.0, clampedZ - spec.bgZ);
  let validMinX = -(maxHalfW * 0.95 - distBgAtZ * tanX);
  let validMaxX = +(maxHalfW * 0.95 - distBgAtZ * tanX);
  let validMinY = -(maxHalfH * 0.95 - distBgAtZ * tanY);
  let validMaxY = +(maxHalfH * 0.95 - distBgAtZ * tanY);

  if (fgBounds && clampedZ > fgBounds.guardMinZ && clampedZ - fgBounds.z > 1.0) {
    const distFgAtZ = clampedZ - fgBounds.z;
    if (fgBounds.minX !== null) validMinX = Math.max(validMinX, fgBounds.minX + distFgAtZ * tanX + 2.0);
    if (fgBounds.maxX !== null) validMaxX = Math.min(validMaxX, fgBounds.maxX - distFgAtZ * tanX - 2.0);
    if (fgBounds.minY !== null) validMinY = Math.max(validMinY, fgBounds.minY + distFgAtZ * tanY + 2.0);
    if (fgBounds.maxY !== null) validMaxY = Math.min(validMaxY, fgBounds.maxY - distFgAtZ * tanY - 2.0);
  }
  if (validMinX > validMaxX) {
    const midX = 0.5 * (validMinX + validMaxX);
    validMinX = midX;
    validMaxX = midX;
  }
  if (validMinY > validMaxY) {
    const midY = 0.5 * (validMinY + validMaxY);
    validMinY = midY;
    validMaxY = midY;
  }

  const refX = THREE.MathUtils.clamp(restX, validMinX, validMaxX);
  const refY = THREE.MathUtils.clamp(restY, validMinY, validMaxY);

  const candX = THREE.MathUtils.clamp(rawPose.x, restX - spec.maxStrafeX, restX + spec.maxStrafeX);
  const candY = THREE.MathUtils.clamp(rawPose.y, restY - spec.maxStrafeY, restY + spec.maxStrafeY);
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
      maxHalfH,
      fgBounds,
      rollRad
    )
  ) {
    return {
      x: candX,
      y: candY,
      z: clampedZ,
      yawDeg: candYawDeg,
      pitchDeg: candPitchDeg,
      rollDeg
    };
  }

  // Stage 1: Joint radial step from feasible anchor (refX, refY, 0, 0) so diagonal excursions
  // preserve proportional yaw & pitch without horizontal greedily starving pitch.
  let loJ = 0.0;
  let hiJ = 1.0;
  for (let iter = 0; iter < 8; iter += 1) {
    const mid = 0.5 * (loJ + hiJ);
    if (
      doFrustumCornersFitOnBackground(
        refX + (candX - refX) * mid,
        refY + (candY - refY) * mid,
        clampedZ,
        candYawRad * mid,
        candPitchRad * mid,
        tanHalfFov,
        aspect,
        spec.bgZ,
        maxHalfW,
        maxHalfH,
        fgBounds,
        rollRad
      )
    ) {
      loJ = mid;
    } else {
      hiJ = mid;
    }
  }

  // Stage 2: Unlock any remaining horizontal slack while keeping joint vertical excursion intact.
  let loH = loJ;
  let hiH = 1.0;
  const jointY = refY + (candY - refY) * loJ;
  const jointPitchRad = candPitchRad * loJ;
  for (let iter = 0; iter < 6; iter += 1) {
    const mid = 0.5 * (loH + hiH);
    if (
      doFrustumCornersFitOnBackground(
        refX + (candX - refX) * mid,
        jointY,
        clampedZ,
        candYawRad * mid,
        jointPitchRad,
        tanHalfFov,
        aspect,
        spec.bgZ,
        maxHalfW,
        maxHalfH,
        fgBounds,
        rollRad
      )
    ) {
      loH = mid;
    } else {
      hiH = mid;
    }
  }
  const finalX = refX + (candX - refX) * loH;
  const finalYawDeg = candYawDeg * loH;
  const finalYawRad = candYawRad * loH;

  // Stage 3: Unlock any remaining vertical slack at finalX / finalYawRad.
  let loV = loJ;
  let hiV = 1.0;
  for (let iter = 0; iter < 6; iter += 1) {
    const mid = 0.5 * (loV + hiV);
    if (
      doFrustumCornersFitOnBackground(
        finalX,
        refY + (candY - refY) * mid,
        clampedZ,
        finalYawRad,
        candPitchRad * mid,
        tanHalfFov,
        aspect,
        spec.bgZ,
        maxHalfW,
        maxHalfH,
        fgBounds,
        rollRad
      )
    ) {
      loV = mid;
    } else {
      hiV = mid;
    }
  }

  return {
    x: finalX,
    y: refY + (candY - refY) * loV,
    z: clampedZ,
    yawDeg: finalYawDeg,
    pitchDeg: candPitchDeg * loV,
    rollDeg
  };
}

export class SpatialSceneManager {
  static _computeGyroAnglesFromDeviceEuler(
    alphaDeg,
    betaDeg,
    gammaDeg,
    screenAngleDeg = 0,
    referenceQuat = null
  ) {
    return computeGyroAnglesFromDeviceEuler(
      alphaDeg,
      betaDeg,
      gammaDeg,
      screenAngleDeg,
      referenceQuat
    );
  }

  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.onSceneChange = options.onSceneChange || (() => {});
    this.onReady = options.onReady || (() => {});
    this.onPortalHover = options.onPortalHover || (() => {});
    this.onPortalUnlock =
      typeof options.onPortalUnlock === 'function' ? options.onPortalUnlock : () => {};
    this.onLoadingProgress =
      typeof options.onLoadingProgress === 'function' ? options.onLoadingProgress : null;

    const assetVer = 'v=neural41';
    this._assetVer = assetVer;
    this.sceneConfigs = {
      shelf_portal: {
        id: 'shelf_portal',
        title: 'קיר המדפים — שער התערוכות (גלריה בנימין)',
        bgTexture: `assets/scenes/scene9_shelf_portal_bg.png?${assetVer}`,
        depthTexture: `assets/scenes/scene9_shelf_portal_bg_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene9_shelf_portal_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene9_shelf_portal_fg.png?${assetVer}`,
        fgExpanded: true,
        boreSightAtRest: true,
        bgZ: -350,
        midZ: -296,
        fgZ: -280,
        minInnerScale: 2.40,
        roomDepthAmplitude: 358.0,
        initialCameraY: -36,
        initialCameraZ: 1740,
        minCameraZ: -260,
        maxCameraZ: 2040,
        maxStrafeX: 480,
        maxStrafeY: 340,
        maxYawDeg: 42,
        maxPitchDeg: 30,
        dollySpeedMult: 1.55,
        midRenderOrder: 1,
        fgRenderOrder: SCENE_GEOMETRY_SPEC.shelfFgRenderOrder,
        particleColor: 0xCFC8BC,
        particleCount: 75,
        hasReflection: false,
        portalShelves: []
      },
      paper_capsules: {
        id: 'paper_capsules',
        title: 'קפסולות נייר תלויות — ׳אולי שם יהיה יותר טוב׳ (גלריה בנימין)',
        bgTexture: `assets/scenes/scene1_herzliya_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene1_herzliya_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene1_herzliya_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene1_herzliya_fg.png?${assetVer}`,
        boreSightAtRest: true,
        roomDepthAmplitude: 92.0,
        initialCameraY: 0,
        initialCameraZ: 400,
        maxCameraZ: 685,
        particleColor: 0xEAD8BE,
        particleCount: 85,
        hasReflection: false
      },
      bene_beitcha: {
        id: 'bene_beitcha',
        title: 'תערוכת ׳בנה ביתך׳ — ארכיטקטורות של שבריריות (מוזיאון רמת גן)',
        bgTexture: `assets/scenes/scene2_denya_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene2_denya_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene2_denya_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene2_denya_mid_depth.png?${assetVer}`,
        fgTexture: `assets/scenes/scene2_denya_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene2_denya_fg_depth.png?${assetVer}`,
        reflectionTexture: `assets/scenes/scene2_reflections.png?${assetVer}`,
        boreSightAtRest: true,
        bgZ: -350,
        midZ: -350,
        midDepthAmplitude: 48.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        fgZ: -336,
        fgDepthAmplitude: 64.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        roomDepthAmplitude: 92.0,
        initialCameraY: 0,
        initialCameraZ: 400,
        maxCameraZ: 685,
        particleColor: 0xB8C7CF,
        particleCount: 90,
        hasReflection: true
      },
      denya_model: {
        id: 'denya_model',
        title: 'דגם בית דניה — ׳בנה ביתך׳ (מודל קרטון, עץ וזיכרון חיפאי)',
        bgTexture: `assets/scenes/scene8_denya_model_bg.png?${assetVer}`,
        depthTexture: `assets/scenes/scene8_denya_model_bg_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene8_denya_model_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene8_denya_model_fg.png?${assetVer}`,
        boreSightAtRest: true,
        fgZ: -85,
        midZ: -155,
        roomDepthAmplitude: 115.0,
        initialCameraY: -22,
        initialCameraZ: 860,
        maxCameraZ: 1280,
        particleColor: 0xD6C8B4,
        particleCount: 70,
        hasReflection: false
      },
      denya_living_room: {
        id: 'denya_living_room',
        title: 'ארוחה בסלון דניה — שער אל הספה והגיטרה ואל ׳שיטוט 2׳',
        bgTexture: `assets/scenes/denya_living_dining_bg.jpg?${assetVer}`,
        depthTexture: `assets/scenes/denya_living_dining_depth.png?${assetVer}`,
        midTexture: `assets/scenes/denya_living_dining_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/denya_living_dining_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 88.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 192,
        midSegmentsY: 128,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/denya_living_dining_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/denya_living_dining_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 195.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: false,
        fgBleedEdges: { left: false, right: true, top: false, bottom: true, guardMinZ: 210.0 },
        boreSightAtRest: true,
        bgZ: -385.0,
        midZ: -382.0,
        fgZ: -374.0,
        minInnerScale: 2.12,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 110.0,
        initialCameraY: 0.0,
        initialCameraZ: 460.0,
        minCameraZ: -120.0,
        maxCameraZ: 640.0,
        maxYawDeg: 18.5,
        maxPitchDeg: 12.5,
        maxStrafeX: 68.0,
        maxStrafeY: 46.0,
        dollySpeedMult: 1.35,
        midRenderOrder: 8,
        fgRenderOrder: 10,
        extraLayers: [
          {
            id: 'wandering2_wall_frame',
            labelHe: 'מסגרת עץ אלון על קיר הסלון בדניה — שער אל ׳שיטוט 2׳',
            texture: `assets/scenes/denya_living_dining_wandering2_frame.png?${assetVer}`,
            depthTexture: `assets/scenes/denya_living_dining_wandering2_frame_depth.png?${assetVer}`,
            zPos: -385.0,
            depthAmplitude: 26.0,
            pinMarginUv: 0.0,
            segmentsX: 160,
            segmentsY: 108,
            renderOrder: 6,
            expanded: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false
          },
          {
            id: 'planted_couch_trio',
            labelHe: 'טל, יערה וכרמל עם הגיטרה על הספה האפורה בסלון דניה',
            texture: `assets/scenes/denya_living_dining_couch_people.png?${assetVer}`,
            depthTexture: `assets/scenes/denya_living_dining_couch_people_depth.png?${assetVer}`,
            zPos: -384.0,
            depthAmplitude: 84.0,
            pinMarginUv: 0.0,
            segmentsX: 192,
            segmentsY: 128,
            renderOrder: 7,
            expanded: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false
          }
        ],
        particleColor: 0xE8DCC4,
        particleCount: 65,
        hasReflection: false,
        portalShelves: []
      },
      denya_couch_guitar: {
        id: 'denya_couch_guitar',
        title: 'ספת הסלון בדניה — נגינה בגיטרה תחת קיר העבודות ושער ׳שיטוט 2׳',
        bgTexture: `assets/scenes/denya_living_couch_bg.jpg?${assetVer}`,
        depthTexture: `assets/scenes/denya_living_couch_depth.png?${assetVer}`,
        midTexture: `assets/scenes/denya_living_couch_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/denya_living_couch_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 82.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/denya_living_couch_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/denya_living_couch_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 125.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: true,
        boreSightAtRest: true,
        bgZ: -350.0,
        midZ: -342.0,
        fgZ: -322.0,
        minInnerScale: 1.98,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 102.0,
        initialCameraY: 0.0,
        initialCameraZ: 455.0,
        minCameraZ: -90.0,
        maxCameraZ: 650.0,
        maxYawDeg: 28.0,
        maxPitchDeg: 20.0,
        maxStrafeX: 220.0,
        maxStrafeY: 140.0,
        dollySpeedMult: 1.30,
        midRenderOrder: 8,
        fgRenderOrder: 10,
        extraLayers: [
          {
            id: 'wandering2_wall_frame',
            labelHe: 'מסגרת עץ אלון על קיר הסלון בדניה — שער אל ׳שיטוט 2׳',
            texture: `assets/scenes/denya_living_couch_wandering2_frame.png?${assetVer}`,
            depthTexture: `assets/scenes/denya_living_couch_wandering2_frame_depth.png?${assetVer}`,
            zPos: -348.0,
            depthAmplitude: 24.0,
            pinMarginUv: 0.0,
            segmentsX: 160,
            segmentsY: 108,
            renderOrder: 6,
            expanded: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false
          }
        ],
        particleColor: 0xE2D7C5,
        particleCount: 60,
        hasReflection: false,
        portalShelves: []
      },
      plan_b_pipe: {
        id: 'plan_b_pipe',
        title: 'Plan B — מבעד לצינור המלט והנייר (גלריה בנימין)',
        bgTexture: `assets/scenes/scene4_plan_b_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene4_plan_b_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene4_plan_b_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene4_plan_b_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 1.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 64,
        midSegmentsY: 44,
        midDepthTest: true,
        fgTexture: `assets/scenes/scene4_plan_b_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene4_plan_b_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 1.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 64,
        fgSegmentsY: 44,
        fgDepthTest: true,
        bgZ: -520,
        minInnerScale: 2.0783,
        fgZ: -35,
        midZ: -180,
        bgSegmentsX: 256,
        bgSegmentsY: 172,
        roomDepthAmplitude: 0.0,
        initialCameraY: 0,
        initialCameraZ: 400,
        minCameraZ: 25,
        maxCameraZ: 685,
        maxYawDeg: 18.0,
        maxPitchDeg: 12.5,
        maxStrafeX: 56.0,
        maxStrafeY: 36.0,
        dollySpeedMult: 1.20,
        midRenderOrder: 4,
        fgRenderOrder: 6,
        tunnelBoreTarget: {
          innerNormX: 879.0 / 1500.0,
          innerNormY: 462.0 / 1000.0,
          targetZ: -330.0
        },
        extraLayers: [
          {
            id: 'pipe_shell',
            labelHe: 'גוף צינור המלט והמנהרה התלת-ממדית',
            texture: `assets/scenes/scene4_plan_b_pipe_shell.png?${assetVer}`,
            depthTexture: `assets/scenes/scene4_plan_b_pipe_shell_depth.png?${assetVer}`,
            zPos: -430,
            renderOrder: 1,
            expanded: true,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: true,
            depthWrite: true,
            depthAmplitude: 470.0,
            alphaCutout: 0.5,
            pinMarginUv: 0.0,
            segmentsX: 256,
            segmentsY: 172
          },
          {
            id: 'hanging_ring',
            labelHe: 'טבעת המלט התלויה מתקרת הצינור',
            texture: `assets/scenes/scene4_plan_b_hanging_ring.png?${assetVer}`,
            zPos: -95,
            renderOrder: 5,
            depthTest: true
          },
          {
            id: 'rear_disc',
            labelHe: 'דיסקית התלייה בפתח האחורי של הצינור',
            texture: `assets/scenes/scene4_plan_b_rear_disc.png?${assetVer}`,
            zPos: -265,
            renderOrder: 3,
            depthTest: true
          }
        ],
        particleColor: 0x9FB7CC,
        particleCount: 65,
        hasReflection: false
      },
      plan_b_pipe_v3: {
        id: 'plan_b_pipe_v3',
        title: 'Plan B (גרסה 3) — תמי ויערה באור הדיפוזי בקצה הצינור',
        bgTexture: `assets/scenes/scene4_plan_b_v3_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene4_plan_b_v3_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene4_plan_b_v3_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene4_plan_b_v3_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 1.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 64,
        midSegmentsY: 44,
        midDepthTest: true,
        fgTexture: `assets/scenes/scene4_plan_b_v3_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene4_plan_b_v3_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 1.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 64,
        fgSegmentsY: 44,
        fgDepthTest: true,
        bgZ: -520,
        minInnerScale: 2.0783,
        fgZ: -35,
        midZ: -180,
        bgSegmentsX: 256,
        bgSegmentsY: 172,
        roomDepthAmplitude: 0.0,
        initialCameraY: 0,
        initialCameraZ: 400,
        minCameraZ: -145,
        maxCameraZ: 685,
        maxYawDeg: 18.0,
        maxPitchDeg: 12.5,
        maxStrafeX: 56.0,
        maxStrafeY: 36.0,
        dollySpeedMult: 1.50,
        midRenderOrder: 6,
        fgRenderOrder: 8,
        tunnelBoreTarget: {
          innerNormX: 878.0 / 1500.0,
          innerNormY: 488.0 / 1000.0,
          targetZ: -332.0
        },
        extraLayers: [
          {
            id: 'pipe_shell',
            labelHe: 'גוף צינור המלט והמנהרה התלת-ממדית',
            texture: `assets/scenes/scene4_plan_b_v3_pipe_shell.png?${assetVer}`,
            depthTexture: `assets/scenes/scene4_plan_b_v3_pipe_shell_depth.png?${assetVer}`,
            zPos: -430,
            renderOrder: 1,
            expanded: true,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: true,
            depthWrite: true,
            depthAmplitude: 470.0,
            alphaCutout: 0.5,
            pinMarginUv: 0.0,
            segmentsX: 256,
            segmentsY: 172
          },
          {
            id: 'hanging_ring',
            labelHe: 'טבעת המלט התלויה מתקרת הצינור',
            texture: `assets/scenes/scene4_plan_b_v3_hanging_ring.png?${assetVer}`,
            zPos: -95,
            renderOrder: 7,
            depthTest: true
          },
          {
            id: 'rear_disc',
            labelHe: 'דיסקית התלייה בפתח האחורי של הצינור',
            texture: `assets/scenes/scene4_plan_b_v3_rear_disc.png?${assetVer}`,
            zPos: -265,
            renderOrder: 5,
            depthTest: true
          },
          {
            id: 'yaara_front',
            labelHe: 'יערה והספר הפתוח באור הדיפוזי בקצה הצינור',
            texture: `assets/scenes/scene4_plan_b_v3_yaara_front.png?${assetVer}`,
            zPos: -322,
            renderOrder: 4,
            depthTest: true
          },
          {
            id: 'tami_back',
            labelHe: 'תמי קוראת עם יערה באור הדיפוזי בקצה הצינור',
            texture: `assets/scenes/scene4_plan_b_v3_tami_back.png?${assetVer}`,
            zPos: -342,
            renderOrder: 3,
            depthTest: true
          },
          {
            id: 'chamber_halo',
            labelHe: 'הילת החלון הדיפוזית המתמזגת לעומק הצינור',
            texture: `assets/scenes/scene4_plan_b_v3_chamber_halo.png?${assetVer}`,
            zPos: -376,
            renderOrder: 2,
            depthTest: true
          }
        ],
        particleColor: 0xA8C0DC,
        particleCount: 65,
        hasReflection: false
      },
      wandering_drawing_1: {
        id: 'wandering_drawing_1',
        title: '׳שיטוט ברישום 1׳ — מרחב רישומי משולב ושער אל ׳מגילה וכף רגל׳',
        bgTexture: `assets/scenes/scene7_wandering1_bg.png?${assetVer}`,
        depthTexture: `assets/scenes/scene7_wandering1_bg_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene7_wandering1_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene7_wandering1_fg.png?${assetVer}`,
        boreSightAtRest: true,
        midZ: -80.0,
        midRenderOrder: 6,
        fgRenderOrder: 8,
        roomDepthAmplitude: 112.0,
        initialCameraY: 0,
        initialCameraZ: 545,
        maxCameraZ: 780,
        extraLayers: [
          {
            id: 'scroll_foot',
            labelHe: 'כף הרגל הקרמית והמוט העטוף בחזית ׳מגילה וכף רגל׳',
            texture: `assets/scenes/scene7_wandering1_scroll_foot.png?${assetVer}`,
            depthTexture: `assets/scenes/scene7_wandering1_scroll_foot_depth.png?${assetVer}`,
            zPos: -48.0,
            depthAmplitude: 28.0,
            pinMarginUv: 0.0,
            segmentsX: 192,
            segmentsY: 128,
            renderOrder: 7,
            expanded: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false
          }
        ],
        particleColor: 0xDDD4C6,
        particleCount: 60,
        hasReflection: false,
        portalShelves: []
      },
      scroll_palm_world: {
        id: 'scroll_palm_world',
        title: 'עולם הדקל והמוטות בתוך המגילה — ׳מגילה וכף רגל׳',
        bgTexture: `assets/scenes/scene14_scroll_world_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene14_scroll_world_bg_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene14_scroll_world_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene14_scroll_world_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 158.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/scene14_scroll_world_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene14_scroll_world_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 196.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: false,
        boreSightAtRest: true,
        bgZ: -360.0,
        midZ: -358.0,
        fgZ: -354.0,
        minInnerScale: 1.96,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 118.0,
        initialCameraY: -8.0,
        initialCameraZ: 535.0,
        minCameraZ: -95.0,
        maxCameraZ: 680.0,
        maxYawDeg: 22.0,
        maxPitchDeg: 15.0,
        maxStrafeX: 115.0,
        maxStrafeY: 72.0,
        dollySpeedMult: 1.35,
        midRenderOrder: 2,
        fgRenderOrder: 3,
        particleColor: 0xE8DEC2,
        particleCount: 65,
        hasReflection: false
      },
      humpty_dumpty_bundle: {
        id: 'humpty_dumpty_bundle',
        title: 'צרור עטוף בבד וחבל — פסל עטוף על משטח הגלריה',
        bgTexture: `assets/scenes/scene15_humpty_bundle_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene15_humpty_bundle_bg_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene15_humpty_bundle_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene15_humpty_bundle_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene15_humpty_bundle_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 168.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: false,
        boreSightAtRest: true,
        bgZ: -350.0,
        midZ: -240.0,
        fgZ: -346.0,
        minInnerScale: 1.94,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 96.0,
        initialCameraY: 0.0,
        initialCameraZ: 530.0,
        minCameraZ: -85.0,
        maxCameraZ: 690.0,
        maxYawDeg: 24.0,
        maxPitchDeg: 16.5,
        maxStrafeX: 135.0,
        maxStrafeY: 85.0,
        dollySpeedMult: 1.30,
        midRenderOrder: 2,
        fgRenderOrder: 4,
        particleColor: 0xDFD4C4,
        particleCount: 55,
        hasReflection: false
      },
      humpty_dumpty_booth: {
        id: 'humpty_dumpty_booth',
        title: 'המפטי דמפטי בתא האפור — סעודת יחיד בסטודיו',
        bgTexture: `assets/scenes/scene16_humpty_booth_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene16_humpty_booth_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene16_humpty_booth_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene16_humpty_booth_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 175.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/scene16_humpty_booth_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene16_humpty_booth_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 195.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: false,
        fgBleedEdges: { top: true, bottom: true, guardMinZ: 210.0 },
        boreSightAtRest: true,
        bgZ: -350.0,
        midZ: -350.0,
        fgZ: -350.0,
        minInnerScale: 1.94,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 145.0,
        initialCameraY: -6.0,
        initialCameraZ: 525.0,
        minCameraZ: -90.0,
        maxCameraZ: 685.0,
        maxYawDeg: 22.0,
        maxPitchDeg: 15.0,
        maxStrafeX: 125.0,
        maxStrafeY: 78.0,
        dollySpeedMult: 1.32,
        midRenderOrder: 6,
        fgRenderOrder: 8,
        particleColor: 0xE2D7C5,
        particleCount: 55,
        hasReflection: false
      },
      tzofen_brick_table: {
        id: 'tzofen_brick_table',
        title: '׳צופן׳ (2011) — שולחן הברזל ו-380 לבני הבוץ עם חלון אל צמרת העץ',
        bgTexture: `assets/scenes/scene12_tzofen_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene12_tzofen_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene12_tzofen_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene12_tzofen_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene12_tzofen_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 175.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: false,
        boreSightAtRest: true,
        bgZ: -350.0,
        midZ: -260.0,
        fgZ: -350.0,
        minInnerScale: 1.88,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 128.0,
        initialCameraY: 52.0,
        initialCameraZ: 755.0,
        minCameraZ: -85.0,
        maxCameraZ: 880.0,
        maxYawDeg: 26.0,
        maxPitchDeg: 18.0,
        maxStrafeX: 165.0,
        maxStrafeY: 95.0,
        dollySpeedMult: 1.35,
        midRenderOrder: 6,
        fgRenderOrder: 8,
        particleColor: 0xE6D8C2,
        particleCount: 60,
        hasReflection: false,
        portalShelves: []
      },
      tzofen_brick_table_v2: {
        id: 'tzofen_brick_table_v2',
        title: '׳צופן׳ (2011) — הפרדת לבני חזית ושולחן (ארכיון)',
        bgTexture: `assets/scenes/scene12_tzofen_v2_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene12_tzofen_v2_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene12_tzofen_v2_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene12_tzofen_v2_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 200.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/scene12_tzofen_v2_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene12_tzofen_v2_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 230.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: false,
        boreSightAtRest: true,
        bgZ: -340.0,
        midZ: -340.0,
        fgZ: -340.0,
        minInnerScale: 1.88,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 155.0,
        initialCameraY: 52.0,
        initialCameraZ: 755.0,
        minCameraZ: -85.0,
        maxCameraZ: 880.0,
        maxYawDeg: 26.0,
        maxPitchDeg: 18.0,
        maxStrafeX: 165.0,
        maxStrafeY: 95.0,
        dollySpeedMult: 1.35,
        midRenderOrder: 6,
        fgRenderOrder: 8,
        particleColor: 0xE6D8C2,
        particleCount: 60,
        hasReflection: false,
        portalShelves: []
      },
      herzliya_tree_canopy: {
        id: 'herzliya_tree_canopy',
        title: 'צמרת העץ בחצר בהרצליה — אור שמש בין הענפים והעלווה',
        bgTexture: `assets/scenes/scene13_herzliya_canopy_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene13_herzliya_canopy_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene13_herzliya_canopy_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene13_herzliya_canopy_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 145.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/scene13_herzliya_canopy_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene13_herzliya_canopy_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 165.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: false,
        fgBleedEdges: { left: true, right: true, top: true, bottom: true, guardMinZ: 210.0 },
        boreSightAtRest: true,
        bgZ: -360.0,
        midZ: -360.0,
        fgZ: -360.0,
        minInnerScale: 2.12,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 112.0,
        initialCameraY: 0.0,
        initialCameraZ: 470.0,
        minCameraZ: -110.0,
        maxCameraZ: 635.0,
        maxYawDeg: 19.0,
        maxPitchDeg: 14.0,
        maxStrafeX: 72.0,
        maxStrafeY: 52.0,
        dollySpeedMult: 1.35,
        midRenderOrder: 2,
        fgRenderOrder: 3,
        particleColor: 0xDDE8C6,
        particleCount: 70,
        hasReflection: false
      },
      bene_beitcha_scroll_foot: {
        id: 'bene_beitcha_scroll_foot',
        title: 'מגילה וכף רגל — תצלום פורים מוצמד מתחת לכף הרגל (׳בנה ביתך׳, 2026)',
        bgTexture: `assets/scenes/scene17_scroll_foot_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene17_scroll_foot_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene17_scroll_foot_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene17_scroll_foot_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 165.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/scene17_scroll_foot_fg.png?${assetVer}`,
        fgExpanded: false,
        boreSightAtRest: true,
        bgZ: -350.0,
        midZ: -350.0,
        fgZ: -180.0,
        minInnerScale: 1.94,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 118.0,
        initialCameraY: -8.0,
        initialCameraZ: 535.0,
        minCameraZ: -90.0,
        maxCameraZ: 695.0,
        maxYawDeg: 23.0,
        maxPitchDeg: 16.0,
        maxStrafeX: 135.0,
        maxStrafeY: 85.0,
        dollySpeedMult: 1.30,
        midRenderOrder: 6,
        fgRenderOrder: 9,
        extraLayers: [
          {
            id: 'pinned_purim_photo',
            labelHe: 'תצלום פורים הישן (תמי ועדה, ~1966) מוצמד מתחת לכף הרגל',
            texture: `assets/scenes/scene17_scroll_foot_photo.png?${assetVer}`,
            depthTexture: `assets/scenes/scene17_scroll_foot_photo_depth.png?${assetVer}`,
            zPos: -350.0,
            depthAmplitude: 165.0,
            pinMarginUv: 0.0,
            segmentsX: 192,
            segmentsY: 128,
            renderOrder: 6.5,
            expanded: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false
          },
          {
            id: 'scroll_black_rod',
            labelHe: 'המוט השחור הנטוי על המגילה',
            texture: `assets/scenes/scene17_scroll_foot_rod.png?${assetVer}`,
            depthTexture: `assets/scenes/scene17_scroll_foot_rod_depth.png?${assetVer}`,
            zPos: -350.0,
            depthAmplitude: 165.0,
            pinMarginUv: 0.0,
            segmentsX: 192,
            segmentsY: 128,
            renderOrder: 7.2,
            expanded: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false
          }
        ],
        particleColor: 0xE4DDD2,
        particleCount: 52,
        hasReflection: false,
        portalShelves: []
      },
      purim_costume_1966: {
        id: 'purim_costume_1966',
        title: 'תמי ועדה בפורים עם סלסלת הקש (~1966)',
        bgTexture: `assets/scenes/scene18_purim_1966_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene18_purim_1966_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene18_purim_1966_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene18_purim_1966_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 165.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/scene18_purim_1966_fg.png?${assetVer}`,
        fgExpanded: false,
        boreSightAtRest: true,
        bgZ: -350.0,
        midZ: -350.0,
        fgZ: -180.0,
        minInnerScale: 1.94,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 108.0,
        initialCameraY: -4.0,
        initialCameraZ: 530.0,
        minCameraZ: -85.0,
        maxCameraZ: 690.0,
        maxYawDeg: 22.0,
        maxPitchDeg: 15.5,
        maxStrafeX: 130.0,
        maxStrafeY: 82.0,
        dollySpeedMult: 1.30,
        midRenderOrder: 6,
        fgRenderOrder: 8,
        particleColor: 0xEAD8BE,
        particleCount: 58,
        hasReflection: false
      },
      herzliya_lawn_carpet: {
        id: 'herzliya_lawn_carpet',
        title: 'דשא הרצליה ו׳הגלגול הבא׳ — טל וגיא מול שטיח האדמה המתגלגל',
        bgTexture: `assets/scenes/scene19_herz_carpet_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene19_herz_carpet_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene19_herz_carpet_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene19_herz_carpet_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 165.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/scene19_herz_carpet_fg.png?${assetVer}`,
        fgExpanded: false,
        boreSightAtRest: true,
        bgZ: -350.0,
        midZ: -350.0,
        fgZ: -180.0,
        minInnerScale: 1.94,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 114.0,
        initialCameraY: -6.0,
        initialCameraZ: 535.0,
        portraitInitialYawDeg: 21.5,
        minCameraZ: -85.0,
        maxCameraZ: 695.0,
        maxYawDeg: 26.0,
        maxPitchDeg: 16.0,
        maxStrafeX: 135.0,
        maxStrafeY: 85.0,
        dollySpeedMult: 1.30,
        midRenderOrder: 6,
        fgRenderOrder: 8,
        carpetUnrollRig: {
          trailLayerId: 'carpet_earth_trail',
          rollLayerId: 'carpet_roll',
          reflectedTrailLayerId: 'carpet_earth_trail_reflection',
          reflectedRollLayerId: 'carpet_roll_reflection',
          clipUvMinX: 0.0,
          initialClipUvMaxX: 0.156667,
          unrolledClipUvMaxX: 0.903333,
          rollShiftInnerPxX: 1080.0,
          rollShiftInnerPxY: 0.0,
          rollShiftWorldZ: 0.0,
          rollTrackWorldZ: -232.235,
          trailTrackWorldZ: -245.835,
          durationMs: 1450
        },
        extraLayers: [
          {
            id: 'carpet_earth_trail',
            labelHe: 'שובל האדמה המתמשך על הדשא — ׳הגלגול הבא׳',
            texture: `assets/scenes/scene19_herz_carpet_earth_full.png?${assetVer}`,
            depthTexture: `assets/scenes/scene19_herz_carpet_earth_full_depth.png?${assetVer}`,
            zPos: -350.0,
            depthAmplitude: 165.0,
            pinMarginUv: 0.0,
            segmentsX: 224,
            segmentsY: 150,
            renderOrder: 7.2,
            expanded: false,
            mirrorLeftEdge: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false,
            clipUvMinX: 0.0,
            clipUvMaxX: 0.156667
          },
          {
            id: 'carpet_roll',
            labelHe: 'גליל השטיח והאדמה — לחצו לגלגול על הדשא',
            texture: `assets/scenes/scene19_herz_carpet_roll.png?${assetVer}`,
            depthTexture: `assets/scenes/scene19_herz_carpet_roll_depth.png?${assetVer}`,
            zPos: -350.0,
            depthAmplitude: 165.0,
            pinMarginUv: 0.0,
            segmentsX: 224,
            segmentsY: 150,
            renderOrder: 7.6,
            expanded: false,
            mirrorLeftEdge: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false
          },
          {
            id: 'carpet_earth_trail_reflection',
            labelHe: 'השתקפות שובל האדמה המתמשך מעבר לשפת המראה השמאלית',
            texture: `assets/scenes/scene19_herz_carpet_earth_full.png?${assetVer}`,
            depthTexture: `assets/scenes/scene19_herz_carpet_earth_full_depth.png?${assetVer}`,
            zPos: -350.0,
            depthAmplitude: 165.0,
            pinMarginUv: 0.0,
            segmentsX: 224,
            segmentsY: 150,
            renderOrder: 7.2,
            expanded: false,
            mirrorLeftEdge: true,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false,
            clipUvMinX: 0.0,
            clipUvMaxX: 0.156667
          },
          {
            id: 'carpet_roll_reflection',
            labelHe: 'השתקפות גליל השטיח מעבר לשפת המראה השמאלית',
            texture: `assets/scenes/scene19_herz_carpet_roll.png?${assetVer}`,
            depthTexture: `assets/scenes/scene19_herz_carpet_roll_depth.png?${assetVer}`,
            zPos: -350.0,
            depthAmplitude: 165.0,
            pinMarginUv: 0.0,
            segmentsX: 224,
            segmentsY: 150,
            renderOrder: 7.6,
            expanded: false,
            mirrorLeftEdge: true,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false
          }
        ],
        particleColor: 0xE8DFC2,
        particleCount: 60,
        hasReflection: false
      },
      vitrine_keys_pen: {
        id: 'vitrine_keys_pen',
        title: 'תיבת הוויטרינה — מדף ורוד, עט וצרור מפתחות (׳בנה ביתך׳, 2026)',
        bgTexture: `assets/scenes/scene20_vitrine_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene20_vitrine_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene20_vitrine_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene20_vitrine_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 175.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/scene20_vitrine_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene20_vitrine_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 195.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: false,
        boreSightAtRest: true,
        bgZ: -350.0,
        midZ: -350.0,
        fgZ: -350.0,
        minInnerScale: 1.94,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 96.0,
        initialCameraY: -4.0,
        initialCameraZ: 525.0,
        minCameraZ: -85.0,
        maxCameraZ: 1400.0,
        maxYawDeg: 22.0,
        maxPitchDeg: 15.5,
        maxStrafeX: 125.0,
        maxStrafeY: 80.0,
        dollySpeedMult: 1.30,
        midRenderOrder: 6,
        fgRenderOrder: 8,
        particleColor: 0xDFD5C5,
        particleCount: 55,
        hasReflection: false
      },
      bar_giora_bedroom: {
        id: 'bar_giora_bedroom',
        title: 'חדר השינה בבר גיורא (חיפה) — וויטרינת המפתחות והעט תלויה מתחת לציור',
        bgTexture: `assets/scenes/scene21_bar_giora_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene21_bar_giora_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene21_bar_giora_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene21_bar_giora_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 175.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/scene21_bar_giora_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene21_bar_giora_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 195.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: true,
        boreSightAtRest: true,
        bgZ: -350.0,
        midZ: -350.0,
        fgZ: -350.0,
        minInnerScale: 2.12,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 118.0,
        initialCameraY: 0.0,
        initialCameraZ: 470.0,
        portraitInitialYawDeg: 7.5,
        minCameraZ: -110.0,
        maxCameraZ: 635.0,
        maxYawDeg: 19.0,
        maxPitchDeg: 14.0,
        maxStrafeX: 72.0,
        maxStrafeY: 48.0,
        dollySpeedMult: 1.28,
        midRenderOrder: 6,
        fgRenderOrder: 8,
        extraLayers: [
          {
            id: 'mounted_wall_vitrine',
            labelHe: 'צל הוויטרינה התלויה צמוד לקיר החדר (מסובבת 130°)',
            texture: `assets/scenes/scene21_bar_giora_vitrine.png?${assetVer}`,
            depthTexture: `assets/scenes/scene21_bar_giora_vitrine_depth.png?${assetVer}`,
            zPos: -350.0,
            depthAmplitude: 175.0,
            pinMarginUv: 0.0,
            segmentsX: 224,
            segmentsY: 150,
            renderOrder: 1.5,
            expanded: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false
          }
        ],
        particleColor: 0xEAE0CF,
        particleCount: 55,
        hasReflection: false,
        portalShelves: []
      },
      paris_cite_white_table: {
        id: 'paris_cite_white_table',
        title: 'הסטודיו ב-Cité des Arts בפריז — תצלום תמי מ-1958 על השולחן הלבן',
        bgTexture: `assets/scenes/scene22_paris_table_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene22_paris_table_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene22_paris_table_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene22_paris_table_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 185.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/scene22_paris_table_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene22_paris_table_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 195.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: false,
        boreSightAtRest: true,
        bgZ: -350.0,
        midZ: -350.0,
        fgZ: -350.0,
        minInnerScale: 1.94,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 118.0,
        initialCameraY: -4.0,
        initialCameraZ: 525.0,
        portraitInitialYawDeg: 7.5,
        minCameraZ: -85.0,
        maxCameraZ: 685.0,
        maxYawDeg: 22.0,
        maxPitchDeg: 15.5,
        maxStrafeX: 125.0,
        maxStrafeY: 80.0,
        dollySpeedMult: 1.30,
        midRenderOrder: 6,
        fgRenderOrder: 8,
        extraLayers: [
          {
            id: 'placed_1958_photo',
            labelHe: 'תצלום תמי הפעוטה מ-1958 מונח על השולחן הלבן — לחצו להתקרבות',
            texture: `assets/scenes/scene22_paris_table_photo.png?${assetVer}`,
            depthTexture: `assets/scenes/scene22_paris_table_photo_depth.png?${assetVer}`,
            zPos: -350.0,
            depthAmplitude: 185.0,
            pinMarginUv: 0.0,
            segmentsX: 224,
            segmentsY: 150,
            renderOrder: 6.5,
            expanded: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false
          }
        ],
        particleColor: 0xE3E6E8,
        particleCount: 55,
        hasReflection: false,
        portalShelves: []
      },
      toddler_tami_1958: {
        id: 'toddler_tami_1958',
        title: 'תמי הפעוטה עומדת (1958) — מבט מקרוב מתוך התצלום המשפחתי',
        bgTexture: `assets/scenes/scene23_tami_1958_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene23_tami_1958_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene23_tami_1958_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene23_tami_1958_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 165.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        midDepthTest: false,
        midDepthWrite: false,
        fgTexture: `assets/scenes/scene23_tami_1958_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene23_tami_1958_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 165.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        fgDepthTest: false,
        fgDepthWrite: false,
        fgExpanded: true,
        boreSightAtRest: true,
        bgZ: -350.0,
        midZ: -350.0,
        fgZ: -350.0,
        minInnerScale: 1.94,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 104.0,
        initialCameraY: -4.0,
        initialCameraZ: 530.0,
        minCameraZ: -85.0,
        maxCameraZ: 690.0,
        maxYawDeg: 22.0,
        maxPitchDeg: 15.5,
        maxStrafeX: 130.0,
        maxStrafeY: 82.0,
        dollySpeedMult: 1.30,
        midRenderOrder: 6,
        fgRenderOrder: 5.8,
        particleColor: 0xE6DFD5,
        particleCount: 52,
        hasReflection: false
      },
      wandering_drawing: {
        id: 'wandering_drawing',
        title: '׳שיטוט 2׳ — רישום איים מרחפים ותבליט דמיוני על נייר',
        bgTexture: `assets/scenes/scene5_wandering_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene5_wandering_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene5_wandering_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene5_wandering_fg.png?${assetVer}`,
        boreSightAtRest: true,
        roomDepthAmplitude: 118.0,
        initialCameraY: 0,
        initialCameraZ: 525,
        maxCameraZ: 780,
        particleColor: 0xD8CFC2,
        particleCount: 65,
        hasReflection: false
      },
      fried_yard_installation: {
        id: 'fried_yard_installation',
        title: 'חצר בית פריד בהרצליה — ׳מרחב לא מוגן 7׳ לפני המעקה הירוק',
        bgTexture: `assets/scenes/scene_railing_occluder_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene_railing_occluder_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene_railing_occluder_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene_railing_occluder_fg.png?${assetVer}`,
        fgExpanded: true,
        boreSightAtRest: true,
        bgZ: -360,
        midZ: -306.04,
        fgZ: -236.18,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 215.0,
        initialCameraY: 0,
        initialCameraZ: 755,
        minCameraZ: 205,
        maxCameraZ: 985,
        maxYawDeg: 34,
        maxPitchDeg: 24,
        maxStrafeX: 255,
        maxStrafeY: 160,
        dollySpeedMult: 1.35,
        midRenderOrder: 3,
        fgRenderOrder: 8,
        extraLayers: [
          {
            id: 'yard_hedge_bank',
            labelHe: 'משוכת הגדר החיה והסבך בחצר בית פריד',
            texture: `assets/scenes/scene_railing_occluder_hedge.png?${assetVer}`,
            depthTexture: `assets/scenes/scene_railing_occluder_hedge_depth.png?${assetVer}`,
            zPos: -356,
            renderOrder: 2,
            expanded: true,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false,
            depthAmplitude: 200.0,
            pinMarginUv: 0.0,
            segmentsX: 192,
            segmentsY: 128
          },
          {
            id: 'green_railing',
            labelHe: 'מעקה הצינור הירוק בשביל הכניסה לבית פריד',
            texture: `assets/scenes/scene_railing_occluder_railing.png?${assetVer}`,
            depthTexture: `assets/scenes/scene_railing_occluder_railing_depth.png?${assetVer}`,
            zPos: -312,
            renderOrder: 4,
            expanded: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false,
            depthAmplitude: 48.0,
            pinMarginUv: 0.0,
            segmentsX: 256,
            segmentsY: 170
          }
        ],
        particleColor: 0xE6DFC6,
        particleCount: 70,
        hasReflection: false,
        portalShelves: []
      },
      fried_house_railing: {
        id: 'fried_house_railing',
        title: 'על המעקה בבית פריד בהרצליה — עם ׳כרית ואבן׳ (2006 / 2021)',
        bgTexture: `assets/scenes/scene10_fried_house_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene10_fried_house_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene10_fried_house_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene10_fried_house_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 148.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        fgTexture: `assets/scenes/scene10_fried_house_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene10_fried_house_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 148.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        boreSightAtRest: true,
        bgZ: -310.0,
        midZ: -278.0,
        fgZ: -276.0,
        bgSegmentsX: 224,
        bgSegmentsY: 150,
        roomDepthAmplitude: 210.0,
        initialCameraY: -56.0,
        initialCameraZ: 865.0,
        minCameraZ: 195.0,
        maxCameraZ: 1050.0,
        maxYawDeg: 34.0,
        maxPitchDeg: 24.0,
        maxStrafeX: 260.0,
        maxStrafeY: 165.0,
        dollySpeedMult: 1.35,
        midRenderOrder: 11.0,
        fgRenderOrder: 11.2,
        extraLayers: [
          {
            id: 'green_handrail',
            labelHe: 'מעקה הצינור הירוק בכניסה לבית פריד',
            texture: `assets/scenes/scene10_fried_house_railing.png?${assetVer}`,
            depthTexture: `assets/scenes/scene10_fried_house_railing_depth.png?${assetVer}`,
            zPos: -274.0,
            renderOrder: 11.4,
            expanded: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false,
            depthAmplitude: 148.0,
            alphaCutout: 0.0,
            pinMarginUv: 0.0,
            segmentsX: 224,
            segmentsY: 150
          }
        ],
        particleColor: 0xEFE2C6,
        particleCount: 65,
        hasReflection: false,
        portalShelves: []
      },
      fried_table_hybrid: {
        id: 'fried_table_hybrid',
        title: 'שולחן האוכל בבית פריד בהרצליה — עם ׳היבריד׳ במהופך (2006 / 2021)',
        bgTexture: `assets/scenes/scene11_fried_table_hybrid_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene11_fried_table_hybrid_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene11_fried_table_hybrid_mid.png?${assetVer}`,
        midDepthTexture: `assets/scenes/scene11_fried_table_hybrid_mid_depth.png?${assetVer}`,
        midDepthAmplitude: 125.0,
        midPinMarginUv: 0.0,
        midSegmentsX: 224,
        midSegmentsY: 150,
        fgTexture: `assets/scenes/scene11_fried_table_hybrid_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene11_fried_table_hybrid_fg_depth.png?${assetVer}`,
        fgDepthAmplitude: 175.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 320,
        fgSegmentsY: 214,
        fgExpanded: true,
        boreSightAtRest: true,
        bgZ: -320.0,
        midZ: -298.0,
        fgZ: -320.0,
        bgSegmentsX: 288,
        bgSegmentsY: 192,
        roomDepthAmplitude: 220.0,
        initialCameraY: 0.0,
        initialCameraZ: 680.0,
        minCameraZ: 185.0,
        maxCameraZ: 960.0,
        maxYawDeg: 32.0,
        maxPitchDeg: 22.0,
        maxStrafeX: 240.0,
        maxStrafeY: 150.0,
        dollySpeedMult: 1.30,
        midRenderOrder: 2,
        fgRenderOrder: 4,
        extraLayers: [
          {
            id: 'table_installation',
            labelHe: 'מיצב ׳היבריד׳ (2021) במהופך לאורך שולחן האוכל בבית פריד',
            texture: `assets/scenes/scene11_fried_table_hybrid_installation.png?${assetVer}`,
            depthTexture: `assets/scenes/scene11_fried_table_hybrid_installation_depth.png?${assetVer}`,
            zPos: -284.0,
            renderOrder: 3,
            expanded: false,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false,
            depthAmplitude: 125.0,
            alphaCutout: 0.0,
            pinMarginUv: 0.0,
            segmentsX: 224,
            segmentsY: 150
          },
          {
            id: 'front_chair',
            labelHe: 'כיסא העץ הקדמי משמאל לשולחן האוכל',
            texture: `assets/scenes/scene11_fried_table_hybrid_front_chair.png?${assetVer}`,
            depthTexture: `assets/scenes/scene11_fried_table_hybrid_front_chair_depth.png?${assetVer}`,
            zPos: -320.0,
            renderOrder: 5,
            expanded: true,
            baseOpacity: 1.0,
            excursionBoost: 0.0,
            depthTest: false,
            depthWrite: false,
            depthAmplitude: 175.0,
            alphaCutout: 0.0,
            pinMarginUv: 0.0,
            segmentsX: 320,
            segmentsY: 214
          }
        ],
        particleColor: 0xEFE4CF,
        particleCount: 60,
        hasReflection: false
      },
      marseille_mirror: {
        id: 'marseille_mirror',
        title: 'תחת תקרת המראה במרסיי — קפסולות ואיים מרחפים',
        bgTexture: `assets/scenes/scene6_marseille_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene6_marseille_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene6_marseille_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene6_marseille_fg.png?${assetVer}`,
        fgDepthTexture: `assets/scenes/scene6_marseille_fg_depth.png?${assetVer}`,
        fgExpanded: true,
        fgBleedEdges: { left: true, right: false, top: true, bottom: true, guardMinZ: 180 },
        boreSightAtRest: true,
        bgZ: -700,
        fgZ: -24,
        fgDepthAmplitude: 24.0,
        fgPinMarginUv: 0.0,
        fgSegmentsX: 224,
        fgSegmentsY: 150,
        minCameraZ: -340,
        maxCameraZ: 1420,
        initialCameraY: -239,
        initialCameraZ: 855,
        maxYawDeg: 58,
        maxPitchDeg: 44,
        maxStrafeX: 560,
        maxStrafeY: 420,
        dollySpeedMult: 1.75,
        midRenderOrder: 7,
        fgRenderOrder: 10,
        extraLayers: [
          {
            id: 'between_chest',
            labelHe: 'קפסולת מפלט מרחפת בחלל שבין הדמויות (מול החזה של גיא)',
            texture: `assets/scenes/scene6_marseille_between_chest.png?${assetVer}`,
            zPos: -28,
            renderOrder: 9,
            expanded: true
          },
          {
            id: 'between_people',
            labelHe: 'מרחב קרוב לצד ראשו של גיא ובין הדמויות',
            texture: `assets/scenes/scene6_marseille_between.png?${assetVer}`,
            zPos: -54,
            renderOrder: 9
          },
          {
            id: 'guy_mid_fg',
            labelHe: 'דמותו של גיא בעומק ביניים (מאחורי קפסולת המפלט)',
            texture: `assets/scenes/scene6_marseille_guy.png?${assetVer}`,
            depthTexture: `assets/scenes/scene6_marseille_guy_depth.png?${assetVer}`,
            zPos: -92,
            depthAmplitude: 36.0,
            pinMarginUv: 0.0,
            segmentsX: 224,
            segmentsY: 150,
            depthTest: false,
            depthWrite: false,
            renderOrder: 8,
            expanded: true
          },
          {
            id: 'upper_mid',
            labelHe: 'מרחב עליון מעל הדמויות (אי רישום מסועף)',
            texture: `assets/scenes/scene6_marseille_upper_mid.png?${assetVer}`,
            zPos: -215,
            renderOrder: 6
          },
          {
            id: 'high_mirror',
            labelHe: 'מרחב גבוה סמוך לתקרת המראה (סירת צריחים מנייר)',
            texture: `assets/scenes/scene6_marseille_high.png?${assetVer}`,
            zPos: -268,
            renderOrder: 5
          },
          {
            id: 'refl_high',
            labelHe: 'השתקפות קרובה במראה (סירת צריחים מנייר)',
            texture: `assets/scenes/scene6_marseille_refl_high.png?${assetVer}`,
            zPos: -355,
            renderOrder: 4,
            baseOpacity: 0.92,
            excursionBoost: 0.08
          },
          {
            id: 'refl_mid',
            labelHe: 'השתקפות אמצעית במראה (אי מסועף וקפסולת אשנב)',
            texture: `assets/scenes/scene6_marseille_refl_mid.png?${assetVer}`,
            zPos: -435,
            renderOrder: 3,
            baseOpacity: 0.92,
            excursionBoost: 0.08
          },
          {
            id: 'refl_between',
            labelHe: 'השתקפות עמוקה במראה (קפסולות ואי בין הדמויות)',
            texture: `assets/scenes/scene6_marseille_refl_between.png?${assetVer}`,
            zPos: -505,
            renderOrder: 2,
            baseOpacity: 0.92,
            excursionBoost: 0.08
          },
          {
            id: 'refl_people',
            labelHe: 'השתקפות תלת-ממדית של הדמויות בכיכר האבן',
            texture: `assets/scenes/scene6_marseille_refl_people.png?${assetVer}`,
            depthTexture: `assets/scenes/scene6_marseille_refl_people_depth.png?${assetVer}`,
            zPos: -675,
            depthAmplitude: 60.0,
            pinMarginUv: 0.0,
            segmentsX: 192,
            segmentsY: 128,
            depthTest: false,
            depthWrite: false,
            renderOrder: 1
          }
        ],
        roomDepthAmplitude: 105.0,
        particleColor: 0xE5DEC9,
        particleCount: 70,
        hasReflection: false
      },
      plan_b_pipe_v1: {
        id: 'plan_b_pipe_v1',
        title: 'Plan B — מבעד לצינור המלט והנייר (גרסה 1 שמורה)',
        bgTexture: `assets/scenes/scene4_plan_b_v1_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene4_plan_b_v1_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene4_plan_b_v1_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene4_plan_b_v1_fg.png?${assetVer}`,
        reflectionTexture: `assets/scenes/scene4_plan_b_v1_reflections.png?${assetVer}`,
        roomDepthAmplitude: 104.0,
        initialCameraY: 0,
        initialCameraZ: 400,
        maxCameraZ: 685,
        particleColor: 0x9FB7CC,
        particleCount: 75,
        hasReflection: true
      },
      marseille_mirror_v1: {
        id: 'marseille_mirror_v1',
        title: 'תחת תקרת המראה במרסיי — גרסה 1 (ארכיון)',
        bgTexture: `assets/scenes/scene6_marseille_v1_bg_inpainted.jpg?${assetVer}`,
        depthTexture: `assets/scenes/scene6_marseille_v1_depth.png?${assetVer}`,
        midTexture: `assets/scenes/scene6_marseille_v1_mid.png?${assetVer}`,
        fgTexture: `assets/scenes/scene6_marseille_v1_fg.png?${assetVer}`,
        reflectionTexture: `assets/scenes/scene6_marseille_v1_reflections.png?${assetVer}`,
        fgExpanded: true,
        boreSightAtRest: true,
        midRenderOrder: 4,
        fgRenderOrder: 6,
        extraLayers: [
          {
            id: 'between_people',
            labelHe: 'מרחב קרוב בין הדמויות (קפסולה ואי רישום)',
            texture: `assets/scenes/scene6_marseille_v1_between.png?${assetVer}`,
            zPos: -70,
            renderOrder: 5
          },
          {
            id: 'upper_mid',
            labelHe: 'מרחב עליון מעל הדמויות (אי רישום מסועף)',
            texture: `assets/scenes/scene6_marseille_v1_upper_mid.png?${assetVer}`,
            zPos: -215,
            renderOrder: 3
          },
          {
            id: 'high_mirror',
            labelHe: 'מרחב גבוה סמוך לתקרת המראה (סירת צריחים מנייר)',
            texture: `assets/scenes/scene6_marseille_v1_high.png?${assetVer}`,
            zPos: -275,
            renderOrder: 2
          }
        ],
        roomDepthAmplitude: 68.0,
        initialCameraY: -185,
        initialCameraZ: 855,
        maxCameraZ: 925,
        reflectionBaseOpacity: 0.92,
        reflectionExcursionBoost: 0.08,
        particleColor: 0xE5DEC9,
        particleCount: 70,
        hasReflection: true
      }
    };

    const rawPortalSpecs =
      Array.isArray(options.portalShelves) && options.portalShelves.length > 0
        ? options.portalShelves
        : SCENE_GEOMETRY_SPEC.shelfPortalShelves;
    this.sceneConfigs.shelf_portal.portalShelves = this._normalizePortalShelfSpecs(
      rawPortalSpecs,
      'shelf_portal'
    );
    const rawYardPortalSpecs =
      Array.isArray(options.yardPortalShelves) && options.yardPortalShelves.length > 0
        ? options.yardPortalShelves
        : SCENE_GEOMETRY_SPEC.yardPortalShelves;
    this.sceneConfigs.fried_yard_installation.portalShelves = this._normalizePortalShelfSpecs(
      rawYardPortalSpecs,
      'fried_yard_installation'
    );
    const rawDenyaLivingPortals =
      Array.isArray(options.denyaLivingPortalShelves) && options.denyaLivingPortalShelves.length > 0
        ? options.denyaLivingPortalShelves
        : SCENE_GEOMETRY_SPEC.denyaLivingPortalShelves;
    this.sceneConfigs.denya_living_room.portalShelves = this._normalizePortalShelfSpecs(
      rawDenyaLivingPortals,
      'denya_living_room'
    );
    const rawDenyaCouchPortals =
      Array.isArray(options.denyaCouchPortalShelves) && options.denyaCouchPortalShelves.length > 0
        ? options.denyaCouchPortalShelves
        : SCENE_GEOMETRY_SPEC.denyaCouchPortalShelves;
    this.sceneConfigs.denya_couch_guitar.portalShelves = this._normalizePortalShelfSpecs(
      rawDenyaCouchPortals,
      'denya_couch_guitar'
    );
    const rawFriedHousePortals =
      Array.isArray(options.friedHousePortalShelves) && options.friedHousePortalShelves.length > 0
        ? options.friedHousePortalShelves
        : SCENE_GEOMETRY_SPEC.friedHousePortalShelves;
    this.sceneConfigs.fried_house_railing.portalShelves = this._normalizePortalShelfSpecs(
      rawFriedHousePortals,
      'fried_house_railing'
    );
    const rawWandering1Portals =
      Array.isArray(options.wandering1PortalShelves) && options.wandering1PortalShelves.length > 0
        ? options.wandering1PortalShelves
        : SCENE_GEOMETRY_SPEC.wandering1PortalShelves;
    this.sceneConfigs.wandering_drawing_1.portalShelves = this._normalizePortalShelfSpecs(
      rawWandering1Portals,
      'wandering_drawing_1'
    );
    const rawTzofenPortals =
      Array.isArray(options.tzofenPortalShelves) && options.tzofenPortalShelves.length > 0
        ? options.tzofenPortalShelves
        : SCENE_GEOMETRY_SPEC.tzofenPortalShelves;
    this.sceneConfigs.tzofen_brick_table.portalShelves = this._normalizePortalShelfSpecs(
      rawTzofenPortals,
      'tzofen_brick_table'
    );
    const rawTzofenV2Portals =
      Array.isArray(options.tzofenV2PortalShelves) && options.tzofenV2PortalShelves.length > 0
        ? options.tzofenV2PortalShelves
        : SCENE_GEOMETRY_SPEC.tzofenV2PortalShelves;
    this.sceneConfigs.tzofen_brick_table_v2.portalShelves = this._normalizePortalShelfSpecs(
      rawTzofenV2Portals,
      'tzofen_brick_table_v2'
    );
    const rawScrollFootPortals =
      Array.isArray(options.scrollFootPortalShelves) && options.scrollFootPortalShelves.length > 0
        ? options.scrollFootPortalShelves
        : SCENE_GEOMETRY_SPEC.scrollFootPortalShelves;
    this.sceneConfigs.bene_beitcha_scroll_foot.portalShelves = this._normalizePortalShelfSpecs(
      rawScrollFootPortals,
      'bene_beitcha_scroll_foot'
    );
    const rawBarGioraPortals =
      Array.isArray(options.barGioraPortalShelves) && options.barGioraPortalShelves.length > 0
        ? options.barGioraPortalShelves
        : SCENE_GEOMETRY_SPEC.barGioraPortalShelves;
    this.sceneConfigs.bar_giora_bedroom.portalShelves = this._normalizePortalShelfSpecs(
      rawBarGioraPortals,
      'bar_giora_bedroom'
    );
    const rawParisTablePortals =
      Array.isArray(options.parisTablePortalShelves) && options.parisTablePortalShelves.length > 0
        ? options.parisTablePortalShelves
        : SCENE_GEOMETRY_SPEC.parisTablePortalShelves;
    this.sceneConfigs.paris_cite_white_table.portalShelves = this._normalizePortalShelfSpecs(
      rawParisTablePortals,
      'paris_cite_white_table'
    );
    this.sceneOrder = Object.keys(this.sceneConfigs);

    let requestedScene = options.initialSceneId || null;
    if (!requestedScene && typeof window !== 'undefined' && window.location) {
      const urlScene = new URLSearchParams(window.location.search).get('scene');
      if (urlScene && this.sceneConfigs[urlScene]) {
        requestedScene = urlScene;
      }
    }
    this.currentSceneId =
      requestedScene && this.sceneConfigs[requestedScene] ? requestedScene : 'shelf_portal';
    this.targetSceneId = null;
    this._pendingTransitionSceneId = null;

    // Additive normalized pointer excursion (kept for subtle hover & automated test harness)
    this.pointer = { x: 0, y: 0, targetX: 0, targetY: 0 };

    // Decoupled 6-DOF camera pose state (world X, Y, Z + Euler YXZ yawDeg, pitchDeg)
    this.baseCameraZ = SCENE_GEOMETRY_SPEC.baseCameraZ;
    const initialRestPose = this._getSceneRestPose(this.currentSceneId);
    this.poseTarget = { ...initialRestPose };
    this.poseCurrent = { ...initialRestPose };

    this._hoverNorm = { x: 0, y: 0, targetX: 0, targetY: 0 };
    this._innerScale = SCENE_GEOMETRY_SPEC.minInnerScale;
    this._layer0Scale = SCENE_GEOMETRY_SPEC.minInnerScale * SCENE_GEOMETRY_SPEC.bgExpansionFactor;

    this.isTransitioning = false;
    this.transitionStartTime = 0;
    this.transitionDuration = SCENE_GEOMETRY_SPEC.transitionDurationMs;
    this.transitionProgress = 0.0;
    this._transitionFromPose = null;
    this._transitionToPose = null;

    // 3D Shelf Portal Flight, Removable Occluders & Hub Return State
    this._portalInstances = [];
    this._portalOwnedResources = [];
    this._erasedPortalOccluders = new Set();
    this._occluderAnimations = new Map();
    this._occluderRegenTimers = new Map();
    this._portalSurroundingOpacity = 1.0;
    this._portalFlight = null;
    this._pendingPortalFlightId = null;
    this.hoveredPortalId = null;
    this.enteredViaPortalId = null;
    this.enteredFromPortalSceneId = null;
    this._portalOriginStack = [];

    this.isPaused = false;
    this.gyroActive = false;
    this.animId = null;
    this.layersReady = false;
    this._lastFrameTimeMs = performance.now();

    this._loadedTextures = 0;
    this._totalTextures = Object.values(this.sceneConfigs).reduce(
      (sum, cfg) =>
        sum +
        4 +
        (cfg.midDepthTexture && typeof cfg.midDepthAmplitude === 'number' ? 1 : 0) +
        (cfg.fgDepthTexture && typeof cfg.fgDepthAmplitude === 'number' ? 1 : 0) +
        (cfg.hasReflection && cfg.reflectionTexture ? 1 : 0) +
        (Array.isArray(cfg.extraLayers)
          ? cfg.extraLayers.reduce((n, el) => n + 1 + (el && el.depthTexture ? 1 : 0), 0)
          : 0) +
        (Array.isArray(cfg.portalShelves)
          ? cfg.portalShelves.reduce(
              (n, ps) =>
                n +
                (ps && ps.occluder && ps.occluder.texture
                  ? 1 + (ps.occluder.depthTexture ? 1 : 0)
                  : 0) +
                (ps && ps.veil && ps.veil.texture ? 1 : 0),
              0
            )
          : 0),
      0
    );
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
    this._mouseMaxTravelPx = 0;
    this._mouseBaseYawDeg = 0;
    this._mouseBasePitchDeg = 0;
    this._mouseBasePosX = 0;
    this._mouseBasePosY = 0;

    this._keysHeld = new Set();
    this._rigSpec = { ...SCENE_GEOMETRY_SPEC };

    // Quaternion-based relative gyroscope state (gimbal-lock free across beta = 90 deg)
    this.gyroPermissionState =
      typeof DeviceOrientationEvent === 'undefined' ? 'unsupported' : 'prompt';
    this._gyroPermissionPending = null;
    this._gyroNoEventsTimer = null;
    this._gyroReferenceQuat = null;
    this._gyroLastValidAlphaDeg = null;
    this._gyroTargetYawDeg = 0;
    this._gyroTargetPitchDeg = 0;
    this._gyroYawDeg = 0;
    this._gyroPitchDeg = 0;
    this._lastScreenAngleDeg = null;
    this._gyroEventsCount = 0;

    this._lastDiagTouchTs = 0;
    this._lastDiagGyroTs = 0;
    this._lastDiagKbBlockedTs = 0;
    this._lastDiagKbDownTs = 0;
    this._lastDiagKbMoveTs = 0;
    this._lastDiagKbClampTs = 0;
    this._lastTouchEndMs = 0;
    this._carpetUnrollStates = new Map();
    this._hoveredCarpetSceneId = null;

    // Collaborative Beta Highlighting, 3D Raycasting & Frame Projection Hooks
    this.interactionCaptureActive = false;
    this._frameListeners = new Set();
    this._textureSamplers = new WeakMap();
    this._raycaster = new THREE.Raycaster();

    const hideCenterHint = () => {
      const hint = document.getElementById('center-hint');
      if (hint && !hint.classList.contains('hidden')) {
        hint.classList.add('hidden');
      }
    };

    const blurActiveUiControlOnCanvasInteraction = () => {
      if (typeof document === 'undefined') return;
      const activeEl = document.activeElement;
      if (
        activeEl &&
        activeEl !== document.body &&
        activeEl !== document.documentElement &&
        activeEl !== this.canvas &&
        typeof activeEl.blur === 'function'
      ) {
        activeEl.blur();
      }
    };

    const isInteractiveTarget = (target) => {
      if (this.interactionCaptureActive) return true;
      if (!target || typeof target.closest !== 'function') return false;
      return !!(
        target.closest('button') ||
        target.closest('input') ||
        target.closest('textarea') ||
        target.closest('select') ||
        target.closest('a') ||
        target.closest('#spatial-sidebar') ||
        target.closest('#info-drawer') ||
        target.closest('#beta-drawer') ||
        target.closest('#spatial-nav-bar') ||
        target.closest('#beta-highlight-banner') ||
        target.closest('#beta-focus-bar') ||
        target.closest('#beta-lightbox') ||
        target.closest('.beta-pin-marker') ||
        target.closest('.exhibition-tabs-row') ||
        target.closest('.scene-pills') ||
        target.closest('.aux-controls') ||
        target.closest('.portal-micro-badge') ||
        target.closest('#portal-hover-plaque')
      );
    };

    this._foldActiveGyroIntoPose = () => {
      if (Math.abs(this._gyroYawDeg) > 1e-3 || Math.abs(this._gyroPitchDeg) > 1e-3) {
        const gyroNormX = this._gyroYawDeg / Math.max(1e-4, this._rigSpec.gyroMaxYawDeg);
        const gyroNormY = this._gyroPitchDeg / Math.max(1e-4, this._rigSpec.gyroMaxPitchDeg);
        const shiftX = -gyroNormX * this._rigSpec.gyroParallaxShiftX;
        const shiftY = gyroNormY * this._rigSpec.gyroParallaxShiftY;
        const foldedTarget = this._clampPose({
          x: this.poseTarget.x + shiftX,
          y: this.poseTarget.y + shiftY,
          z: this.poseTarget.z,
          yawDeg: this.poseTarget.yawDeg + this._gyroYawDeg,
          pitchDeg: this.poseTarget.pitchDeg + this._gyroPitchDeg
        });
        this.poseTarget.x = foldedTarget.x;
        this.poseTarget.y = foldedTarget.y;
        this.poseTarget.yawDeg = foldedTarget.yawDeg;
        this.poseTarget.pitchDeg = foldedTarget.pitchDeg;

        const foldedCurrent = this._clampPose({
          x: this.poseCurrent.x + shiftX,
          y: this.poseCurrent.y + shiftY,
          z: this.poseCurrent.z,
          yawDeg: this.poseCurrent.yawDeg + this._gyroYawDeg,
          pitchDeg: this.poseCurrent.pitchDeg + this._gyroPitchDeg
        });
        this.poseCurrent.x = foldedCurrent.x;
        this.poseCurrent.y = foldedCurrent.y;
        this.poseCurrent.yawDeg = foldedCurrent.yawDeg;
        this.poseCurrent.pitchDeg = foldedCurrent.pitchDeg;
      }
      this._gyroReferenceQuat = null;
      this._gyroTargetYawDeg = 0;
      this._gyroTargetPitchDeg = 0;
      this._gyroYawDeg = 0;
      this._gyroPitchDeg = 0;
    };

    this._applyDollyAlongGaze = (forwardDelta) => {
      if (this._portalFlight) return;
      const effectiveYawRad = THREE.MathUtils.degToRad(this.poseTarget.yawDeg + this._gyroYawDeg);
      const step = this._computeTunnelBoreDollyDelta(forwardDelta, effectiveYawRad);
      const nextRaw = {
        x: this.poseTarget.x + step.dx,
        y: this.poseTarget.y + step.dy,
        z: this.poseTarget.z + step.dz,
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

    const isSyntheticTouchMouse = (ev) => {
      if (this._isTouching) return true;
      if (ev && ev.sourceCapabilities && ev.sourceCapabilities.firesTouchEvents === true) {
        return true;
      }
      if (ev && ev.sourceCapabilities && ev.sourceCapabilities.firesTouchEvents === false) {
        return false;
      }
      return (
        this._lastTouchEndMs > 0 &&
        performance.now() - this._lastTouchEndMs < SCENE_GEOMETRY_SPEC.touchSyntheticMouseGuardMs
      );
    };

    // Desktop Mouse Drag (Rotate / Strafe) + Subtle Ambient Hover + Portal Cubby Raycast
    this._onMouseDown = (e) => {
      if (isSyntheticTouchMouse(e)) {
        return;
      }
      if (isInteractiveTarget(e.target) || this._portalFlight) return;
      if (e.button !== 0 && e.button !== 2) return;
      blurActiveUiControlOnCanvasInteraction();
      this._isMouseDragging = true;
      this._mouseDragMode = (e.button === 2 || e.shiftKey) ? 'strafe' : 'look';
      this._mouseDownX = e.clientX;
      this._mouseDownY = e.clientY;
      this._mouseMaxTravelPx = 0;
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
      if (this._portalFlight) return;
      if (this._isMouseDragging) {
        const dx = e.clientX - this._mouseDownX;
        const dy = e.clientY - this._mouseDownY;
        this._mouseMaxTravelPx = Math.max(this._mouseMaxTravelPx, Math.hypot(dx, dy));
        const activeCfg = this.sceneConfigs[this.targetSceneId || this.currentSceneId] || null;
        const strafeMult =
          activeCfg && typeof activeCfg.dollySpeedMult === 'number' ? activeCfg.dollySpeedMult : 1.0;
        const yawMult =
          activeCfg && typeof activeCfg.maxYawDeg === 'number'
            ? activeCfg.maxYawDeg / SCENE_GEOMETRY_SPEC.maxYawDeg
            : 1.0;
        const pitchMult =
          activeCfg && typeof activeCfg.maxPitchDeg === 'number'
            ? activeCfg.maxPitchDeg / SCENE_GEOMETRY_SPEC.maxPitchDeg
            : 1.0;
        if (this._mouseDragMode === 'strafe') {
          const nextClamped = this._clampPose({
            x: this._mouseBasePosX - dx * SCENE_GEOMETRY_SPEC.mouseStrafeScale * strafeMult,
            y: this._mouseBasePosY + dy * SCENE_GEOMETRY_SPEC.mouseStrafeScale * strafeMult,
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
            yawDeg:
              this._mouseBaseYawDeg + (dx / spanX) * SCENE_GEOMETRY_SPEC.mouseYawSpanDeg * yawMult,
            pitchDeg:
              this._mouseBasePitchDeg +
              (dy / spanY) * SCENE_GEOMETRY_SPEC.mousePitchSpanDeg * pitchMult
          });
          this.poseTarget.yawDeg = nextClamped.yawDeg;
          this.poseTarget.pitchDeg = nextClamped.pitchDeg;
        }
      } else if (!this._isTouching) {
        this._hoverNorm.targetX = (e.clientX / Math.max(1, this.width)) * 2 - 1;
        this._hoverNorm.targetY = -(e.clientY / Math.max(1, this.height)) * 2 + 1;
        this._updatePortalHoverState(e.clientX, e.clientY, e.target);
      }
    };

    this._onMouseUp = (e) => {
      if (isSyntheticTouchMouse(e)) {
        this._isMouseDragging = false;
        return;
      }
      if (!this._isMouseDragging) return;
      const wasClick =
        e &&
        e.button === 0 &&
        this._mouseMaxTravelPx < 8 &&
        !this.interactionCaptureActive &&
        !this.isTransitioning &&
        !this._portalFlight;
      this._isMouseDragging = false;

      if (wasClick && this.hasPortalShelves(this.currentSceneId)) {
        const hitPortal = this._raycastPortalShelfAtClient(e.clientX, e.clientY);
        if (hitPortal) {
          if (this.isPortalLocked(hitPortal)) {
            this.erasePortalOccluder(hitPortal);
            this._updatePortalHoverState(e.clientX, e.clientY, e.target);
          } else {
            this.flyIntoPortalShelf(hitPortal);
          }
          if (this.canvas) {
            this.canvas.style.cursor = this.hoveredPortalId ? 'pointer' : 'grab';
          }
          return;
        }
      }
      if (wasClick && this._raycastCarpetUnrollAtClient(e.clientX, e.clientY)) {
        this.toggleCarpetUnroll(this.currentSceneId);
        hideCenterHint();
        if (this.canvas) {
          this.canvas.style.cursor = 'pointer';
        }
        return;
      }

      if (this.canvas) {
        this.canvas.style.cursor =
          this.hoveredPortalId || this._hoveredCarpetSceneId === this.currentSceneId
            ? 'pointer'
            : 'grab';
      }
    };

    this._onContextMenu = (e) => {
      if (!isInteractiveTarget(e.target)) {
        e.preventDefault();
      }
    };

    // Desktop Scroll Wheel / Trackpad Pinch -> Forward & Backward Dolly
    this._onWheel = (e) => {
      if (isInteractiveTarget(e.target) || this._portalFlight) return;
      if (e.cancelable) e.preventDefault();
      const activeCfg = this.sceneConfigs[this.targetSceneId || this.currentSceneId] || null;
      const speedMult =
        activeCfg && typeof activeCfg.dollySpeedMult === 'number' ? activeCfg.dollySpeedMult : 1.0;
      const scale = e.ctrlKey
        ? SCENE_GEOMETRY_SPEC.trackpadPinchDollyScale
        : SCENE_GEOMETRY_SPEC.wheelDollyScale;
      const forwardDelta = -e.deltaY * scale * speedMult;
      this._applyDollyAlongGaze(forwardDelta);
      hideCenterHint();
    };

    // Mobile Multi-Touch: 1-Finger Look Rotation, 2-Finger Pinch Dolly + Midpoint Strafe, Tap Portal / Double-Tap Reset
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
      if (isInteractiveTarget(e.target) || this._portalFlight) return;
      blurActiveUiControlOnCanvasInteraction();
      if (!this._isTouching && this.gyroActive) {
        this._foldActiveGyroIntoPose();
      }
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
      if (!this._isTouching || this._portalFlight) return;
      if (e.cancelable) e.preventDefault();
      const activeCfg = this.sceneConfigs[this.targetSceneId || this.currentSceneId] || null;
      const speedMult =
        activeCfg && typeof activeCfg.dollySpeedMult === 'number' ? activeCfg.dollySpeedMult : 1.0;
      const yawMult =
        activeCfg && typeof activeCfg.maxYawDeg === 'number'
          ? activeCfg.maxYawDeg / SCENE_GEOMETRY_SPEC.maxYawDeg
          : 1.0;
      const pitchMult =
        activeCfg && typeof activeCfg.maxPitchDeg === 'number'
          ? activeCfg.maxPitchDeg / SCENE_GEOMETRY_SPEC.maxPitchDeg
          : 1.0;

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
          yawDeg:
            this._touchBaseYawDeg + (dx / spanX) * SCENE_GEOMETRY_SPEC.touchYawSpanDeg * yawMult,
          pitchDeg:
            this._touchBasePitchDeg +
            (dy / spanY) * SCENE_GEOMETRY_SPEC.touchPitchSpanDeg * pitchMult
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
        const dollyStep = distDelta * SCENE_GEOMETRY_SPEC.pinchDollyScale * speedMult;
        const strafeRight = -midDx * SCENE_GEOMETRY_SPEC.touchStrafeScale * speedMult;
        const strafeUp = midDy * SCENE_GEOMETRY_SPEC.touchStrafeScale * speedMult;
        const dollyVec = this._computeTunnelBoreDollyDelta(dollyStep, effectiveYawRad);

        const nextClamped = this._clampPose({
          x:
            this.poseTarget.x +
            dollyVec.dx +
            Math.cos(effectiveYawRad) * strafeRight,
          y: this.poseTarget.y + dollyVec.dy + strafeUp,
          z:
            this.poseTarget.z +
            dollyVec.dz -
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
        this._lastTouchEndMs = nowPerf;
        const isSingleTap =
          this._touchMode === 'look' &&
          this._touchMaxTravelPx < 16 &&
          nowPerf - this._touchStartTimeMs < 260;

        this._isTouching = false;
        this._touchMode = 'none';
        // Next gyro event captures resting hand posture as neutral without snapping camera
        this._gyroReferenceQuat = null;
        this._gyroTargetYawDeg = 0;
        this._gyroTargetPitchDeg = 0;

        if (isSingleTap) {
          if (
            this.hasPortalShelves(this.currentSceneId) &&
            !this.interactionCaptureActive &&
            !this.isTransitioning &&
            !this._portalFlight
          ) {
            const hitPortal = this._raycastPortalShelfAtClient(
              this._touchStartX,
              this._touchStartY
            );
            if (hitPortal) {
              this._lastSingleTapMs = 0;
              hideCenterHint();
              if (this.isPortalLocked(hitPortal)) {
                this.erasePortalOccluder(hitPortal);
              } else {
                this.flyIntoPortalShelf(hitPortal);
              }
              return;
            }
          }
          if (
            !this.interactionCaptureActive &&
            !this.isTransitioning &&
            !this._portalFlight &&
            this._raycastCarpetUnrollAtClient(this._touchStartX, this._touchStartY)
          ) {
            this._lastSingleTapMs = 0;
            hideCenterHint();
            this.toggleCarpetUnroll(this.currentSceneId);
            return;
          }
          if (nowPerf - this._lastSingleTapMs < 300) {
            this.resetCameraPose();
            this._lastSingleTapMs = 0;
          } else {
            this._lastSingleTapMs = nowPerf;
          }
        }
      }
    };

    const resolveKeyCode = (e) => {
      if (!e) return '';
      if (typeof e.code === 'string' && e.code.length > 0 && e.code !== 'Unidentified') {
        return e.code;
      }
      const lower = typeof e.key === 'string' ? e.key.toLowerCase() : '';
      const fallbackMap = {
        w: 'KeyW',
        a: 'KeyA',
        s: 'KeyS',
        d: 'KeyD',
        e: 'KeyE',
        q: 'KeyQ',
        r: 'KeyR',
        f: 'KeyF',
        "'": 'KeyW',
        'ש': 'KeyA',
        'ד': 'KeyS',
        'ג': 'KeyD',
        'ק': 'KeyE',
        '/': 'KeyQ',
        'ר': 'KeyR',
        'כ': 'KeyF',
        arrowup: 'ArrowUp',
        arrowdown: 'ArrowDown',
        arrowleft: 'ArrowLeft',
        arrowright: 'ArrowRight',
        pageup: 'PageUp',
        pagedown: 'PageDown',
        ' ': 'Space'
      };
      return fallbackMap[lower] || '';
    };

    this._onDeviceOrientation = (e) => {
      if (!e || !this.gyroActive || this._isTouching || this._portalFlight) return;
      if (!Number.isFinite(e.beta) || !Number.isFinite(e.gamma)) return;

      if (Number.isFinite(e.alpha)) {
        if (this._gyroLastValidAlphaDeg === null && this._gyroEventsCount > 0) {
          this._foldActiveGyroIntoPose();
        }
        this._gyroLastValidAlphaDeg = e.alpha;
      }
      const alpha = this._gyroLastValidAlphaDeg !== null ? this._gyroLastValidAlphaDeg : 0.0;
      const beta = e.beta;
      const gamma = e.gamma;
      const screenAngleDeg = getScreenOrientationAngleDeg();

      if (this._lastScreenAngleDeg !== null && this._lastScreenAngleDeg !== screenAngleDeg) {
        this._foldActiveGyroIntoPose();
      }
      this._lastScreenAngleDeg = screenAngleDeg;

      this._gyroEventsCount += 1;
      const gyroSample = computeGyroAnglesFromDeviceEuler(
        alpha,
        beta,
        gamma,
        screenAngleDeg,
        this._gyroReferenceQuat,
        this._rigSpec
      );
      this._gyroReferenceQuat = gyroSample.referenceQuat;
      this._gyroTargetYawDeg = gyroSample.yawDeg;
      this._gyroTargetPitchDeg = gyroSample.pitchDeg;

      if (
        Math.abs(gyroSample.yawDeg) > this._rigSpec.gyroHintDismissYawDeg ||
        Math.abs(gyroSample.pitchDeg) > this._rigSpec.gyroHintDismissPitchDeg
      ) {
        hideCenterHint();
      }

      const now = Date.now();
      if (window.__sendDiag && now - this._lastDiagGyroTs > 2000) {
        this._lastDiagGyroTs = now;
        window.__sendDiag('gyro_sample', {
          count: this._gyroEventsCount,
          yaw: Number(this._gyroTargetYawDeg.toFixed(1)),
          pitch: Number(this._gyroTargetPitchDeg.toFixed(1)),
          beta: Number(beta.toFixed(1))
        });
      }
    };

    this._onScreenOrientationChange = () => {
      if (this.gyroActive) {
        this._foldActiveGyroIntoPose();
      }
      this._lastScreenAngleDeg = getScreenOrientationAngleDeg();
    };

    this._onVisibilityChange = () => {
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
        this._keysHeld.clear();
        this._isMouseDragging = false;
        this._isTouching = false;
        this._touchMode = 'none';
        if (this.gyroActive) {
          this._foldActiveGyroIntoPose();
        }
      } else {
        this._gyroReferenceQuat = null;
        this._gyroLastValidAlphaDeg = null;
      }
    };

    this._onResize = () => {
      this.width = window.innerWidth;
      this.height = window.innerHeight;
      this.camera.aspect = this.width / this.height;
      this.renderer.setSize(this.width, this.height);
      const dpr = this.renderer ? this.renderer.getPixelRatio() : 1;
      const rtW = Math.max(1, Math.round(this.width * dpr));
      const rtH = Math.max(1, Math.round(this.height * dpr));
      if (this.rtCurrent) this.rtCurrent.setSize(rtW, rtH);
      if (this.rtNext) this.rtNext.setSize(rtW, rtH);
      this.applyDynamicWidescreenScaling();
    };

    this._onKeyDown = (e) => {
      if (e.metaKey || e.ctrlKey) {
        this._keysHeld.clear();
        return;
      }
      if (e.altKey) {
        return;
      }
      const code = resolveKeyCode(e);
      const isNavOrControlKey =
        KEYBOARD_NAV_CODES.has(code) ||
        code === 'KeyR' ||
        code === 'KeyF' ||
        code === 'Space';

      let activeEl = typeof document !== 'undefined' ? document.activeElement : null;
      let activeTag = activeEl && activeEl.tagName ? activeEl.tagName.toUpperCase() : '';
      if (
        activeEl &&
        activeTag !== 'BODY' &&
        activeTag !== 'HTML' &&
        typeof activeEl.blur === 'function'
      ) {
        const insideClosedContainer =
          typeof activeEl.closest === 'function' &&
          Boolean(
            activeEl.closest(
              '#beta-drawer:not(.open), #info-drawer:not(.open), #beta-lightbox.hidden, #spatial-sidebar.collapsed'
            )
          );
        const detachedOrHidden =
          activeEl.isConnected === false ||
          (typeof activeEl.getClientRects === 'function' &&
            activeEl.getClientRects().length === 0);
        if (insideClosedContainer || detachedOrHidden) {
          const staleId = activeEl.id || '';
          const staleTag = activeTag;
          activeEl.blur();
          activeEl = typeof document !== 'undefined' ? document.activeElement : null;
          activeTag = activeEl && activeEl.tagName ? activeEl.tagName.toUpperCase() : '';
          if (window.__sendDiag && isNavOrControlKey) {
            window.__sendDiag('keyboard_stale_focus_recovered', {
              code,
              key: e.key,
              staleTag,
              staleId,
              reason: insideClosedContainer ? 'closed_drawer_or_sidebar' : 'hidden_element',
              scene: this.currentSceneId
            });
          }
        }
      }

      const openModalOrDrawerEl =
        typeof document !== 'undefined'
          ? document.querySelector(
              '#info-drawer.open, #beta-drawer.open, #beta-lightbox:not(.hidden)'
            )
          : null;
      const openModalOrDrawer = Boolean(openModalOrDrawerEl);
      const textInputFocused =
        activeTag === 'INPUT' ||
        activeTag === 'TEXTAREA' ||
        activeTag === 'SELECT' ||
        Boolean(activeEl && activeEl.isContentEditable);

      if (this.interactionCaptureActive || openModalOrDrawer || textInputFocused) {
        if (window.__sendDiag && isNavOrControlKey) {
          const now = Date.now();
          if (now - this._lastDiagKbBlockedTs > 1000) {
            this._lastDiagKbBlockedTs = now;
            window.__sendDiag('keyboard_nav_blocked', {
              code,
              key: e.key,
              reason: this.interactionCaptureActive
                ? 'interaction_capture'
                : openModalOrDrawer
                  ? `open_modal:${openModalOrDrawerEl.id || 'modal'}`
                  : 'text_input_focused',
              activeTag,
              activeId: (activeEl && activeEl.id) || '',
              scene: this.currentSceneId
            });
          }
        }
        return;
      }

      if (e.key >= '1' && e.key <= '9') {
        const sceneIdx = Number(e.key) - 1;
        if (this.sceneOrder[sceneIdx]) {
          this.transitionTo(this.sceneOrder[sceneIdx]);
        }
      }
      if (
        e.key === 'Escape' &&
        this.enteredViaPortalId &&
        this.currentSceneId !== (this.enteredFromPortalSceneId || 'shelf_portal')
      ) {
        this.returnToPortalHub();
        return;
      }
      if (code === 'Space') {
        e.preventDefault();
        this.isPaused = !this.isPaused;
      }
      if (code === 'KeyR') {
        this.resetCameraPose();
      }
      if (code === 'KeyF') {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(() => {});
        } else {
          document.exitFullscreen().catch(() => {});
        }
      }
      if (KEYBOARD_NAV_CODES.has(code)) {
        if (
          code === 'PageUp' ||
          code === 'PageDown' ||
          code.startsWith('Arrow')
        ) {
          e.preventDefault();
        }
        if (this._portalFlight || !this.layersReady) {
          if (window.__sendDiag) {
            const now = Date.now();
            if (now - this._lastDiagKbBlockedTs > 1000) {
              this._lastDiagKbBlockedTs = now;
              window.__sendDiag('keyboard_nav_blocked', {
                code,
                key: e.key,
                reason: !this.layersReady ? 'layers_not_ready' : 'portal_flight_active',
                scene: this.currentSceneId
              });
            }
          }
          return;
        }
        const wasAlreadyHeld = this._keysHeld.has(code);
        this._keysHeld.add(code);
        hideCenterHint();
        if (!wasAlreadyHeld && !e.repeat && window.__sendDiag) {
          const now = Date.now();
          if (now - this._lastDiagKbDownTs > 800) {
            this._lastDiagKbDownTs = now;
            window.__sendDiag('keyboard_nav_keydown', {
              code,
              key: e.key,
              scene: this.currentSceneId,
              isTransitioning: this.isTransitioning,
              pendingScene: this._pendingTransitionSceneId,
              keysHeld: Array.from(this._keysHeld),
              z: Math.round(this.poseTarget.z)
            });
          }
        }
      }
    };

    this._onKeyUp = (e) => {
      const code = resolveKeyCode(e);
      if (code) {
        this._keysHeld.delete(code);
      }
    };

    this._onWindowBlur = () => {
      this._keysHeld.clear();
      this._isMouseDragging = false;
      this._isTouching = false;
      this._touchMode = 'none';
    };

    window.__PROTOTYPE_STATE__ = {
      currentScene: this.currentSceneId,
      cameraPosition: {
        x: this.poseCurrent.x,
        y: this.poseCurrent.y,
        z: this.poseCurrent.z
      },
      cameraRotationDeg: { yaw: 0, pitch: 0, roll: 0 },
      gyroActive: this.gyroActive,
      gyroPermissionState: this.gyroPermissionState,
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

  _computeGyroAnglesFromDeviceEuler(
    alphaDeg,
    betaDeg,
    gammaDeg,
    screenAngleDeg = 0,
    referenceQuat = null
  ) {
    return computeGyroAnglesFromDeviceEuler(
      alphaDeg,
      betaDeg,
      gammaDeg,
      screenAngleDeg,
      referenceQuat
    );
  }

  isPortalHostScene(sceneId = this.currentSceneId) {
    return this.hasPortalShelves(sceneId);
  }

  hasPortalShelves(sceneId = this.currentSceneId) {
    const cfg = this.sceneConfigs && sceneId ? this.sceneConfigs[sceneId] : null;
    return Boolean(cfg && Array.isArray(cfg.portalShelves) && cfg.portalShelves.length > 0);
  }

  _sceneNeedsLocalClipping(sceneId) {
    const cfg = this.sceneConfigs && sceneId ? this.sceneConfigs[sceneId] : null;
    if (!cfg || !Array.isArray(cfg.portalShelves) || cfg.portalShelves.length === 0) {
      return false;
    }
    return cfg.portalShelves.some(
      (spec) => spec && spec.embedDiorama !== false && !spec.wallMount
    );
  }

  _normalizePortalShelfSpecs(rawSpecs, hostSceneId = 'shelf_portal') {
    if (!Array.isArray(rawSpecs)) return [];
    const defaultAspect = this.width / Math.max(1, this.height);
    const defaultFov =
      defaultAspect < 1.0 ? SCENE_GEOMETRY_SPEC.portraitFovDeg : SCENE_GEOMETRY_SPEC.desktopFovDeg;
    const hostCfg = (this.sceneConfigs && this.sceneConfigs[hostSceneId]) || {};
    const hostMinInner =
      typeof hostCfg.minInnerScale === 'number'
        ? Math.max(SCENE_GEOMETRY_SPEC.minInnerScale, hostCfg.minInnerScale)
        : SCENE_GEOMETRY_SPEC.minInnerScale;
    const hostBgZ =
      typeof hostCfg.bgZ === 'number' ? hostCfg.bgZ : SCENE_GEOMETRY_SPEC.bgZ;
    const hostBoreY =
      hostCfg.boreSightAtRest && typeof hostCfg.initialCameraY === 'number'
        ? hostCfg.initialCameraY
        : 0.0;
    const hostBoreZ =
      typeof hostCfg.initialCameraZ === 'number'
        ? hostCfg.initialCameraZ
        : SCENE_GEOMETRY_SPEC.baseCameraZ;

    return rawSpecs
      .filter((spec) => spec && spec.targetSceneId && this.sceneConfigs[spec.targetSceneId])
      .map((spec, idx) => {
        const targetCfg = this.sceneConfigs[spec.targetSceneId];
        const geom = computePortalWorldGeometry(
          spec,
          hostMinInner,
          hostBoreY,
          hostBoreZ,
          hostBgZ,
          defaultFov,
          defaultAspect
        );
        let occluder = null;
        if (spec.occluder && typeof spec.occluder === 'object' && spec.occluder.texture) {
          const rawTex = String(spec.occluder.texture);
          const versionedTex = rawTex.includes('?')
            ? rawTex
            : `${rawTex}?${this._assetVer || 'v=neural32'}`;
          const rawDepthTex = spec.occluder.depthTexture
            ? String(spec.occluder.depthTexture)
            : null;
          const versionedDepthTex = rawDepthTex
            ? rawDepthTex.includes('?')
              ? rawDepthTex
              : `${rawDepthTex}?${this._assetVer || 'v=neural32'}`
            : null;
          const occWorldZ =
            typeof spec.occluder.worldZ === 'number'
              ? spec.occluder.worldZ
              : geom.worldZ +
                (SCENE_GEOMETRY_SPEC.portalOccluderWorldZ -
                  SCENE_GEOMETRY_SPEC.portalDefaultWorldZ);
          const occGeom = Array.isArray(spec.occluder.innerBoxPx)
            ? computePortalWorldGeometry(
                { innerBoxPx: spec.occluder.innerBoxPx, worldZ: occWorldZ, recessDepth: 0 },
                hostMinInner,
                hostBoreY,
                hostBoreZ,
                hostBgZ,
                defaultFov,
                defaultAspect
              )
            : null;
          occluder = {
            texture: versionedTex,
            depthTexture: versionedDepthTex,
            depthAmplitude:
              typeof spec.occluder.depthAmplitude === 'number'
                ? spec.occluder.depthAmplitude
                : 0.0,
            pinMarginUv:
              typeof spec.occluder.pinMarginUv === 'number'
                ? spec.occluder.pinMarginUv
                : SCENE_GEOMETRY_SPEC.depthEdgePinMarginUv,
            segmentsX:
              typeof spec.occluder.segmentsX === 'number' ? spec.occluder.segmentsX : 192,
            segmentsY:
              typeof spec.occluder.segmentsY === 'number' ? spec.occluder.segmentsY : 128,
            worldX: occGeom ? occGeom.worldX : geom.worldX,
            worldY: occGeom ? occGeom.worldY : geom.worldY,
            worldZ: occWorldZ,
            width: occGeom ? occGeom.width : geom.width,
            height: occGeom ? occGeom.height : geom.height,
            renderOrder:
              typeof spec.occluder.renderOrder === 'number'
                ? spec.occluder.renderOrder
                : SCENE_GEOMETRY_SPEC.shelfFgRenderOrder + 1,
            fullInnerCanvas: Boolean(spec.occluder.fullInnerCanvas),
            alphaHitThreshold:
              typeof spec.occluder.alphaHitThreshold === 'number'
                ? spec.occluder.alphaHitThreshold
                : SCENE_GEOMETRY_SPEC.portalOccluderDefaultAlphaHit,
            innerBoxPx: Array.isArray(spec.occluder.innerBoxPx)
              ? [...spec.occluder.innerBoxPx]
              : null,
            labelHe: spec.occluder.labelHe || 'חפץ על המדף',
            coverageRatio:
              typeof spec.occluder.coverageRatio === 'number' ? spec.occluder.coverageRatio : 0.35,
            locksPortal: spec.occluder.locksPortal !== false,
            companionFall:
              spec.occluder.companionFall && typeof spec.occluder.companionFall === 'object'
                ? {
                    extraLayerId: String(spec.occluder.companionFall.extraLayerId || ''),
                    pivotInnerPx: Array.isArray(spec.occluder.companionFall.pivotInnerPx)
                      ? [...spec.occluder.companionFall.pivotInnerPx]
                      : [750, 500],
                    pivotWorldZ:
                      typeof spec.occluder.companionFall.pivotWorldZ === 'number'
                        ? spec.occluder.companionFall.pivotWorldZ
                        : null,
                    fallAngleDeg:
                      typeof spec.occluder.companionFall.fallAngleDeg === 'number'
                        ? spec.occluder.companionFall.fallAngleDeg
                        : -78.0,
                    shiftWorldX:
                      typeof spec.occluder.companionFall.shiftWorldX === 'number'
                        ? spec.occluder.companionFall.shiftWorldX
                        : 0.0,
                    shiftWorldY:
                      typeof spec.occluder.companionFall.shiftWorldY === 'number'
                        ? spec.occluder.companionFall.shiftWorldY
                        : 0.0
                  }
                : null
          };
        }
        let veil = null;
        if (spec.veil && typeof spec.veil === 'object' && spec.veil.texture) {
          const rawVeilTex = String(spec.veil.texture);
          const versionedVeilTex = rawVeilTex.includes('?')
            ? rawVeilTex
            : `${rawVeilTex}?${this._assetVer || 'v=neural32'}`;
          veil = {
            texture: versionedVeilTex,
            worldZ:
              typeof spec.veil.worldZ === 'number' ? spec.veil.worldZ : geom.worldZ + 0.05,
            renderOrder:
              typeof spec.veil.renderOrder === 'number' ? spec.veil.renderOrder : 6.2,
            fullInnerCanvas: spec.veil.fullInnerCanvas !== false,
            restOpacity:
              typeof spec.veil.restOpacity === 'number' ? spec.veil.restOpacity : 1.0,
            hoverOpacity:
              typeof spec.veil.hoverOpacity === 'number' ? spec.veil.hoverOpacity : 0.56,
            dissolveEndProgress:
              typeof spec.veil.dissolveEndProgress === 'number'
                ? spec.veil.dissolveEndProgress
                : 0.58
          };
        }
        const isDirectCrossfadeOrReverse =
          spec.portalMode === 'crossfade' || spec.portalMode === 'reverse_transition';
        const rotationDeg = typeof spec.rotationDeg === 'number' ? spec.rotationDeg : 0.0;
        const dioramaRollDeg = typeof spec.dioramaRollDeg === 'number' ? spec.dioramaRollDeg : 0.0;
        const flightRollDeg = typeof spec.flightRollDeg === 'number' ? spec.flightRollDeg : 0.0;
        const isPortrait = defaultAspect < 1.0;
        const arrivalYawDeg =
          isPortrait && typeof targetCfg.portraitInitialYawDeg === 'number'
            ? targetCfg.portraitInitialYawDeg
            : 0.0;
        const effectiveFitZ = computePortalEffectiveFitZ(
          geom,
          defaultFov,
          defaultAspect,
          arrivalYawDeg,
          0.0,
          flightRollDeg,
          rotationDeg
        );
        const defaultRestTint =
          hostSceneId === 'shelf_portal'
            ? SCENE_GEOMETRY_SPEC.shelfCubbyRestTintRgb
            : SCENE_GEOMETRY_SPEC.neutralTintRgb;
        const dioramaRestTintRgb =
          Array.isArray(spec.dioramaRestTintRgb) && spec.dioramaRestTintRgb.length === 3
            ? [
                Number(spec.dioramaRestTintRgb[0]),
                Number(spec.dioramaRestTintRgb[1]),
                Number(spec.dioramaRestTintRgb[2])
              ]
            : [...defaultRestTint];
        const keepHostBackdropOpaque = Boolean(spec.keepHostBackdropOpaque);
        const outerWallFadeRangeT =
          Array.isArray(spec.outerWallFadeRangeT) && spec.outerWallFadeRangeT.length === 2
            ? [Number(spec.outerWallFadeRangeT[0]), Number(spec.outerWallFadeRangeT[1])]
            : null;
        let wallMount = null;
        if (spec.wallMount && typeof spec.wallMount === 'object') {
          const wm = spec.wallMount;
          const orders = wm.dioramaRenderOrders || {};
          wallMount = {
            hostCenterInnerPx: Array.isArray(wm.hostCenterInnerPx)
              ? [Number(wm.hostCenterInnerPx[0]), Number(wm.hostCenterInnerPx[1])]
              : [750.0, 500.0],
            hostBoxSidePx: typeof wm.hostBoxSidePx === 'number' ? wm.hostBoxSidePx : 118.0,
            hostWallZ: typeof wm.hostWallZ === 'number' ? wm.hostWallZ : geom.worldZ,
            hostWallSlopeDzDx:
              typeof wm.hostWallSlopeDzDx === 'number' ? wm.hostWallSlopeDzDx : 0.0,
            hostWallSlopeDzDy:
              typeof wm.hostWallSlopeDzDy === 'number' ? wm.hostWallSlopeDzDy : 0.0,
            hostWallClearanceWorld:
              typeof wm.hostWallClearanceWorld === 'number' ? wm.hostWallClearanceWorld : 0.08,
            hostRefInnerScale:
              typeof wm.hostRefInnerScale === 'number' ? wm.hostRefInnerScale : hostMinInner,
            targetBoxCenterInnerPx: Array.isArray(wm.targetBoxCenterInnerPx)
              ? [Number(wm.targetBoxCenterInnerPx[0]), Number(wm.targetBoxCenterInnerPx[1])]
              : [750.0, 500.0],
            targetBoxSidePx: typeof wm.targetBoxSidePx === 'number' ? wm.targetBoxSidePx : 864.0,
            targetPlasterDepthNorm:
              typeof wm.targetPlasterDepthNorm === 'number' ? wm.targetPlasterDepthNorm : 0.2,
            cavityInnerBoxPx:
              Array.isArray(wm.cavityInnerBoxPx) && wm.cavityInnerBoxPx.length === 4
                ? wm.cavityInnerBoxPx.map(Number)
                : [352.0, 120.0, 1148.0, 895.0],
            cavityFeatherPx: typeof wm.cavityFeatherPx === 'number' ? wm.cavityFeatherPx : 4.0,
            hostLayerId: typeof wm.hostLayerId === 'string' ? wm.hostLayerId : null,
            pointPairsInnerPx: Array.isArray(wm.pointPairsInnerPx)
              ? wm.pointPairsInnerPx.map((pair) => [
                  [Number(pair[0][0]), Number(pair[0][1])],
                  [Number(pair[1][0]), Number(pair[1][1])]
                ])
              : null,
            flatAtRest: Boolean(wm.flatAtRest),
            inPlaneRollDeg:
              typeof wm.inPlaneRollDeg === 'number' ? wm.inPlaneRollDeg : rotationDeg,
            dioramaRenderOrders: {
              bg: typeof orders.bg === 'number' ? orders.bg : 2.0,
              mid: typeof orders.mid === 'number' ? orders.mid : 3.0,
              fg: typeof orders.fg === 'number' ? orders.fg : 4.0,
              dust: typeof orders.dust === 'number' ? orders.dust : 4.5
            }
          };
        }
        return {
          id: spec.id || `portal_shelf_${idx + 1}`,
          hostSceneId,
          targetSceneId: spec.targetSceneId,
          labelHe: spec.labelHe || targetCfg.title || spec.targetSceneId,
          shortBadgeHe: typeof spec.shortBadgeHe === 'string' ? spec.shortBadgeHe : null,
          showMicroBadge: spec.showMicroBadge !== false,
          badgeOffsetXFactor:
            typeof spec.badgeOffsetXFactor === 'number' ? spec.badgeOffsetXFactor : undefined,
          badgeOffsetYFactor:
            typeof spec.badgeOffsetYFactor === 'number' ? spec.badgeOffsetYFactor : undefined,
          portalMode: typeof spec.portalMode === 'string' ? spec.portalMode : undefined,
          embedDiorama:
            typeof spec.embedDiorama === 'boolean'
              ? spec.embedDiorama
              : !isDirectCrossfadeOrReverse,
          keepHostBackdropOpaque,
          outerWallFadeRangeT,
          wallMount,
          showFrame:
            typeof spec.showFrame === 'boolean' ? spec.showFrame : !isDirectCrossfadeOrReverse,
          restFrameOpacity:
            typeof spec.restFrameOpacity === 'number'
              ? spec.restFrameOpacity
              : hostSceneId === 'shelf_portal'
                ? 0.24
                : 0.0,
          frameRenderOrder:
            typeof spec.frameRenderOrder === 'number'
              ? spec.frameRenderOrder
              : hostSceneId === 'shelf_portal'
                ? SCENE_GEOMETRY_SPEC.shelfFgRenderOrder - 1
                : 6.5,
          fadeStartProgress:
            typeof spec.fadeStartProgress === 'number' ? spec.fadeStartProgress : undefined,
          rotationDeg,
          dioramaRollDeg,
          flightRollDeg,
          dioramaRestTintRgb,
          innerBoxPx: Array.isArray(spec.innerBoxPx) ? [...spec.innerBoxPx] : null,
          worldX: Number(geom.worldX.toFixed(3)),
          worldY: Number(geom.worldY.toFixed(3)),
          worldZ: Number(geom.worldZ.toFixed(3)),
          width: Number(geom.width.toFixed(3)),
          height: Number(geom.height.toFixed(3)),
          recessDepth: Number(geom.recessDepth.toFixed(3)),
          fitZ: Number(geom.fitZ.toFixed(3)),
          effectiveFitZ: Number(effectiveFitZ.toFixed(3)),
          heightFitZ: Number(geom.heightFitZ.toFixed(3)),
          widthFitZ: Number(geom.widthFitZ.toFixed(3)),
          occluder,
          veil,
          previewBgTex: spec.previewBgTex || targetCfg.bgTexture,
          previewBgDepth: spec.previewBgDepth || targetCfg.depthTexture,
          previewMidTex: spec.previewMidTex || targetCfg.midTexture,
          previewMidDepth: spec.previewMidDepth || null,
          previewFgTex: spec.previewFgTex || targetCfg.fgTexture,
          previewFgDepth: spec.previewFgDepth || null
        };
      });
  }

  _canonicalizePortalSpecsForComparison(specs) {
    if (!Array.isArray(specs)) return [];
    return specs.map((s) => ({
      id: s.id,
      hostSceneId: s.hostSceneId,
      targetSceneId: s.targetSceneId,
      portalMode: s.portalMode,
      embedDiorama: s.embedDiorama,
      keepHostBackdropOpaque: Boolean(s.keepHostBackdropOpaque),
      outerWallFadeRangeT: s.outerWallFadeRangeT || null,
      wallMount: s.wallMount || null,
      showFrame: s.showFrame,
      restFrameOpacity: s.restFrameOpacity,
      frameRenderOrder: s.frameRenderOrder,
      fadeStartProgress: s.fadeStartProgress,
      rotationDeg: s.rotationDeg,
      dioramaRollDeg: s.dioramaRollDeg,
      flightRollDeg: s.flightRollDeg,
      dioramaRestTintRgb: s.dioramaRestTintRgb,
      innerBoxPx: s.innerBoxPx,
      worldZ: s.worldZ,
      recessDepth: s.recessDepth,
      occluder: s.occluder
        ? {
            texture: s.occluder.texture,
            depthTexture: s.occluder.depthTexture,
            depthAmplitude: s.occluder.depthAmplitude,
            pinMarginUv: s.occluder.pinMarginUv,
            segmentsX: s.occluder.segmentsX,
            segmentsY: s.occluder.segmentsY,
            worldZ: s.occluder.worldZ,
            renderOrder: s.occluder.renderOrder,
            fullInnerCanvas: s.occluder.fullInnerCanvas,
            alphaHitThreshold: s.occluder.alphaHitThreshold,
            innerBoxPx: s.occluder.innerBoxPx,
            locksPortal: s.occluder.locksPortal,
            companionFall: s.occluder.companionFall
          }
        : null,
      veil: s.veil,
      previewBgTex: s.previewBgTex,
      previewBgDepth: s.previewBgDepth,
      previewMidTex: s.previewMidTex,
      previewMidDepth: s.previewMidDepth,
      previewFgTex: s.previewFgTex,
      previewFgDepth: s.previewFgDepth
    }));
  }

  _getSceneRestPose(sceneId) {
    const cfg = this.sceneConfigs && sceneId ? this.sceneConfigs[sceneId] : null;
    const isPortrait = this.width / Math.max(1, this.height) < 1.0;
    const yawDeg =
      isPortrait && cfg && typeof cfg.portraitInitialYawDeg === 'number'
        ? cfg.portraitInitialYawDeg
        : 0;
    return {
      x: 0,
      y: cfg && typeof cfg.initialCameraY === 'number' ? cfg.initialCameraY : 0,
      z: cfg && typeof cfg.initialCameraZ === 'number' ? cfg.initialCameraZ : this.baseCameraZ,
      yawDeg,
      pitchDeg: 0,
      rollDeg: 0
    };
  }

  _computeTunnelBoreAxisAtZ(sceneId, camZ) {
    const restPose = this._getSceneRestPose(sceneId);
    const activeCfg = this.sceneConfigs && sceneId ? this.sceneConfigs[sceneId] : null;
    const activeObj = this.sceneObjects && sceneId ? this.sceneObjects[sceneId] : null;
    const sceneMinZ =
      activeCfg && typeof activeCfg.minCameraZ === 'number'
        ? activeCfg.minCameraZ
        : SCENE_GEOMETRY_SPEC.minCameraZ;
    if (!activeCfg || !activeCfg.tunnelBoreTarget || !activeObj) {
      return {
        hasTunnel: false,
        centerX: 0.0,
        centerY: restPose.y,
        strafeScale: 1.0,
        targetX: 0.0,
        targetY: restPose.y,
        arrivalZ: sceneMinZ,
        spanZ: SCENE_GEOMETRY_SPEC.tunnelMinSpanZ
      };
    }

    const boreZ = activeCfg.boreSightAtRest ? restPose.z : this.baseCameraZ;
    const boreY = activeCfg.boreSightAtRest ? restPose.y : 0.0;
    const boreDistBg = Math.max(1.0, boreZ - activeObj.sceneBgZ);
    const tRatio = (boreZ - activeCfg.tunnelBoreTarget.targetZ) / boreDistBg;
    const planeW = SCENE_GEOMETRY_SPEC.bgBaseWidth * activeObj.innerScale * tRatio;
    const planeH = SCENE_GEOMETRY_SPEC.bgBaseHeight * activeObj.innerScale * tRatio;
    const targetX = (activeCfg.tunnelBoreTarget.innerNormX - 0.5) * planeW;
    const targetY = boreY * (1.0 - tRatio) + (0.5 - activeCfg.tunnelBoreTarget.innerNormY) * planeH;
    const arrivalZ = Math.max(
      activeCfg.tunnelBoreTarget.targetZ + SCENE_GEOMETRY_SPEC.tunnelArrivalClearanceZ,
      sceneMinZ
    );
    const spanZ = Math.min(SCENE_GEOMETRY_SPEC.tunnelMinSpanZ, arrivalZ - restPose.z);
    const tBore = THREE.MathUtils.clamp(
      (restPose.z - camZ) / Math.max(1.0, restPose.z - arrivalZ),
      0.0,
      1.0
    );
    const strafeScale =
      1.0 - (1.0 - SCENE_GEOMETRY_SPEC.tunnelMinStrafeScale) * tBore;
    return {
      hasTunnel: true,
      centerX: targetX * tBore,
      centerY: restPose.y + (targetY - restPose.y) * tBore,
      strafeScale,
      targetX,
      targetY,
      arrivalZ,
      spanZ
    };
  }

  _computeTunnelBoreDollyDelta(forwardDelta, effectiveYawRad) {
    const rawDx = -Math.sin(effectiveYawRad) * forwardDelta;
    const rawDz = -Math.cos(effectiveYawRad) * forwardDelta;
    const activeId = this.targetSceneId || this.currentSceneId;
    const bore = this._computeTunnelBoreAxisAtZ(activeId, this.poseTarget.z);
    if (!bore.hasTunnel) {
      return { dx: rawDx, dy: 0, dz: rawDz };
    }

    const activeCfg = this.sceneConfigs[activeId];
    const restPose = this._getSceneRestPose(activeId);
    const sceneMinZ =
      typeof activeCfg.minCameraZ === 'number'
        ? activeCfg.minCameraZ
        : SCENE_GEOMETRY_SPEC.minCameraZ;
    const activeMaxZ = this._getActiveMaxCameraZ();
    const clampedNextZ = THREE.MathUtils.clamp(this.poseTarget.z + rawDz, sceneMinZ, activeMaxZ);
    const tunnelZBefore = THREE.MathUtils.clamp(this.poseTarget.z, bore.arrivalZ, restPose.z);
    const tunnelZAfter = THREE.MathUtils.clamp(clampedNextZ, bore.arrivalZ, restPose.z);
    const tunnelDz = tunnelZAfter - tunnelZBefore;

    const boreDx = (bore.targetX / bore.spanZ) * tunnelDz;
    const boreDy = ((bore.targetY - restPose.y) / bore.spanZ) * tunnelDz;
    return {
      dx: rawDx + boreDx,
      dy: boreDy,
      dz: rawDz
    };
  }

  _getActiveMaxCameraZ() {
    const curCfg = this.sceneConfigs[this.currentSceneId];
    const curMax =
      curCfg && typeof curCfg.maxCameraZ === 'number'
        ? curCfg.maxCameraZ
        : SCENE_GEOMETRY_SPEC.maxCameraZ;
    if (this.targetSceneId && this.sceneConfigs[this.targetSceneId]) {
      const nextCfg = this.sceneConfigs[this.targetSceneId];
      const nextMax =
        typeof nextCfg.maxCameraZ === 'number'
          ? nextCfg.maxCameraZ
          : SCENE_GEOMETRY_SPEC.maxCameraZ;
      return Math.max(curMax, nextMax);
    }
    return curMax;
  }

  _clampPoseForScene(sceneId, rawPose, maxCameraZOverride = null) {
    const activeCfg = this.sceneConfigs && sceneId ? this.sceneConfigs[sceneId] : null;
    const activeObj = this.sceneObjects && sceneId ? this.sceneObjects[sceneId] : null;
    const activeLayer0Scale =
      activeObj && typeof activeObj.layer0Scale === 'number'
        ? activeObj.layer0Scale
        : this._layer0Scale;
    const sceneBgZ =
      activeCfg && typeof activeCfg.bgZ === 'number' ? activeCfg.bgZ : SCENE_GEOMETRY_SPEC.bgZ;
    const sceneMinZ =
      activeCfg && typeof activeCfg.minCameraZ === 'number'
        ? activeCfg.minCameraZ
        : SCENE_GEOMETRY_SPEC.minCameraZ;
    const sceneMaxZ =
      typeof maxCameraZOverride === 'number'
        ? maxCameraZOverride
        : activeCfg && typeof activeCfg.maxCameraZ === 'number'
          ? activeCfg.maxCameraZ
          : SCENE_GEOMETRY_SPEC.maxCameraZ;
    const sceneMaxYaw =
      activeCfg && typeof activeCfg.maxYawDeg === 'number'
        ? activeCfg.maxYawDeg
        : SCENE_GEOMETRY_SPEC.maxYawDeg;
    const sceneMaxPitch =
      activeCfg && typeof activeCfg.maxPitchDeg === 'number'
        ? activeCfg.maxPitchDeg
        : SCENE_GEOMETRY_SPEC.maxPitchDeg;
    const sceneMaxStrafeX =
      activeCfg && typeof activeCfg.maxStrafeX === 'number'
        ? activeCfg.maxStrafeX
        : SCENE_GEOMETRY_SPEC.maxStrafeX;
    const sceneMaxStrafeY =
      activeCfg && typeof activeCfg.maxStrafeY === 'number'
        ? activeCfg.maxStrafeY
        : SCENE_GEOMETRY_SPEC.maxStrafeY;
    const prelimZ = THREE.MathUtils.clamp(rawPose.z, sceneMinZ, sceneMaxZ);
    const bore = this._computeTunnelBoreAxisAtZ(sceneId, prelimZ);
    return clampPoseToRoomFrustum(
      rawPose,
      this.camera ? this.camera.fov : SCENE_GEOMETRY_SPEC.desktopFovDeg,
      this.camera ? this.camera.aspect : this.width / Math.max(1, this.height),
      activeLayer0Scale,
      {
        ...SCENE_GEOMETRY_SPEC,
        bgZ: sceneBgZ,
        minCameraZ: sceneMinZ,
        restCameraX: bore.centerX,
        restCameraY: bore.centerY,
        maxCameraZ: sceneMaxZ,
        maxYawDeg: sceneMaxYaw,
        maxPitchDeg: sceneMaxPitch,
        maxStrafeX: sceneMaxStrafeX * bore.strafeScale,
        maxStrafeY: sceneMaxStrafeY * bore.strafeScale,
        fgBleedBounds: activeObj && activeObj.fgBleedBounds ? activeObj.fgBleedBounds : null
      }
    );
  }

  _clampPose(rawPose) {
    const activeId = this.targetSceneId || this.currentSceneId;
    return this._clampPoseForScene(activeId, rawPose, this._getActiveMaxCameraZ());
  }

  setCameraPose(partialPose = {}, immediate = false) {
    const merged = {
      x: typeof partialPose.x === 'number' ? partialPose.x : this.poseTarget.x,
      y: typeof partialPose.y === 'number' ? partialPose.y : this.poseTarget.y,
      z: typeof partialPose.z === 'number' ? partialPose.z : this.poseTarget.z,
      yawDeg: typeof partialPose.yawDeg === 'number' ? partialPose.yawDeg : this.poseTarget.yawDeg,
      pitchDeg:
        typeof partialPose.pitchDeg === 'number' ? partialPose.pitchDeg : this.poseTarget.pitchDeg,
      rollDeg:
        typeof partialPose.rollDeg === 'number'
          ? partialPose.rollDeg
          : this.poseTarget.rollDeg || 0
    };
    const clamped = this._clampPose(merged);
    this.poseTarget.x = clamped.x;
    this.poseTarget.y = clamped.y;
    this.poseTarget.z = clamped.z;
    this.poseTarget.yawDeg = clamped.yawDeg;
    this.poseTarget.pitchDeg = clamped.pitchDeg;
    this.poseTarget.rollDeg = clamped.rollDeg || 0;

    if (immediate) {
      this.poseCurrent.x = clamped.x;
      this.poseCurrent.y = clamped.y;
      this.poseCurrent.z = clamped.z;
      this.poseCurrent.yawDeg = clamped.yawDeg;
      this.poseCurrent.pitchDeg = clamped.pitchDeg;
      this.poseCurrent.rollDeg = clamped.rollDeg || 0;
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
    this._gyroReferenceQuat = null;
    this._gyroTargetYawDeg = 0;
    this._gyroTargetPitchDeg = 0;
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
    const activeId = this.targetSceneId || this.currentSceneId;
    return this.setCameraPose(this._getSceneRestPose(activeId), immediate);
  }

  initRenderer() {
    THREE.ColorManagement.enabled = false;
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
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.localClippingEnabled = this._sceneNeedsLocalClipping(this.currentSceneId);
    this.renderer.setClearColor(0x121314, 1.0);
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    const aspect = this.width / this.height;
    const initialFov =
      aspect < 1.0 ? SCENE_GEOMETRY_SPEC.portraitFovDeg : SCENE_GEOMETRY_SPEC.desktopFovDeg;
    this.camera = new THREE.PerspectiveCamera(
      initialFov,
      aspect,
      SCENE_GEOMETRY_SPEC.cameraNearPlane,
      SCENE_GEOMETRY_SPEC.cameraFarPlane
    );
    this.camera.rotation.order = 'YXZ';
    this.camera.position.set(this.poseCurrent.x, this.poseCurrent.y, this.poseCurrent.z);
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

  _loadTextureIntoSlot(url, targetTex, onSlotDone, isInitialScene, onSlotError = null) {
    this.textureLoader.load(
      url,
      (loadedTex) => {
        targetTex.image = loadedTex.image;
        targetTex.needsUpdate = true;
        this._loadedTextures += 1;
        if (isInitialScene && !this.layersReady && this.onLoadingProgress) {
          const initObj = this.sceneObjects[this.currentSceneId];
          const initLoaded = initObj ? initObj.loadedCount + 1 : this._loadedTextures;
          const initTotal = initObj ? initObj.expectedCount : 5;
          this.onLoadingProgress(initLoaded, initTotal);
        }
        if (window.__sendDiag) {
          window.__sendDiag('texture_loaded', {
            url: url,
            loaded: this._loadedTextures,
            total: this._totalTextures,
            w: loadedTex.image ? loadedTex.image.width : 0,
            h: loadedTex.image ? loadedTex.image.height : 0
          });
        }
        onSlotDone();
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
        if (typeof onSlotError === 'function') {
          onSlotError(new Error(msg));
        }
      }
    );
  }

  _getOrCreateSharedUrlTextureEntry(url) {
    if (!this._sharedUrlTextures) {
      this._sharedUrlTextures = new Map();
    }
    let entry = this._sharedUrlTextures.get(url);
    if (!entry) {
      const tex = new THREE.Texture();
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      entry = {
        url,
        tex,
        loaded: false,
        promise: null
      };
      this._sharedUrlTextures.set(url, entry);
    }
    return entry;
  }

  _loadSharedUrlTexture(url, isInitialScene = false) {
    if (!url) return Promise.resolve(null);
    const entry = this._getOrCreateSharedUrlTextureEntry(url);
    if (entry.loaded) return Promise.resolve(entry.tex);
    if (entry.promise) return entry.promise;
    entry.promise = new Promise((resolve, reject) => {
      this._loadTextureIntoSlot(
        url,
        entry.tex,
        () => {
          entry.loaded = true;
          resolve(entry.tex);
        },
        isInitialScene,
        (err) => {
          entry.promise = null;
          reject(err);
        }
      );
    });
    return entry.promise;
  }

  _loadSceneTextures(sceneId, isInitialScene = false) {
    const obj = this.sceneObjects[sceneId];
    const cfg = this.sceneConfigs[sceneId];
    if (!obj || !cfg) return Promise.resolve();
    if (obj.texturesReady) return Promise.resolve();
    if (obj.loadPromise) return obj.loadPromise;

    const slots = [
      { url: cfg.bgTexture, tex: obj.bgTex },
      { url: cfg.depthTexture, tex: obj.depthTex },
      { url: cfg.midTexture, tex: obj.midTex },
      { url: cfg.fgTexture, tex: obj.fgTex }
    ];
    if (obj.hasMidDepthRelief && cfg.midDepthTexture && obj.midDepthTex) {
      slots.push({ url: cfg.midDepthTexture, tex: obj.midDepthTex });
    }
    if (obj.hasFgDepthRelief && cfg.fgDepthTexture && obj.fgDepthTex) {
      slots.push({ url: cfg.fgDepthTexture, tex: obj.fgDepthTex });
    }
    if (cfg.hasReflection && cfg.reflectionTexture && obj.refTex) {
      slots.push({ url: cfg.reflectionTexture, tex: obj.refTex });
    }
    if (Array.isArray(obj.extraLayers)) {
      for (const extra of obj.extraLayers) {
        slots.push({ url: extra.textureUrl, tex: extra.tex });
        if (extra.depthTextureUrl && extra.depthTex) {
          slots.push({ url: extra.depthTextureUrl, tex: extra.depthTex });
        }
      }
    }
    const sharedUrls = new Set();
    for (const inst of this._portalInstances) {
      if (inst.hostSceneId !== sceneId) continue;
      if (inst.spec.occluder && inst.spec.occluder.texture) {
        sharedUrls.add(inst.spec.occluder.texture);
      }
      if (inst.spec.occluder && inst.spec.occluder.depthTexture) {
        sharedUrls.add(inst.spec.occluder.depthTexture);
      }
      if (inst.spec.veil && inst.spec.veil.texture) {
        sharedUrls.add(inst.spec.veil.texture);
      }
      for (const cp of inst.childPortalMeshes || []) {
        const cs = cp.childSpec;
        if (cs && cs.occluder && cs.occluder.texture) {
          sharedUrls.add(cs.occluder.texture);
        }
        if (cs && cs.occluder && cs.occluder.depthTexture) {
          sharedUrls.add(cs.occluder.depthTexture);
        }
        if (cs && cs.veil && cs.veil.texture) {
          sharedUrls.add(cs.veil.texture);
        }
      }
    }
    const sharedUrlList = Array.from(sharedUrls);
    obj.expectedCount = slots.length + sharedUrlList.length;
    obj.loadedCount = 0;

    obj.loadPromise = new Promise((resolve, reject) => {
      let failed = false;
      const handleSlotDone = () => {
        if (failed) return;
        obj.loadedCount += 1;
        if (obj.loadedCount >= obj.expectedCount) {
          obj.texturesReady = true;
          if (!this._portalFlight) {
            let anyDioramaRevealed = false;
            for (const inst of this._portalInstances) {
              if (inst.targetSceneId === sceneId && inst.embedDiorama && inst.dioramaGroup) {
                const hostOp =
                  this.sceneObjects[inst.hostSceneId] &&
                  typeof this.sceneObjects[inst.hostSceneId].portalSurroundingOpacity === 'number'
                    ? this.sceneObjects[inst.hostSceneId].portalSurroundingOpacity
                    : 1.0;
                inst.dioramaGroup.visible = hostOp > 0.02;
                anyDioramaRevealed = true;
              }
              // Surface mounts sample the host layer's depth map, so place them once the host loads.
              if (inst.hostSceneId === sceneId && inst.spec.wallMount && inst.dioramaGroup) {
                anyDioramaRevealed = true;
              }
            }
            if (anyDioramaRevealed) {
              this._updatePortalShelfTransforms();
              if (this._sceneNeedsLocalClipping(this.currentSceneId)) {
                this._updatePortalClippingPlanes();
              }
            }
          }
          resolve();
        }
      };
      const handleSlotError = (err) => {
        if (failed) return;
        failed = true;
        obj.loadPromise = null;
        reject(err);
      };
      for (const slot of slots) {
        this._loadTextureIntoSlot(
          slot.url,
          slot.tex,
          handleSlotDone,
          isInitialScene,
          handleSlotError
        );
      }
      for (const sUrl of sharedUrlList) {
        this._loadSharedUrlTexture(sUrl, isInitialScene).then(handleSlotDone, handleSlotError);
      }
    });
    return obj.loadPromise;
  }

  _getSharedPlaneGeo(w, h, sx = 1, sy = 1) {
    if (!this._sharedPlaneGeos) {
      this._sharedPlaneGeos = new Map();
    }
    const cacheKey = `${w}_${h}_${sx}_${sy}`;
    let geo = this._sharedPlaneGeos.get(cacheKey);
    if (!geo) {
      geo = new THREE.PlaneGeometry(w, h, sx, sy);
      this._sharedPlaneGeos.set(cacheKey, geo);
    }
    return geo;
  }

  initScenes() {
    this.textureLoader = new THREE.TextureLoader();
    this._dustSpriteTexture = createSoftRadialDustTexture();
    this.sceneObjects = {};
    this._sharedPlaneGeos = new Map();
    const getSharedPlaneGeo = (w, h, sx = 1, sy = 1) => this._getSharedPlaneGeo(w, h, sx, sy);

    const distBg = this.baseCameraZ - SCENE_GEOMETRY_SPEC.bgZ;

    for (const [key, cfg] of Object.entries(this.sceneConfigs)) {
      const scene = new THREE.Scene();
      const group = new THREE.Group();
      scene.add(group);

      const sceneBgZ = typeof cfg.bgZ === 'number' ? cfg.bgZ : SCENE_GEOMETRY_SPEC.bgZ;
      const sceneMidZ = typeof cfg.midZ === 'number' ? cfg.midZ : SCENE_GEOMETRY_SPEC.midZ;
      const sceneFgZ = typeof cfg.fgZ === 'number' ? cfg.fgZ : SCENE_GEOMETRY_SPEC.fgZ;
      const segX =
        typeof cfg.bgSegmentsX === 'number' ? cfg.bgSegmentsX : SCENE_GEOMETRY_SPEC.bgSegmentsX;
      const segY =
        typeof cfg.bgSegmentsY === 'number' ? cfg.bgSegmentsY : SCENE_GEOMETRY_SPEC.bgSegmentsY;
      const roomDepthAmplitude =
        typeof cfg.roomDepthAmplitude === 'number'
          ? cfg.roomDepthAmplitude
          : SCENE_GEOMETRY_SPEC.roomDepthReliefAmplitude;

      // Layer 0: Subdivided 2700x1800 Outpainted + LaMa-Inpainted Room Mesh with Depth Anything V2 Relief
      const bgGeo = getSharedPlaneGeo(
        SCENE_GEOMETRY_SPEC.bgBaseWidth,
        SCENE_GEOMETRY_SPEC.bgBaseHeight,
        segX,
        segY
      );
      const bgTex = new THREE.Texture();
      const depthTex = new THREE.Texture();
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
          uDepthAmplitude: { value: roomDepthAmplitude },
          uDistBg: { value: this.baseCameraZ - sceneBgZ },
          uPinMarginUv: { value: SCENE_GEOMETRY_SPEC.roomPinMarginUv },
          uOpticalCenterLocalX: { value: 0.0 },
          uOpticalCenterLocalY: { value: 0.0 },
          uOpacity: { value: 1.0 },
          uAlphaCutout: { value: 0.0 },
          uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
          uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
          uFadeToClear: { value: 0.0 },
          uClipUvMinX: { value: 0.0 },
          uClipUvMaxX: { value: 1.0 },
          uCavityUvRect: { value: new THREE.Vector4(0.0, 0.0, 1.0, 1.0) },
          uCavityFeatherUv: { value: new THREE.Vector2(0.001, 0.001) },
          uOuterWallOpacity: { value: 1.0 }
        },
        clipping: false,
        depthWrite: true,
        depthTest: true
      });
      const layer0 = new THREE.Mesh(bgGeo, bgMat);
      layer0.position.set(0, 0, sceneBgZ);
      group.add(layer0);

      // Layer 1: Neural-Segmented Midground Sculptures (1500x1000 inner frame, optional 3D depth relief)
      const hasMidDepthRelief = Boolean(
        cfg.midDepthTexture && typeof cfg.midDepthAmplitude === 'number'
      );
      const midSegX = hasMidDepthRelief
        ? typeof cfg.midSegmentsX === 'number'
          ? cfg.midSegmentsX
          : 224
        : 1;
      const midSegY = hasMidDepthRelief
        ? typeof cfg.midSegmentsY === 'number'
          ? cfg.midSegmentsY
          : 150
        : 1;
      const midGeo = getSharedPlaneGeo(
        SCENE_GEOMETRY_SPEC.midBaseWidth,
        SCENE_GEOMETRY_SPEC.midBaseHeight,
        midSegX,
        midSegY
      );
      const midTex = new THREE.Texture();
      midTex.minFilter = THREE.LinearFilter;
      midTex.magFilter = THREE.LinearFilter;
      const midAlphaCutout = typeof cfg.midAlphaCutout === 'number' ? cfg.midAlphaCutout : 0.0;
      const midPinMarginUv = typeof cfg.midPinMarginUv === 'number' ? cfg.midPinMarginUv : 0.0;
      let midDepthTex = null;
      let midMat = null;
      if (hasMidDepthRelief) {
        midDepthTex = new THREE.Texture();
        midDepthTex.minFilter = THREE.LinearFilter;
        midDepthTex.magFilter = THREE.LinearFilter;
        midMat = new THREE.ShaderMaterial({
          vertexShader: ROOM_DEPTH_VERTEX_SHADER,
          fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
          uniforms: {
            uColorMap: { value: midTex },
            uDepthMap: { value: midDepthTex },
            uDepthAmplitude: { value: cfg.midDepthAmplitude },
            uDistBg: { value: Math.max(1.0, this.baseCameraZ - sceneMidZ) },
            uPinMarginUv: { value: midPinMarginUv },
            uOpticalCenterLocalX: { value: 0.0 },
            uOpticalCenterLocalY: { value: 0.0 },
            uOpacity: { value: 1.0 },
            uAlphaCutout: { value: midAlphaCutout },
            uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
            uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
            uFadeToClear: { value: 0.0 },
            uClipUvMinX: { value: 0.0 },
            uClipUvMaxX: { value: 1.0 },
            uCavityUvRect: { value: new THREE.Vector4(0.0, 0.0, 1.0, 1.0) },
            uCavityFeatherUv: { value: new THREE.Vector2(0.001, 0.001) },
            uOuterWallOpacity: { value: 1.0 }
          },
          clipping: false,
          transparent: midAlphaCutout <= 0.0,
          depthWrite: Boolean(cfg.midDepthWrite),
          depthTest: Boolean(cfg.midDepthTest)
        });
      } else {
        midMat = new THREE.MeshBasicMaterial({
          map: midTex,
          transparent: true,
          depthWrite: false,
          depthTest: false,
          fog: false,
          toneMapped: false
        });
      }
      const layer1 = new THREE.Mesh(midGeo, midMat);
      layer1.position.set(0, 0, sceneMidZ);
      layer1.renderOrder = typeof cfg.midRenderOrder === 'number' ? cfg.midRenderOrder : 2;
      group.add(layer1);

      // Layer 2: Neural-Segmented Foreground Sculptures / Subjects (optional 3D depth relief)
      const hasFgDepthRelief = Boolean(
        cfg.fgDepthTexture && typeof cfg.fgDepthAmplitude === 'number'
      );
      const fgSegX = hasFgDepthRelief
        ? typeof cfg.fgSegmentsX === 'number'
          ? cfg.fgSegmentsX
          : 224
        : 1;
      const fgSegY = hasFgDepthRelief
        ? typeof cfg.fgSegmentsY === 'number'
          ? cfg.fgSegmentsY
          : 150
        : 1;
      const fgGeo = getSharedPlaneGeo(
        SCENE_GEOMETRY_SPEC.fgBaseWidth,
        SCENE_GEOMETRY_SPEC.fgBaseHeight,
        fgSegX,
        fgSegY
      );
      const fgTex = new THREE.Texture();
      fgTex.minFilter = THREE.LinearFilter;
      fgTex.magFilter = THREE.LinearFilter;
      const fgAlphaCutout = typeof cfg.fgAlphaCutout === 'number' ? cfg.fgAlphaCutout : 0.0;
      const fgPinMarginUv = typeof cfg.fgPinMarginUv === 'number' ? cfg.fgPinMarginUv : 0.0;
      let fgDepthTex = null;
      let fgMat = null;
      if (hasFgDepthRelief) {
        fgDepthTex = new THREE.Texture();
        fgDepthTex.minFilter = THREE.LinearFilter;
        fgDepthTex.magFilter = THREE.LinearFilter;
        fgMat = new THREE.ShaderMaterial({
          vertexShader: ROOM_DEPTH_VERTEX_SHADER,
          fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
          uniforms: {
            uColorMap: { value: fgTex },
            uDepthMap: { value: fgDepthTex },
            uDepthAmplitude: { value: cfg.fgDepthAmplitude },
            uDistBg: { value: Math.max(1.0, this.baseCameraZ - sceneFgZ) },
            uPinMarginUv: { value: fgPinMarginUv },
            uOpticalCenterLocalX: { value: 0.0 },
            uOpticalCenterLocalY: { value: 0.0 },
            uOpacity: { value: 1.0 },
            uAlphaCutout: { value: fgAlphaCutout },
            uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
            uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
            uFadeToClear: { value: 0.0 },
            uClipUvMinX: { value: 0.0 },
            uClipUvMaxX: { value: 1.0 },
            uCavityUvRect: { value: new THREE.Vector4(0.0, 0.0, 1.0, 1.0) },
            uCavityFeatherUv: { value: new THREE.Vector2(0.001, 0.001) },
            uOuterWallOpacity: { value: 1.0 }
          },
          clipping: false,
          transparent: fgAlphaCutout <= 0.0,
          depthWrite: Boolean(cfg.fgDepthWrite),
          depthTest: Boolean(cfg.fgDepthTest)
        });
      } else {
        fgMat = new THREE.MeshBasicMaterial({
          map: fgTex,
          transparent: true,
          depthWrite: false,
          depthTest: false,
          fog: false,
          toneMapped: false
        });
      }
      const layer2 = new THREE.Mesh(fgGeo, fgMat);
      layer2.position.set(0, 0, sceneFgZ);
      layer2.renderOrder = typeof cfg.fgRenderOrder === 'number' ? cfg.fgRenderOrder : 3;
      group.add(layer2);

      // Layer 0.5: Bounded Architectural / Overhead Mirror Reflections (Z = -340, strictly behind Mid/Fg)
      let reflectionMesh = null;
      let refTex = null;
      if (cfg.hasReflection && cfg.reflectionTexture) {
        const refGeo = getSharedPlaneGeo(
          SCENE_GEOMETRY_SPEC.fgBaseWidth,
          SCENE_GEOMETRY_SPEC.fgBaseHeight
        );
        refTex = new THREE.Texture();
        refTex.minFilter = THREE.LinearFilter;
        const refMat = new THREE.MeshBasicMaterial({
          map: refTex,
          transparent: true,
          opacity: typeof cfg.reflectionBaseOpacity === 'number' ? cfg.reflectionBaseOpacity : 0.35,
          depthWrite: false,
          depthTest: false,
          fog: false,
          toneMapped: false,
          blending: THREE.NormalBlending
        });
        reflectionMesh = new THREE.Mesh(refGeo, refMat);
        reflectionMesh.position.set(0, 0, SCENE_GEOMETRY_SPEC.reflectionZ);
        reflectionMesh.renderOrder = 1;
        group.add(reflectionMesh);
      }

      // Optional intermediate 2.5D floating or depth-displaced RGBA artwork planes (e.g. Z = -70, -215, -275)
      const extraLayers = [];
      let extraTextureCount = 0;
      if (Array.isArray(cfg.extraLayers)) {
        for (const elCfg of cfg.extraLayers) {
          const hasDepthRelief = Boolean(
            elCfg.depthTexture && typeof elCfg.depthAmplitude === 'number'
          );
          const elSegX = hasDepthRelief
            ? typeof elCfg.segmentsX === 'number'
              ? elCfg.segmentsX
              : 256
            : 1;
          const elSegY = hasDepthRelief
            ? typeof elCfg.segmentsY === 'number'
              ? elCfg.segmentsY
              : 172
            : 1;
          const elGeo = getSharedPlaneGeo(
            SCENE_GEOMETRY_SPEC.fgBaseWidth,
            SCENE_GEOMETRY_SPEC.fgBaseHeight,
            elSegX,
            elSegY
          );
          const elTex = new THREE.Texture();
          elTex.minFilter = THREE.LinearFilter;
          elTex.magFilter = THREE.LinearFilter;
          extraTextureCount += 1;

          const baseOpacity = typeof elCfg.baseOpacity === 'number' ? elCfg.baseOpacity : 1.0;
          const alphaCutout = typeof elCfg.alphaCutout === 'number' ? elCfg.alphaCutout : 0.0;
          const clipUvMinX = typeof elCfg.clipUvMinX === 'number' ? elCfg.clipUvMinX : 0.0;
          const clipUvMaxX = typeof elCfg.clipUvMaxX === 'number' ? elCfg.clipUvMaxX : 1.0;
          const pinMarginUv =
            typeof elCfg.pinMarginUv === 'number'
              ? elCfg.pinMarginUv
              : SCENE_GEOMETRY_SPEC.depthEdgePinMarginUv;
          const mirrorLeftEdge = Boolean(elCfg.mirrorLeftEdge);
          const meshSide = mirrorLeftEdge ? THREE.DoubleSide : THREE.FrontSide;
          const useDepthWrite = Boolean(elCfg.depthWrite);
          const useDepthTest = hasDepthRelief ? elCfg.depthTest !== false : Boolean(elCfg.depthTest);

          let elDepthTex = null;
          let elMat = null;
          if (hasDepthRelief) {
            elDepthTex = new THREE.Texture();
            elDepthTex.minFilter = THREE.LinearFilter;
            elDepthTex.magFilter = THREE.LinearFilter;
            extraTextureCount += 1;
            elMat = new THREE.ShaderMaterial({
              vertexShader: ROOM_DEPTH_VERTEX_SHADER,
              fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
              uniforms: {
                uColorMap: { value: elTex },
                uDepthMap: { value: elDepthTex },
                uDepthAmplitude: { value: elCfg.depthAmplitude },
                uDistBg: { value: Math.max(1.0, this.baseCameraZ - elCfg.zPos) },
                uPinMarginUv: { value: pinMarginUv },
                uOpticalCenterLocalX: {
                  value: mirrorLeftEdge ? -SCENE_GEOMETRY_SPEC.fgBaseWidth : 0.0
                },
                uOpticalCenterLocalY: { value: 0.0 },
                uOpacity: { value: baseOpacity },
                uAlphaCutout: { value: alphaCutout },
                uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
                uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
                uFadeToClear: { value: 0.0 },
                uClipUvMinX: { value: clipUvMinX },
                uClipUvMaxX: { value: clipUvMaxX },
                uCavityUvRect: { value: new THREE.Vector4(0.0, 0.0, 1.0, 1.0) },
                uCavityFeatherUv: { value: new THREE.Vector2(0.001, 0.001) },
                uOuterWallOpacity: { value: 1.0 }
              },
              side: meshSide,
              clipping: false,
              transparent: alphaCutout <= 0.0,
              depthWrite: useDepthWrite,
              depthTest: useDepthTest
            });
          } else {
            elMat = new THREE.MeshBasicMaterial({
              map: elTex,
              side: meshSide,
              transparent: true,
              opacity: baseOpacity,
              depthWrite: useDepthWrite,
              depthTest: useDepthTest,
              fog: false,
              toneMapped: false
            });
          }

          const elMesh = new THREE.Mesh(elGeo, elMat);
          elMesh.position.set(0, 0, elCfg.zPos);
          elMesh.renderOrder = typeof elCfg.renderOrder === 'number' ? elCfg.renderOrder : 2;
          group.add(elMesh);
          extraLayers.push({
            id: elCfg.id,
            labelHe: elCfg.labelHe,
            textureUrl: elCfg.texture,
            depthTextureUrl: hasDepthRelief ? elCfg.depthTexture : null,
            zPos: elCfg.zPos,
            renderOrder: elMesh.renderOrder,
            expanded: !!elCfg.expanded,
            mirrorLeftEdge,
            baseOpacity,
            excursionBoost: typeof elCfg.excursionBoost === 'number' ? elCfg.excursionBoost : 0.0,
            hasDepthRelief,
            depthAmplitude: hasDepthRelief ? elCfg.depthAmplitude : 0.0,
            alphaCutout,
            pinMarginUv,
            depthWrite: useDepthWrite,
            depthTest: useDepthTest,
            mesh: elMesh,
            tex: elTex,
            depthTex: elDepthTex
          });
        }
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
      layer3.renderOrder = SCENE_GEOMETRY_SPEC.particlesRenderOrder;
      group.add(layer3);

      this.sceneObjects[key] = {
        scene,
        group,
        layer0,
        layer1,
        layer2,
        layer3,
        reflectionMesh,
        extraLayers,
        sceneBgZ,
        sceneMidZ,
        sceneFgZ,
        roomDepthAmplitude,
        hasMidDepthRelief,
        midDepthAmplitude: hasMidDepthRelief ? cfg.midDepthAmplitude : 0.0,
        midAlphaCutout,
        midPinMarginUv,
        hasFgDepthRelief,
        fgDepthAmplitude: hasFgDepthRelief ? cfg.fgDepthAmplitude : 0.0,
        fgAlphaCutout,
        fgPinMarginUv,
        innerScale: this._innerScale,
        layer0Scale: this._layer0Scale,
        bgTex,
        depthTex,
        midTex,
        midDepthTex,
        fgTex,
        fgDepthTex,
        refTex,
        texturesReady: false,
        gpuPrewarmed: false,
        loadedCount: 0,
        expectedCount:
          4 +
          (refTex ? 1 : 0) +
          (midDepthTex ? 1 : 0) +
          (fgDepthTex ? 1 : 0) +
          extraTextureCount,
        loadPromise: null
      };
    }

    this._initPortalShelves();
    this.applyDynamicWidescreenScaling();

    // Full Bundle Preloading (#27): load active scene AND all embedded portal dioramas
    // + prewarm GPU shaders before lifting the loading curtain so zero elements pop in later.
    this._loadSceneBundleReady(this.currentSceneId, true).then(() => {
      this.checkAllLayersReady();
      this._warmBackgroundScenesFromManifest();
    });
  }

  _getRequiredBundleSceneIds(sceneId) {
    const ordered = [];
    if (sceneId && this.sceneConfigs[sceneId]) {
      ordered.push(sceneId);
    }
    const shelves = this.getPortalShelves(sceneId);
    for (const spec of shelves) {
      if (
        spec &&
        spec.targetSceneId &&
        this.sceneConfigs[spec.targetSceneId] &&
        !ordered.includes(spec.targetSceneId)
      ) {
        ordered.push(spec.targetSceneId);
        if (spec.embedDiorama !== false) {
          const childShelves = this.getPortalShelves(spec.targetSceneId);
          for (const childSpec of childShelves) {
            if (
              childSpec &&
              childSpec.embedDiorama !== false &&
              childSpec.targetSceneId &&
              this.sceneConfigs[childSpec.targetSceneId] &&
              !ordered.includes(childSpec.targetSceneId)
            ) {
              ordered.push(childSpec.targetSceneId);
            }
          }
        }
      }
    }
    return ordered;
  }

  _isSceneBundleReady(sceneId) {
    const bundleIds = this._getRequiredBundleSceneIds(sceneId);
    if (bundleIds.length === 0) return false;
    return bundleIds.every((id) => Boolean(this.sceneObjects[id] && this.sceneObjects[id].texturesReady));
  }

  async _loadSceneBundleReady(sceneId, isInitialScene = false) {
    const bundleIds = this._getRequiredBundleSceneIds(sceneId);
    await Promise.all(bundleIds.map((id) => this._loadSceneTextures(id, isInitialScene && id === sceneId)));
    if (this._disposed || !this.renderer) return;

    if (this.hasPortalShelves(sceneId)) {
      this._ensurePortalShelvesInitialized(sceneId);
      this._setHostPortalSurroundingOpacity(sceneId, 1.0, null);
      this._updatePortalShelfTransforms();
    }
    for (const id of bundleIds) {
      if (this.sceneObjects[id] && this.sceneObjects[id].texturesReady) {
        this._prewarmSceneGpuResources(id);
      }
    }
    const hostObj = this.sceneObjects[sceneId];
    if (hostObj && hostObj.scene) {
      const prevClip = this.renderer.localClippingEnabled;
      this.renderer.localClippingEnabled = this._sceneNeedsLocalClipping(sceneId);
      this.renderer.compile(hostObj.scene, this.camera);
      this.renderer.localClippingEnabled = prevClip;
    }
  }

  _applyCameraRigManifestConfig(rawRig) {
    if (!rawRig || typeof rawRig !== 'object') return;
    const num = (v, fallback) => (typeof v === 'number' && Number.isFinite(v) ? v : fallback);
    this._rigSpec = {
      ...this._rigSpec,
      desktopFovDeg: num(rawRig.desktop_fov_deg, this._rigSpec.desktopFovDeg),
      portraitFovDeg: num(rawRig.portrait_fov_deg, this._rigSpec.portraitFovDeg),
      keyMoveSpeedPerSec: num(rawRig.key_move_speed_per_sec, this._rigSpec.keyMoveSpeedPerSec),
      keyElevationSpeedPerSec: num(
        rawRig.key_elevation_speed_per_sec,
        this._rigSpec.keyElevationSpeedPerSec
      ),
      gyroGain: num(rawRig.gyro_gain, this._rigSpec.gyroGain),
      gyroMaxYawDeg: num(rawRig.gyro_max_yaw_deg, this._rigSpec.gyroMaxYawDeg),
      gyroMaxPitchDeg: num(rawRig.gyro_max_pitch_deg, this._rigSpec.gyroMaxPitchDeg),
      gyroSmoothingRate: num(rawRig.gyro_smoothing_rate, this._rigSpec.gyroSmoothingRate),
      gyroRecenterThresholdDeg: num(
        rawRig.gyro_recenter_threshold_deg,
        this._rigSpec.gyroRecenterThresholdDeg
      ),
      gyroParallaxShiftX: num(rawRig.gyro_parallax_shift_x, this._rigSpec.gyroParallaxShiftX),
      gyroParallaxShiftY: num(rawRig.gyro_parallax_shift_y, this._rigSpec.gyroParallaxShiftY),
      gyroHintDismissYawDeg: num(
        rawRig.gyro_hint_dismiss_yaw_deg,
        this._rigSpec.gyroHintDismissYawDeg
      ),
      gyroHintDismissPitchDeg: num(
        rawRig.gyro_hint_dismiss_pitch_deg,
        this._rigSpec.gyroHintDismissPitchDeg
      ),
      gyroNoEventsTimeoutMs: num(
        rawRig.gyro_no_events_timeout_ms,
        this._rigSpec.gyroNoEventsTimeoutMs
      ),
      portalFlightDurationMs: num(
        rawRig.portal_flight_duration_ms,
        this._rigSpec.portalFlightDurationMs
      ),
      portalFadeStartProgress: num(
        rawRig.portal_fade_start_progress,
        this._rigSpec.portalFadeStartProgress
      ),
      portalDustRenderOrder: num(
        rawRig.portal_dust_render_order,
        this._rigSpec.portalDustRenderOrder
      ),
      portalOccluderRegenDelayMs: num(
        rawRig.portal_occluder_regen_delay_ms,
        this._rigSpec.portalOccluderRegenDelayMs
      ),
      clearColorHex: num(rawRig.clear_color_hex, this._rigSpec.clearColorHex)
    };
  }

  _warmPortalTargetScenes(onlyHostSceneId = null) {
    for (const portal of this._portalInstances) {
      if (onlyHostSceneId && portal.hostSceneId !== onlyHostSceneId) continue;
      const targetId = portal.spec && portal.spec.targetSceneId;
      if (
        targetId &&
        this.sceneConfigs[targetId] &&
        (!this._manifestSceneIds || this._manifestSceneIds.has(targetId))
      ) {
        this._loadSceneTextures(targetId, false).catch(() => {});
      }
    }
  }

  async _warmBackgroundScenesFromManifest() {
    if (typeof location !== 'undefined' && location.protocol === 'file:') {
      return;
    }
    await new Promise((resolve) =>
      setTimeout(resolve, SCENE_GEOMETRY_SPEC.backgroundWarmInitialDelayMs)
    );
    if (this._disposed) return;

    let manifestSceneIds = null;
    if (typeof fetch === 'function' && typeof AbortController !== 'undefined') {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2500);
      try {
        const resp = await fetch('assets/scenes/scene_manifest.json', {
          cache: 'no-store',
          signal: controller.signal
        });
        if (resp.ok) {
          const manifest = await resp.json();
          if (manifest && manifest.pipeline_config && manifest.pipeline_config.camera_rig) {
            this._applyCameraRigManifestConfig(manifest.pipeline_config.camera_rig);
          }
          manifestSceneIds = new Set();
          if (Array.isArray(manifest.scenes)) {
            for (const entry of manifest.scenes) {
              const sid = entry && (entry.id || entry.scene_id);
              if (typeof sid === 'string') {
                manifestSceneIds.add(sid);
                const rawShelves = entry.portalShelves || entry.portal_shelves;
                const layoutVer = entry.portalLayoutVersion || entry.portal_layout_version;
                if (
                  sid === 'shelf_portal' &&
                  layoutVer === 'full_wall_v4' &&
                  Array.isArray(rawShelves) &&
                  rawShelves.length > 0
                ) {
                  this.updatePortalShelves(rawShelves, 'shelf_portal');
                } else if (
                  sid !== 'shelf_portal' &&
                  this.sceneConfigs[sid] &&
                  Array.isArray(rawShelves) &&
                  rawShelves.length > 0
                ) {
                  this.updatePortalShelves(rawShelves, sid);
                }
              }
            }
          } else if (manifest.scenes && typeof manifest.scenes === 'object') {
            for (const [k, entry] of Object.entries(manifest.scenes)) {
              const sid = (entry && (entry.id || entry.scene_id)) || k;
              if (typeof sid === 'string') {
                manifestSceneIds.add(sid);
                const rawShelves = entry && (entry.portalShelves || entry.portal_shelves);
                const layoutVer =
                  entry && (entry.portalLayoutVersion || entry.portal_layout_version);
                if (
                  sid === 'shelf_portal' &&
                  layoutVer === 'full_wall_v4' &&
                  Array.isArray(rawShelves) &&
                  rawShelves.length > 0
                ) {
                  this.updatePortalShelves(rawShelves, 'shelf_portal');
                } else if (
                  sid !== 'shelf_portal' &&
                  this.sceneConfigs[sid] &&
                  Array.isArray(rawShelves) &&
                  rawShelves.length > 0
                ) {
                  this.updatePortalShelves(rawShelves, sid);
                }
              }
            }
          }
          this._manifestSceneIds = manifestSceneIds;
        }
      } catch (_) {
        // Non-fatal manifest probe; explicit scene transitions still load on demand
      } finally {
        clearTimeout(timeoutId);
      }
    }

    for (const otherId of this.sceneOrder) {
      if (this._disposed) return;
      if (otherId === this.currentSceneId) continue;
      if (!manifestSceneIds || manifestSceneIds.has(otherId)) {
        try {
          await this._loadSceneTextures(otherId, false);
          if (!this._disposed && this.sceneObjects[otherId] && this.sceneObjects[otherId].texturesReady) {
            this._prewarmSceneGpuResources(otherId);
          }
        } catch (_) {
          // Scene assets still synthesizing offline; leave loadPromise=null for on-demand retry
        }
        await new Promise((resolve) =>
          setTimeout(resolve, SCENE_GEOMETRY_SPEC.backgroundWarmStepDelayMs)
        );
      }
    }
  }

  _disposePortalShelves(onlyHostSceneId = null) {
    const remaining = [];
    for (const inst of this._portalInstances) {
      if (onlyHostSceneId && inst.hostSceneId !== onlyHostSceneId) {
        remaining.push(inst);
        continue;
      }
      const hostObj = this.sceneObjects && this.sceneObjects[inst.hostSceneId];
      if (hostObj && hostObj.group) {
        if (inst.dioramaGroup) hostObj.group.remove(inst.dioramaGroup);
        if (inst.frameLine) hostObj.group.remove(inst.frameLine);
        if (inst.occluderMesh) hostObj.group.remove(inst.occluderMesh);
        if (inst.veilMesh) hostObj.group.remove(inst.veilMesh);
      }
      if (Array.isArray(inst.ownedResources)) {
        for (const res of inst.ownedResources) {
          if (res && typeof res.dispose === 'function') {
            res.dispose();
          }
        }
      }
    }
    this._portalInstances = remaining;
    if (!onlyHostSceneId) {
      for (const res of this._portalOwnedResources) {
        if (res && typeof res.dispose === 'function') {
          res.dispose();
        }
      }
      this._portalOwnedResources = [];
    }
  }

  _ensurePortalShelvesInitialized(hostSceneId = null) {
    const targetHosts = hostSceneId
      ? [hostSceneId]
      : Object.keys(this.sceneConfigs).filter((sid) => this.hasPortalShelves(sid));
    let anyMissing = false;
    for (const sid of targetHosts) {
      if (!this.hasPortalShelves(sid)) continue;
      if (!this._portalInstances.some((inst) => inst.hostSceneId === sid)) {
        this._initPortalShelves(sid);
        anyMissing = true;
      }
    }
    if (anyMissing) {
      this._updatePortalShelfTransforms();
    }
  }

  updatePortalShelves(rawSpecs, hostSceneId = 'shelf_portal') {
    const hostCfg = this.sceneConfigs && this.sceneConfigs[hostSceneId];
    if (!hostCfg) return;
    const nextSpecs = this._normalizePortalShelfSpecs(rawSpecs, hostSceneId);
    const prevSpecs = hostCfg.portalShelves || [];
    const existingForHost = this._portalInstances.filter((p) => p.hostSceneId === hostSceneId);
    const unchanged =
      existingForHost.length === nextSpecs.length &&
      JSON.stringify(this._canonicalizePortalSpecsForComparison(prevSpecs)) ===
        JSON.stringify(this._canonicalizePortalSpecsForComparison(nextSpecs));
    if (unchanged) {
      return;
    }
    hostCfg.portalShelves = nextSpecs;
    if (this._portalFlight) {
      this._pendingPortalShelfRebuild = hostSceneId;
      return;
    }
    if (existingForHost.length > 0 || this.currentSceneId === hostSceneId) {
      this._initPortalShelves(hostSceneId);
      this._updatePortalShelfTransforms();
      if (this.currentSceneId === hostSceneId) {
        this._warmPortalTargetScenes(hostSceneId);
      }
    }
  }

  _resolvePortalTexture(customUrl, defaultUrl, sharedTex) {
    if (!customUrl || customUrl === defaultUrl) {
      return sharedTex;
    }
    const entry = this._getOrCreateSharedUrlTextureEntry(customUrl);
    this._loadSharedUrlTexture(customUrl, false).catch(() => {});
    return entry.tex;
  }

  _computeCavityUvUniforms(wm) {
    if (!wm || !Array.isArray(wm.cavityInnerBoxPx) || wm.cavityInnerBoxPx.length !== 4) {
      return {
        rect: new THREE.Vector4(0.0, 0.0, 1.0, 1.0),
        feather: new THREE.Vector2(0.001, 0.001)
      };
    }
    const expW = SCENE_GEOMETRY_SPEC.innerFrameWidthPx * SCENE_GEOMETRY_SPEC.bgExpansionFactor;
    const expH = SCENE_GEOMETRY_SPEC.innerFrameHeightPx * SCENE_GEOMETRY_SPEC.bgExpansionFactor;
    const ox = 0.5 * (expW - SCENE_GEOMETRY_SPEC.innerFrameWidthPx);
    const oy = 0.5 * (expH - SCENE_GEOMETRY_SPEC.innerFrameHeightPx);
    const [x0, y0, x1, y1] = wm.cavityInnerBoxPx;
    const fPx = Math.max(0.5, typeof wm.cavityFeatherPx === 'number' ? wm.cavityFeatherPx : 4.0);
    return {
      rect: new THREE.Vector4(
        (ox + x0) / expW,
        1.0 - (oy + y1) / expH,
        (ox + x1) / expW,
        1.0 - (oy + y0) / expH
      ),
      feather: new THREE.Vector2(fPx / expW, fPx / expH)
    };
  }

  _initPortalShelves(onlyHostSceneId = null) {
    this._disposePortalShelves(onlyHostSceneId);
    const hostIds = onlyHostSceneId
      ? [onlyHostSceneId]
      : Object.keys(this.sceneConfigs).filter((sid) => this.hasPortalShelves(sid));

    for (const hostSceneId of hostIds) {
      const hostObj = this.sceneObjects && this.sceneObjects[hostSceneId];
      const hostCfg = this.sceneConfigs && this.sceneConfigs[hostSceneId];
      if (!hostObj || !hostCfg || !Array.isArray(hostCfg.portalShelves)) continue;
      const shouldEagerLoadOccluder = Boolean(hostObj.loadPromise || hostObj.texturesReady);

      for (const spec of hostCfg.portalShelves) {
        const targetObj = this.sceneObjects[spec.targetSceneId];
        const targetCfg = this.sceneConfigs[spec.targetSceneId];
        if (!targetObj || !targetCfg) continue;

        const ownedResources = [];
        const dioramaTintMaterials = [];
        const embedDiorama = spec.embedDiorama !== false;
        const showFrame = spec.showFrame !== false;

        let clipPlanes = null;
        let dioramaGroup = null;
        let bgMesh = null;
        let bgMat = null;
        let refMesh = null;
        let midMesh = null;
        let midMat = null;
        const extraMeshes = [];
        let fgMesh = null;
        let fgMat = null;
        let dustMesh = null;
        let dustMat = null;
        const childPortalMeshes = [];

        if (embedDiorama) {
          const isWallMount = Boolean(spec.wallMount);
          clipPlanes = isWallMount
            ? null
            : [
                new THREE.Plane(),
                new THREE.Plane(),
                new THREE.Plane(),
                new THREE.Plane()
              ];
          const useLocalClip = Boolean(clipPlanes);
          const wmCavity = isWallMount ? this._computeCavityUvUniforms(spec.wallMount) : null;
          const wmOrders = isWallMount ? spec.wallMount.dioramaRenderOrders : null;

          dioramaGroup = new THREE.Group();
          const baseOrder = SCENE_GEOMETRY_SPEC.portalBaseRenderOrder;
          const targetMidOrder =
            typeof targetCfg.midRenderOrder === 'number' ? targetCfg.midRenderOrder : 2;
          const targetFgOrder =
            typeof targetCfg.fgRenderOrder === 'number' ? targetCfg.fgRenderOrder : 3;

          const bgColorTex = this._resolvePortalTexture(
            spec.previewBgTex,
            targetCfg.bgTexture,
            targetObj.bgTex
          );
          const bgDepthTex = this._resolvePortalTexture(
            spec.previewBgDepth,
            targetCfg.depthTexture,
            targetObj.depthTex
          );
          const midColorTex = this._resolvePortalTexture(
            spec.previewMidTex,
            targetCfg.midTexture,
            targetObj.midTex
          );
          const fgColorTex = this._resolvePortalTexture(
            spec.previewFgTex,
            targetCfg.fgTexture,
            targetObj.fgTex
          );

          bgMat = new THREE.ShaderMaterial({
            vertexShader: ROOM_DEPTH_VERTEX_SHADER,
            fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
            uniforms: {
              uColorMap: { value: bgColorTex },
              uDepthMap: { value: bgDepthTex },
              uDepthAmplitude: { value: targetObj.roomDepthAmplitude },
              uDistBg: { value: this.baseCameraZ - targetObj.sceneBgZ },
              uPinMarginUv: { value: SCENE_GEOMETRY_SPEC.roomPinMarginUv },
              uOpticalCenterLocalX: { value: 0.0 },
              uOpticalCenterLocalY: { value: 0.0 },
              uOpacity: { value: 1.0 },
              uAlphaCutout: { value: 0.0 },
              uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
              uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
              uFadeToClear: { value: 0.0 },
              uClipUvMinX: { value: 0.0 },
              uClipUvMaxX: { value: 1.0 },
              uCavityUvRect: {
                value: wmCavity ? wmCavity.rect : new THREE.Vector4(0.0, 0.0, 1.0, 1.0)
              },
              uCavityFeatherUv: {
                value: wmCavity ? wmCavity.feather : new THREE.Vector2(0.001, 0.001)
              },
              uOuterWallOpacity: { value: isWallMount ? 0.0 : 1.0 }
            },
            clipping: useLocalClip,
            clippingPlanes: clipPlanes,
            transparent: isWallMount,
            depthWrite: !isWallMount,
            depthTest: !isWallMount,
            depthFunc: isWallMount ? THREE.LessEqualDepth : THREE.AlwaysDepth
          });
          ownedResources.push(bgMat);
          dioramaTintMaterials.push(bgMat);
          bgMesh = new THREE.Mesh(targetObj.layer0.geometry, bgMat);
          bgMesh.renderOrder = wmOrders ? wmOrders.bg : baseOrder;
          dioramaGroup.add(bgMesh);

          if (targetObj.reflectionMesh && targetObj.refTex) {
            const refMat = new THREE.MeshBasicMaterial({
              map: targetObj.refTex,
              transparent: true,
              opacity:
                typeof targetCfg.reflectionBaseOpacity === 'number'
                  ? targetCfg.reflectionBaseOpacity
                  : 0.35,
              depthWrite: false,
              depthTest: false,
              fog: false,
              toneMapped: false,
              clippingPlanes: clipPlanes
            });
            ownedResources.push(refMat);
            dioramaTintMaterials.push(refMat);
            refMesh = new THREE.Mesh(targetObj.reflectionMesh.geometry, refMat);
            refMesh.renderOrder = baseOrder + 0.5;
            dioramaGroup.add(refMesh);
          }

          if (targetObj.hasMidDepthRelief && targetObj.midDepthTex) {
            midMat = new THREE.ShaderMaterial({
              vertexShader: ROOM_DEPTH_VERTEX_SHADER,
              fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
              uniforms: {
                uColorMap: { value: midColorTex },
                uDepthMap: { value: targetObj.midDepthTex },
                uDepthAmplitude: { value: targetObj.midDepthAmplitude },
                uDistBg: { value: Math.max(1.0, this.baseCameraZ - targetObj.sceneMidZ) },
                uPinMarginUv: { value: targetObj.midPinMarginUv },
                uOpticalCenterLocalX: { value: 0.0 },
                uOpticalCenterLocalY: { value: 0.0 },
                uOpacity: { value: 1.0 },
                uAlphaCutout: { value: targetObj.midAlphaCutout },
                uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
                uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
                uFadeToClear: { value: 0.0 },
                uClipUvMinX: { value: 0.0 },
                uClipUvMaxX: { value: 1.0 },
                uCavityUvRect: { value: new THREE.Vector4(0.0, 0.0, 1.0, 1.0) },
                uCavityFeatherUv: { value: new THREE.Vector2(0.001, 0.001) },
                uOuterWallOpacity: { value: 1.0 }
              },
              clipping: useLocalClip,
              clippingPlanes: clipPlanes,
              transparent: targetObj.midAlphaCutout <= 0.0,
              depthWrite: Boolean(targetCfg.midDepthWrite),
              depthTest: Boolean(targetCfg.midDepthTest)
            });
          } else {
            midMat = new THREE.MeshBasicMaterial({
              map: midColorTex,
              transparent: true,
              depthWrite: false,
              depthTest: false,
              fog: false,
              toneMapped: false,
              clippingPlanes: clipPlanes
            });
          }
          ownedResources.push(midMat);
          dioramaTintMaterials.push(midMat);
          midMesh = new THREE.Mesh(targetObj.layer1.geometry, midMat);
          midMesh.renderOrder = wmOrders ? wmOrders.mid : baseOrder + targetMidOrder;
          dioramaGroup.add(midMesh);

          if (Array.isArray(targetObj.extraLayers)) {
            for (const extra of targetObj.extraLayers) {
              let exMat = null;
              if (extra.hasDepthRelief && extra.depthTex) {
                exMat = new THREE.ShaderMaterial({
                  vertexShader: ROOM_DEPTH_VERTEX_SHADER,
                  fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
                  uniforms: {
                    uColorMap: { value: extra.tex },
                    uDepthMap: { value: extra.depthTex },
                    uDepthAmplitude: { value: extra.depthAmplitude },
                    uDistBg: { value: Math.max(1.0, this.baseCameraZ - extra.zPos) },
                    uPinMarginUv: { value: extra.pinMarginUv },
                    uOpticalCenterLocalX: {
                      value: extra.mirrorLeftEdge ? -SCENE_GEOMETRY_SPEC.fgBaseWidth : 0.0
                    },
                    uOpticalCenterLocalY: { value: 0.0 },
                    uOpacity: { value: extra.baseOpacity },
                    uAlphaCutout: { value: extra.alphaCutout },
                    uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
                    uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
                    uFadeToClear: { value: 0.0 },
                    uClipUvMinX: { value: 0.0 },
                    uClipUvMaxX: { value: 1.0 },
                    uCavityUvRect: { value: new THREE.Vector4(0.0, 0.0, 1.0, 1.0) },
                    uCavityFeatherUv: { value: new THREE.Vector2(0.001, 0.001) },
                    uOuterWallOpacity: { value: 1.0 }
                  },
                  side: extra.mirrorLeftEdge ? THREE.DoubleSide : THREE.FrontSide,
                  clipping: useLocalClip,
                  clippingPlanes: clipPlanes,
                  transparent: extra.alphaCutout <= 0.0,
                  depthWrite: Boolean(extra.depthWrite),
                  depthTest: Boolean(extra.depthTest)
                });
              } else {
                exMat = new THREE.MeshBasicMaterial({
                  map: extra.tex,
                  transparent: true,
                  opacity: extra.baseOpacity,
                  depthWrite: false,
                  depthTest: Boolean(extra.depthTest),
                  fog: false,
                  toneMapped: false,
                  clippingPlanes: clipPlanes
                });
              }
              ownedResources.push(exMat);
              dioramaTintMaterials.push(exMat);
              const exMesh = new THREE.Mesh(extra.mesh.geometry, exMat);
              exMesh.renderOrder = baseOrder + extra.renderOrder;
              dioramaGroup.add(exMesh);
              extraMeshes.push({ sourceExtra: extra, mesh: exMesh, mat: exMat });
            }
          }

          if (targetObj.hasFgDepthRelief && targetObj.fgDepthTex) {
            fgMat = new THREE.ShaderMaterial({
              vertexShader: ROOM_DEPTH_VERTEX_SHADER,
              fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
              uniforms: {
                uColorMap: { value: fgColorTex },
                uDepthMap: { value: targetObj.fgDepthTex },
                uDepthAmplitude: { value: targetObj.fgDepthAmplitude },
                uDistBg: { value: Math.max(1.0, this.baseCameraZ - targetObj.sceneFgZ) },
                uPinMarginUv: { value: targetObj.fgPinMarginUv },
                uOpticalCenterLocalX: { value: 0.0 },
                uOpticalCenterLocalY: { value: 0.0 },
                uOpacity: { value: 1.0 },
                uAlphaCutout: { value: targetObj.fgAlphaCutout },
                uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
                uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
                uFadeToClear: { value: 0.0 },
                uClipUvMinX: { value: 0.0 },
                uClipUvMaxX: { value: 1.0 },
                uCavityUvRect: { value: new THREE.Vector4(0.0, 0.0, 1.0, 1.0) },
                uCavityFeatherUv: { value: new THREE.Vector2(0.001, 0.001) },
                uOuterWallOpacity: { value: 1.0 }
              },
              clipping: useLocalClip,
              clippingPlanes: clipPlanes,
              transparent: targetObj.fgAlphaCutout <= 0.0,
              depthWrite: Boolean(targetCfg.fgDepthWrite),
              depthTest: Boolean(targetCfg.fgDepthTest)
            });
          } else {
            fgMat = new THREE.MeshBasicMaterial({
              map: fgColorTex,
              transparent: true,
              depthWrite: false,
              depthTest: false,
              fog: false,
              toneMapped: false,
              clippingPlanes: clipPlanes
            });
          }
          ownedResources.push(fgMat);
          dioramaTintMaterials.push(fgMat);
          fgMesh = new THREE.Mesh(targetObj.layer2.geometry, fgMat);
          fgMesh.renderOrder = wmOrders ? wmOrders.fg : baseOrder + targetFgOrder;
          dioramaGroup.add(fgMesh);

          dustMat = targetObj.layer3.material.clone();
          dustMat.clippingPlanes = clipPlanes;
          if (isWallMount) {
            dustMat.opacity = 0.0;
          }
          ownedResources.push(dustMat);
          dustMesh = new THREE.Points(targetObj.layer3.geometry, dustMat);
          if (isWallMount) {
            dustMesh.visible = false;
          }
          dustMesh.renderOrder = wmOrders
            ? wmOrders.dust
            : hostSceneId === 'shelf_portal'
              ? this._rigSpec.portalDustRenderOrder
              : baseOrder + targetFgOrder + 0.5;
          dioramaGroup.add(dustMesh);

          // Child portal representation meshes attached to dioramaGroup (with full depth relief & shared URL textures)
          childPortalMeshes.length = 0;
          if (Array.isArray(targetCfg.portalShelves)) {
            for (const childSpec of targetCfg.portalShelves) {
              const childTargetObj = this.sceneObjects[childSpec.targetSceneId];
              const childTargetCfg = this.sceneConfigs[childSpec.targetSceneId];

              let childDioBgMesh = null;
              let childWallMount = null;
              if (childSpec.embedDiorama !== false && childTargetObj && childTargetCfg) {
                if (childSpec.wallMount) {
                  const cWmGroup = new THREE.Group();
                  const cWmCav = this._computeCavityUvUniforms(childSpec.wallMount);
                  const cWmOrders = childSpec.wallMount.dioramaRenderOrders || {
                    bg: 2.0,
                    mid: 3.0,
                    fg: 4.0
                  };
                  const cBgMat = new THREE.ShaderMaterial({
                    vertexShader: ROOM_DEPTH_VERTEX_SHADER,
                    fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
                    uniforms: {
                      uColorMap: { value: childTargetObj.bgTex },
                      uDepthMap: { value: childTargetObj.depthTex },
                      uDepthAmplitude: { value: childTargetObj.roomDepthAmplitude },
                      uDistBg: { value: this.baseCameraZ - childTargetObj.sceneBgZ },
                      uPinMarginUv: { value: SCENE_GEOMETRY_SPEC.roomPinMarginUv },
                      uOpticalCenterLocalX: { value: 0.0 },
                      uOpticalCenterLocalY: { value: 0.0 },
                      uOpacity: { value: 1.0 },
                      uAlphaCutout: { value: 0.0 },
                      uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
                      uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
                      uFadeToClear: { value: 0.0 },
                      uClipUvMinX: { value: 0.0 },
                      uClipUvMaxX: { value: 1.0 },
                      uCavityUvRect: { value: cWmCav.rect },
                      uCavityFeatherUv: { value: cWmCav.feather },
                      uOuterWallOpacity: { value: 0.0 }
                    },
                    clipping: useLocalClip,
                    clippingPlanes: clipPlanes,
                    transparent: true,
                    depthWrite: false,
                    depthTest: false
                  });
                  const cBgMesh = new THREE.Mesh(childTargetObj.layer0.geometry, cBgMat);
                  cBgMesh.renderOrder = baseOrder + cWmOrders.bg;
                  cWmGroup.add(cBgMesh);

                  let cMidMat = null;
                  if (childTargetObj.hasMidDepthRelief && childTargetObj.midDepthTex) {
                    cMidMat = new THREE.ShaderMaterial({
                      vertexShader: ROOM_DEPTH_VERTEX_SHADER,
                      fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
                      uniforms: {
                        uColorMap: { value: childTargetObj.midTex },
                        uDepthMap: { value: childTargetObj.midDepthTex },
                        uDepthAmplitude: { value: childTargetObj.midDepthAmplitude },
                        uDistBg: {
                          value: Math.max(1.0, this.baseCameraZ - childTargetObj.sceneMidZ)
                        },
                        uPinMarginUv: { value: childTargetObj.midPinMarginUv },
                        uOpticalCenterLocalX: { value: 0.0 },
                        uOpticalCenterLocalY: { value: 0.0 },
                        uOpacity: { value: 1.0 },
                        uAlphaCutout: { value: childTargetObj.midAlphaCutout },
                        uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
                        uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
                        uFadeToClear: { value: 0.0 },
                        uClipUvMinX: { value: 0.0 },
                        uClipUvMaxX: { value: 1.0 },
                        uCavityUvRect: { value: new THREE.Vector4(0.0, 0.0, 1.0, 1.0) },
                        uCavityFeatherUv: { value: new THREE.Vector2(0.001, 0.001) },
                        uOuterWallOpacity: { value: 1.0 }
                      },
                      clipping: useLocalClip,
                      clippingPlanes: clipPlanes,
                      transparent: true,
                      depthWrite: false,
                      depthTest: false
                    });
                  } else {
                    cMidMat = new THREE.MeshBasicMaterial({
                      map: childTargetObj.midTex,
                      transparent: true,
                      depthWrite: false,
                      depthTest: false,
                      fog: false,
                      toneMapped: false,
                      clippingPlanes: clipPlanes
                    });
                  }
                  const cMidMesh = new THREE.Mesh(childTargetObj.layer1.geometry, cMidMat);
                  cMidMesh.renderOrder = baseOrder + cWmOrders.mid;
                  cWmGroup.add(cMidMesh);

                  let cFgMat = null;
                  if (childTargetObj.hasFgDepthRelief && childTargetObj.fgDepthTex) {
                    cFgMat = new THREE.ShaderMaterial({
                      vertexShader: ROOM_DEPTH_VERTEX_SHADER,
                      fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
                      uniforms: {
                        uColorMap: { value: childTargetObj.fgTex },
                        uDepthMap: { value: childTargetObj.fgDepthTex },
                        uDepthAmplitude: { value: childTargetObj.fgDepthAmplitude },
                        uDistBg: {
                          value: Math.max(1.0, this.baseCameraZ - childTargetObj.sceneFgZ)
                        },
                        uPinMarginUv: { value: childTargetObj.fgPinMarginUv },
                        uOpticalCenterLocalX: { value: 0.0 },
                        uOpticalCenterLocalY: { value: 0.0 },
                        uOpacity: { value: 1.0 },
                        uAlphaCutout: { value: childTargetObj.fgAlphaCutout },
                        uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
                        uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
                        uFadeToClear: { value: 0.0 },
                        uClipUvMinX: { value: 0.0 },
                        uClipUvMaxX: { value: 1.0 },
                        uCavityUvRect: { value: new THREE.Vector4(0.0, 0.0, 1.0, 1.0) },
                        uCavityFeatherUv: { value: new THREE.Vector2(0.001, 0.001) },
                        uOuterWallOpacity: { value: 1.0 }
                      },
                      clipping: useLocalClip,
                      clippingPlanes: clipPlanes,
                      transparent: true,
                      depthWrite: false,
                      depthTest: false
                    });
                  } else {
                    cFgMat = new THREE.MeshBasicMaterial({
                      map: childTargetObj.fgTex,
                      transparent: true,
                      depthWrite: false,
                      depthTest: false,
                      fog: false,
                      toneMapped: false,
                      clippingPlanes: clipPlanes
                    });
                  }
                  const cFgMesh = new THREE.Mesh(childTargetObj.layer2.geometry, cFgMat);
                  cFgMesh.renderOrder = baseOrder + cWmOrders.fg;
                  cWmGroup.add(cFgMesh);

                  ownedResources.push(cBgMat, cMidMat, cFgMat);
                  dioramaTintMaterials.push(cBgMat, cMidMat, cFgMat);
                  dioramaGroup.add(cWmGroup);
                  childWallMount = {
                    group: cWmGroup,
                    bgMesh: cBgMesh,
                    bgMat: cBgMat,
                    midMesh: cMidMesh,
                    midMat: cMidMat,
                    fgMesh: cFgMesh,
                    fgMat: cFgMat,
                    childTargetId: childSpec.targetSceneId
                  };
                } else {
                  const childBgTex = this._resolvePortalTexture(
                    childSpec.previewBgTex,
                    childTargetCfg.bgTexture,
                    childTargetObj.bgTex
                  );
                  const childBgMat = new THREE.MeshBasicMaterial({
                    map: childBgTex,
                    transparent: true,
                    opacity: 1.0,
                    depthWrite: false,
                    depthTest: true,
                    fog: false,
                    toneMapped: false,
                    clippingPlanes: clipPlanes
                  });
                  ownedResources.push(childBgMat);
                  dioramaTintMaterials.push(childBgMat);
                  const childBgGeo = this._getSharedPlaneGeo(1, 1, 1, 1);
                  childDioBgMesh = new THREE.Mesh(childBgGeo, childBgMat);
                  childDioBgMesh.renderOrder = baseOrder + 1.5;
                  dioramaGroup.add(childDioBgMesh);
                }
              }

              let childFrameLine = null;
              if (childSpec.showFrame !== false) {
                const bevelPts = [
                  new THREE.Vector3(-0.5, 0.5, 0),
                  new THREE.Vector3(0.5, 0.5, 0),
                  new THREE.Vector3(0.5, -0.5, 0),
                  new THREE.Vector3(-0.5, -0.5, 0)
                ];
                const frameGeo = new THREE.BufferGeometry().setFromPoints(bevelPts);
                const frameMat = new THREE.LineBasicMaterial({
                  color: 0xE6B980,
                  transparent: true,
                  opacity:
                    typeof childSpec.restFrameOpacity === 'number'
                      ? childSpec.restFrameOpacity
                      : 0.24,
                  depthWrite: false,
                  depthTest: false,
                  clippingPlanes: clipPlanes
                });
                ownedResources.push(frameGeo, frameMat);
                childFrameLine = new THREE.LineLoop(frameGeo, frameMat);
                childFrameLine.renderOrder =
                  typeof childSpec.frameRenderOrder === 'number'
                    ? baseOrder + childSpec.frameRenderOrder
                    : baseOrder + 5.0;
                dioramaGroup.add(childFrameLine);
              }

              let childOccMesh = null;
              let childOccMat = null;
              let childOccTex = null;
              let childOccDepthTex = null;
              if (childSpec.occluder && childSpec.occluder.texture) {
                childOccTex = this._getOrCreateSharedUrlTextureEntry(childSpec.occluder.texture).tex;
                if (shouldEagerLoadOccluder) {
                  this._loadSharedUrlTexture(childSpec.occluder.texture, false).catch(() => {});
                }
                const isChildOccErased = this._erasedPortalOccluders.has(childSpec.id);
                const hasChildOccDepth = Boolean(
                  childSpec.occluder.fullInnerCanvas &&
                    childSpec.occluder.depthTexture &&
                    childSpec.occluder.depthAmplitude > 0.0
                );
                if (hasChildOccDepth) {
                  childOccDepthTex = this._getOrCreateSharedUrlTextureEntry(
                    childSpec.occluder.depthTexture
                  ).tex;
                  if (shouldEagerLoadOccluder) {
                    this._loadSharedUrlTexture(childSpec.occluder.depthTexture, false).catch(
                      () => {}
                    );
                  }
                  const occGeo = this._getSharedPlaneGeo(
                    SCENE_GEOMETRY_SPEC.fgBaseWidth,
                    SCENE_GEOMETRY_SPEC.fgBaseHeight,
                    childSpec.occluder.segmentsX,
                    childSpec.occluder.segmentsY
                  );
                  childOccMat = new THREE.ShaderMaterial({
                    vertexShader: ROOM_DEPTH_VERTEX_SHADER,
                    fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
                    uniforms: {
                      uColorMap: { value: childOccTex },
                      uDepthMap: { value: childOccDepthTex },
                      uDepthAmplitude: { value: childSpec.occluder.depthAmplitude },
                      uDistBg: {
                        value: Math.max(1.0, this.baseCameraZ - childSpec.occluder.worldZ)
                      },
                      uPinMarginUv: { value: childSpec.occluder.pinMarginUv },
                      uOpticalCenterLocalX: { value: 0.0 },
                      uOpticalCenterLocalY: { value: 0.0 },
                      uOpacity: { value: isChildOccErased ? 0.0 : 1.0 },
                      uAlphaCutout: { value: 0.0 },
                      uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
                      uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
                      uFadeToClear: { value: 0.0 },
                      uClipUvMinX: { value: 0.0 },
                      uClipUvMaxX: { value: 1.0 },
                      uCavityUvRect: { value: new THREE.Vector4(0.0, 0.0, 1.0, 1.0) },
                      uCavityFeatherUv: { value: new THREE.Vector2(0.001, 0.001) },
                      uOuterWallOpacity: { value: 1.0 }
                    },
                    clipping: useLocalClip,
                    clippingPlanes: clipPlanes,
                    transparent: true,
                    depthWrite: false,
                    depthTest: false
                  });
                  ownedResources.push(childOccMat);
                  dioramaTintMaterials.push(childOccMat);
                  childOccMesh = new THREE.Mesh(occGeo, childOccMat);
                } else {
                  const occGeo = childSpec.occluder.fullInnerCanvas
                    ? this._getSharedPlaneGeo(
                        SCENE_GEOMETRY_SPEC.fgBaseWidth,
                        SCENE_GEOMETRY_SPEC.fgBaseHeight,
                        1,
                        1
                      )
                    : this._getSharedPlaneGeo(1, 1, 1, 1);
                  childOccMat = new THREE.MeshBasicMaterial({
                    map: childOccTex,
                    transparent: true,
                    opacity: isChildOccErased ? 0.0 : 1.0,
                    depthWrite: false,
                    depthTest: false,
                    fog: false,
                    toneMapped: false,
                    clippingPlanes: clipPlanes
                  });
                  ownedResources.push(childOccMat);
                  dioramaTintMaterials.push(childOccMat);
                  childOccMesh = new THREE.Mesh(occGeo, childOccMat);
                }
                childOccMesh.visible = !isChildOccErased;
                childOccMesh.renderOrder =
                  typeof childSpec.occluder.renderOrder === 'number'
                    ? baseOrder + childSpec.occluder.renderOrder
                    : baseOrder + 6.0;
                dioramaGroup.add(childOccMesh);
              }

              let childVeilMesh = null;
              let childVeilMat = null;
              let childVeilTex = null;
              if (childSpec.veil && childSpec.veil.texture) {
                childVeilTex = this._getOrCreateSharedUrlTextureEntry(childSpec.veil.texture).tex;
                if (shouldEagerLoadOccluder) {
                  this._loadSharedUrlTexture(childSpec.veil.texture, false).catch(() => {});
                }
                const veilGeo = childSpec.veil.fullInnerCanvas
                  ? this._getSharedPlaneGeo(
                      SCENE_GEOMETRY_SPEC.fgBaseWidth,
                      SCENE_GEOMETRY_SPEC.fgBaseHeight,
                      1,
                      1
                    )
                  : this._getSharedPlaneGeo(1, 1, 1, 1);
                const vOp =
                  typeof childSpec.veil.restOpacity === 'number' ? childSpec.veil.restOpacity : 1.0;
                childVeilMat = new THREE.MeshBasicMaterial({
                  map: childVeilTex,
                  transparent: true,
                  opacity: vOp,
                  depthWrite: false,
                  depthTest: false,
                  fog: false,
                  toneMapped: false,
                  clippingPlanes: clipPlanes
                });
                ownedResources.push(childVeilMat);
                dioramaTintMaterials.push(childVeilMat);
                childVeilMesh = new THREE.Mesh(veilGeo, childVeilMat);
                childVeilMesh.visible = vOp > 0.005;
                childVeilMesh.renderOrder =
                  typeof childSpec.veil.renderOrder === 'number'
                    ? baseOrder + childSpec.veil.renderOrder
                    : baseOrder + 6.5;
                dioramaGroup.add(childVeilMesh);
              }

              childPortalMeshes.push({
                childSpec,
                targetSceneId: spec.targetSceneId,
                childDioBgMesh,
                childWallMount,
                childFrameLine,
                childOccMesh,
                childOccMat,
                childOccTex,
                childOccDepthTex,
                childVeilMesh,
                childVeilMat,
                childVeilTex
              });
            }
          }

          dioramaGroup.visible =
            hostSceneId === 'shelf_portal' ? true : Boolean(targetObj.texturesReady);
          hostObj.group.add(dioramaGroup);
        }

        let frameLine = null;
        let frameMat = null;
        if (showFrame) {
          const bevelPts = [
            new THREE.Vector3(-0.5, 0.5, 0),
            new THREE.Vector3(0.5, 0.5, 0),
            new THREE.Vector3(0.5, -0.5, 0),
            new THREE.Vector3(-0.5, -0.5, 0)
          ];
          const frameGeo = new THREE.BufferGeometry().setFromPoints(bevelPts);
          frameMat = new THREE.LineBasicMaterial({
            color: 0xE6B980,
            transparent: true,
            opacity: typeof spec.restFrameOpacity === 'number' ? spec.restFrameOpacity : 0.24,
            depthWrite: false,
            depthTest: false
          });
          ownedResources.push(frameGeo, frameMat);
          frameLine = new THREE.LineLoop(frameGeo, frameMat);
          frameLine.renderOrder =
            typeof spec.frameRenderOrder === 'number'
              ? spec.frameRenderOrder
              : SCENE_GEOMETRY_SPEC.shelfFgRenderOrder - 1;
          hostObj.group.add(frameLine);
        }

        let occluderMesh = null;
        let occluderMat = null;
        let occluderTex = null;
        let occluderDepthTex = null;
        const isErased = this._erasedPortalOccluders.has(spec.id);
        if (spec.occluder && spec.occluder.texture) {
          occluderTex = this._getOrCreateSharedUrlTextureEntry(spec.occluder.texture).tex;
          if (shouldEagerLoadOccluder) {
            this._loadSharedUrlTexture(spec.occluder.texture, false).catch(() => {});
          }

          const hasOccDepth = Boolean(
            spec.occluder.fullInnerCanvas &&
              spec.occluder.depthTexture &&
              spec.occluder.depthAmplitude > 0.0
          );
          if (hasOccDepth) {
            occluderDepthTex = this._getOrCreateSharedUrlTextureEntry(
              spec.occluder.depthTexture
            ).tex;
            if (shouldEagerLoadOccluder) {
              this._loadSharedUrlTexture(spec.occluder.depthTexture, false).catch(() => {});
            }
            const occGeo = this._getSharedPlaneGeo(
              SCENE_GEOMETRY_SPEC.fgBaseWidth,
              SCENE_GEOMETRY_SPEC.fgBaseHeight,
              spec.occluder.segmentsX,
              spec.occluder.segmentsY
            );
            occluderMat = new THREE.ShaderMaterial({
              vertexShader: ROOM_DEPTH_VERTEX_SHADER,
              fragmentShader: ROOM_DEPTH_FRAGMENT_SHADER,
              uniforms: {
                uColorMap: { value: occluderTex },
                uDepthMap: { value: occluderDepthTex },
                uDepthAmplitude: { value: spec.occluder.depthAmplitude },
                uDistBg: { value: Math.max(1.0, this.baseCameraZ - spec.occluder.worldZ) },
                uPinMarginUv: { value: spec.occluder.pinMarginUv },
                uOpticalCenterLocalX: { value: 0.0 },
                uOpticalCenterLocalY: { value: 0.0 },
                uOpacity: { value: isErased ? 0.0 : 1.0 },
                uAlphaCutout: { value: 0.0 },
                uTint: { value: new THREE.Color(1.0, 1.0, 1.0) },
                uClearColor: { value: new THREE.Color(this._rigSpec.clearColorHex) },
                uFadeToClear: { value: 0.0 },
                uClipUvMinX: { value: 0.0 },
                uClipUvMaxX: { value: 1.0 },
                uCavityUvRect: { value: new THREE.Vector4(0.0, 0.0, 1.0, 1.0) },
                uCavityFeatherUv: { value: new THREE.Vector2(0.001, 0.001) },
                uOuterWallOpacity: { value: 1.0 }
              },
              clipping: false,
              transparent: true,
              depthWrite: false,
              depthTest: false
            });
            ownedResources.push(occluderMat);
            occluderMesh = new THREE.Mesh(occGeo, occluderMat);
            occluderMesh.visible = !isErased;
            occluderMesh.renderOrder = spec.occluder.renderOrder;
            hostObj.group.add(occluderMesh);
          } else {
            const occGeo = spec.occluder.fullInnerCanvas
              ? this._getSharedPlaneGeo(
                  SCENE_GEOMETRY_SPEC.fgBaseWidth,
                  SCENE_GEOMETRY_SPEC.fgBaseHeight,
                  1,
                  1
                )
              : this._getSharedPlaneGeo(1, 1, 1, 1);
            occluderMat = new THREE.MeshBasicMaterial({
              map: occluderTex,
              transparent: true,
              opacity: isErased ? 0.0 : 1.0,
              depthWrite: false,
              depthTest: false,
              fog: false,
              toneMapped: false
            });
            ownedResources.push(occluderMat);
            occluderMesh = new THREE.Mesh(occGeo, occluderMat);
            occluderMesh.visible = !isErased;
            occluderMesh.renderOrder = spec.occluder.renderOrder;
            hostObj.group.add(occluderMesh);
          }
        }

        let veilMesh = null;
        let veilMat = null;
        let veilTex = null;
        const initVeilOp =
          spec.veil && typeof spec.veil.restOpacity === 'number' ? spec.veil.restOpacity : 1.0;
        if (spec.veil && spec.veil.texture) {
          veilTex = this._getOrCreateSharedUrlTextureEntry(spec.veil.texture).tex;
          if (shouldEagerLoadOccluder) {
            this._loadSharedUrlTexture(spec.veil.texture, false).catch(() => {});
          }
          const veilGeo = spec.veil.fullInnerCanvas
            ? this._getSharedPlaneGeo(
                SCENE_GEOMETRY_SPEC.fgBaseWidth,
                SCENE_GEOMETRY_SPEC.fgBaseHeight,
                1,
                1
              )
            : this._getSharedPlaneGeo(1, 1, 1, 1);
          veilMat = new THREE.MeshBasicMaterial({
            map: veilTex,
            transparent: true,
            opacity: initVeilOp,
            depthWrite: false,
            depthTest: false,
            fog: false,
            toneMapped: false
          });
          ownedResources.push(veilMat);
          veilMesh = new THREE.Mesh(veilGeo, veilMat);
          veilMesh.visible = initVeilOp > 0.005;
          veilMesh.renderOrder = spec.veil.renderOrder;
          hostObj.group.add(veilMesh);
        }

        const instRecord = {
          spec,
          hostSceneId,
          targetSceneId: spec.targetSceneId,
          embedDiorama,
          showFrame,
          ownedResources,
          dioramaTintMaterials,
          clipPlanes,
          dioramaGroup,
          bgMesh,
          bgMat,
          refMesh,
          midMesh,
          midMat,
          extraMeshes,
          fgMesh,
          fgMat,
          dustMesh,
          dustMat,
          frameLine,
          frameMat,
          occluderMesh,
          occluderMat,
          occluderTex,
          occluderDepthTex,
          occluderBaseGeom: null,
          occluderPlaneScale: 1.0,
          occluderPlaneCenterY: 0.0,
          occluderPlaneWorldW: SCENE_GEOMETRY_SPEC.fgBaseWidth,
          occluderPlaneWorldH: SCENE_GEOMETRY_SPEC.fgBaseHeight,
          occluderBoreDist: 400.0,
          occluderOpticalCenterLocalY: 0.0,
          occluderAnimFactor: isErased ? 0.0 : 1.0,
          veilMesh,
          veilMat,
          veilTex,
          veilCurrentOpacity: initVeilOp,
          veilTargetOpacity: initVeilOp,
          veilFlightStartOpacity: initVeilOp,
          wallMountOuterOpacity: 0.0,
          childPortalMeshes: childPortalMeshes || []
        };
        this._applyDioramaInstanceTint(instRecord, spec.dioramaRestTintRgb);
        this._portalInstances.push(instRecord);
      }
    }
  }

  _computeWallMountDioramaTransform(spec, hostSceneId) {
    const wm = spec && spec.wallMount;
    const hostObj = this.sceneObjects && this.sceneObjects[hostSceneId];
    const hostCfg = this.sceneConfigs && this.sceneConfigs[hostSceneId];
    const targetObj = this.sceneObjects && spec && this.sceneObjects[spec.targetSceneId];
    const targetCfg = this.sceneConfigs && spec && this.sceneConfigs[spec.targetSceneId];
    if (!wm || !hostObj || !hostCfg || !targetObj || !targetCfg) return null;

    const halfInnerW = 0.5 * SCENE_GEOMETRY_SPEC.innerFrameWidthPx;
    const halfInnerH = 0.5 * SCENE_GEOMETRY_SPEC.innerFrameHeightPx;

    const sV = targetObj.innerScale;
    const targetRest = this._getSceneRestPose(spec.targetSceneId);
    const tBoreY = targetCfg.boreSightAtRest ? targetRest.y : 0.0;
    const tBoreZ = targetCfg.boreSightAtRest ? targetRest.z : this.baseCameraZ;

    let tDio;
    let qDio;
    let lambdaP;
    let pH0;
    let zPlasterV;
    let restMatrix = null;
    if (Array.isArray(wm.pointPairsInnerPx) && wm.pointPairsInnerPx.length >= 3) {
      // Surface mount (e.g. the Purim photo lying tilted on the scroll floor): fit the similarity
      // from matched target-frame points onto the host layer's real displaced 3D surface.
      const fit = this._fitSurfaceMountSimilarity(wm, hostObj, targetObj);
      if (!fit) return null;
      ({ tDio, qDio, lambdaP, restMatrix } = fit);
      pH0 = fit.hostCentroid;
      zPlasterV = targetObj.sceneBgZ;
    } else {
      const sH = hostObj.innerScale;
      const hostRest = this._getSceneRestPose(hostSceneId);
      const hBoreY = hostCfg.boreSightAtRest ? hostRest.y : 0.0;
      const hBoreZ = hostCfg.boreSightAtRest ? hostRest.z : this.baseCameraZ;
      const distBgH = Math.max(1.0, hBoreZ - hostObj.sceneBgZ);
      const rsH = (hBoreZ - wm.hostWallZ) / distBgH;

      const distBgV = Math.max(1.0, tBoreZ - targetObj.sceneBgZ);
      zPlasterV = targetObj.sceneBgZ + targetObj.roomDepthAmplitude * wm.targetPlasterDepthNorm;
      const rsV = (tBoreZ - zPlasterV) / distBgV;

      lambdaP =
        (wm.hostBoxSidePx * sH * rsH) / Math.max(1e-5, wm.targetBoxSidePx * sV * rsV);

      const refScale = typeof wm.hostRefInnerScale === 'number' && wm.hostRefInnerScale > 0 ? wm.hostRefInnerScale : sH;
      const scaleRatio = sH / refScale;
      const dzDx = (wm.hostWallSlopeDzDx || 0.0) / scaleRatio;
      const dzDy = (wm.hostWallSlopeDzDy || 0.0) / scaleRatio;
      const wallYaw = Math.atan(-dzDx);
      const wallPitch = Math.atan(dzDy * Math.cos(wallYaw));
      const rollRad = THREE.MathUtils.degToRad(wm.inPlaneRollDeg || 0.0);
      qDio = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(wallPitch, wallYaw, rollRad, 'YXZ')
      );

      const [cxH, cyH] = wm.hostCenterInnerPx;
      const [cxV, cyV] = wm.targetBoxCenterInnerPx;
      pH0 = new THREE.Vector3(
        0.5 * sH * (cxH - halfInnerW) * rsH,
        hBoreY + (-0.5 * sH * (cyH - halfInnerH) - hBoreY) * rsH,
        wm.hostWallZ + (wm.hostWallClearanceWorld || 0.0)
      );
      const pV0 = new THREE.Vector3(
        0.5 * sV * (cxV - halfInnerW) * rsV,
        tBoreY + (-0.5 * sV * (cyV - halfInnerH) - tBoreY) * rsV,
        zPlasterV
      );
      tDio = pH0.clone().sub(pV0.clone().applyQuaternion(qDio).multiplyScalar(lambdaP));
    }

    const restLocalVec = new THREE.Vector3(targetRest.x, targetRest.y, targetRest.z);
    const arrivalCamPos = tDio
      .clone()
      .add(restLocalVec.applyQuaternion(qDio).multiplyScalar(lambdaP));
    const qTargetRest = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(
        THREE.MathUtils.degToRad(targetRest.pitchDeg || 0.0),
        THREE.MathUtils.degToRad(targetRest.yawDeg || 0.0),
        THREE.MathUtils.degToRad(targetRest.rollDeg || 0.0),
        'YXZ'
      )
    );
    const arrivalCamQuat = qDio.clone().multiply(qTargetRest);

    return {
      hostAnchor: pH0,
      restMatrix,
      tDio,
      qDio,
      lambdaP,
      sV,
      tBoreY,
      tBoreZ,
      zPlasterV,
      arrivalCamPos,
      arrivalCamQuat
    };
  }

  /** Bilinear red-channel sample (0..1) of a loaded texture image at uv (flipY convention). */
  _sampleTextureRed(tex, uvx, uvy) {
    const img = tex && tex.image;
    const w = img && (img.naturalWidth || img.width);
    const h = img && (img.naturalHeight || img.height);
    if (!w || !h) return null;
    if (!tex.userData) tex.userData = {};
    if (!tex.userData.redSampler || tex.userData.redSampler.img !== img) {
      const cv = document.createElement('canvas');
      cv.width = w;
      cv.height = h;
      const ctx = cv.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(img, 0, 0, w, h);
      tex.userData.redSampler = { img, w, h, data: ctx.getImageData(0, 0, w, h).data };
    }
    const { w: sw, h: sh, data } = tex.userData.redSampler;
    const fx = THREE.MathUtils.clamp(uvx * sw - 0.5, 0, sw - 1);
    const fy = THREE.MathUtils.clamp((1.0 - uvy) * sh - 0.5, 0, sh - 1);
    const x0 = Math.floor(fx);
    const y0 = Math.floor(fy);
    const x1 = Math.min(sw - 1, x0 + 1);
    const y1 = Math.min(sh - 1, y0 + 1);
    const ax = fx - x0;
    const ay = fy - y0;
    const r = (x, y) => data[(y * sw + x) * 4] / 255.0;
    return (
      (r(x0, y0) * (1 - ax) + r(x1, y0) * ax) * (1 - ay) +
      (r(x0, y1) * (1 - ax) + r(x1, y1) * ax) * ay
    );
  }

  /**
   * World position of a host layer's rendered surface at inner-frame pixel (px, py), evaluating
   * ROOM_DEPTH_VERTEX_SHADER's displacement on the CPU so mounts sit on the real relief.
   */
  _sampleLayerSurfaceWorldPoint(layer, px, py) {
    const mesh = layer && layer.mesh;
    const u = mesh && mesh.material && mesh.material.uniforms;
    const params = mesh && mesh.geometry && mesh.geometry.parameters;
    if (!mesh || !params) return null;
    const uvx = px / SCENE_GEOMETRY_SPEC.innerFrameWidthPx;
    const uvy = 1.0 - py / SCENE_GEOMETRY_SPEC.innerFrameHeightPx;
    const lx = (uvx - 0.5) * params.width;
    const ly = (uvy - 0.5) * params.height;
    let displaced = new THREE.Vector3(lx, ly, 0.0);
    if (layer.hasDepthRelief && u && u.uDepthMap) {
      const raw = this._sampleTextureRed(layer.depthTex, uvx, uvy);
      if (raw === null) return null;
      const pinMargin = u.uPinMarginUv ? u.uPinMarginUv.value : 0.0;
      const ss = (e, x) => {
        const t = THREE.MathUtils.clamp(x / e, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
      };
      const pin =
        pinMargin > 1e-5
          ? ss(pinMargin, uvx) * ss(pinMargin, 1.0 - uvx) * ss(pinMargin, uvy) * ss(pinMargin, 1.0 - uvy)
          : 1.0;
      const zDisp = raw * u.uDepthAmplitude.value * pin;
      const distBg = Math.max(1.0, u.uDistBg.value);
      const rayScale = (distBg - zDisp) / distBg;
      const ocx = u.uOpticalCenterLocalX.value;
      const ocy = u.uOpticalCenterLocalY.value;
      displaced = new THREE.Vector3(
        ocx + (lx - ocx) * rayScale,
        ocy + (ly - ocy) * rayScale,
        zDisp
      );
    }
    mesh.updateWorldMatrix(true, false);
    return displaced.applyMatrix4(mesh.matrixWorld);
  }

  /**
   * Surface mount fit for a print lying in the host scene (the Purim photo on the scroll floor).
   *
   * Rest: an affine map A0 sends the target frame (flat at z = bgZ) onto the parallelogram that lies
   * on the host rest camera's rays through four photo corners, so at rest the embedded scene
   * projects pixel-exactly onto the printed photo. Its depth along the rays is pinned to the host
   * layer's real displaced surface at the photo centre.
   * Landing: the closest similarity S1 (uniform scale + rotation) to A0, because only a similarity
   * keeps the arrival camera's view identical to the standalone target scene.
   */
  _fitSurfaceMountSimilarity(wm, hostObj, targetObj) {
    const layer = Array.isArray(hostObj.extraLayers)
      ? hostObj.extraLayers.find((el) => el && el.id === wm.hostLayerId)
      : null;
    const u = layer && layer.mesh && layer.mesh.material && layer.mesh.material.uniforms;
    const params = layer && layer.mesh && layer.mesh.geometry && layer.mesh.geometry.parameters;
    if (!layer || !u || !params) return null;
    const sV = targetObj.innerScale;
    const halfW = 0.5 * SCENE_GEOMETRY_SPEC.innerFrameWidthPx;
    const halfH = 0.5 * SCENE_GEOMETRY_SPEC.innerFrameHeightPx;
    const pairs = wm.pointPairsInnerPx;
    const tOf = ([tu, tv]) => new THREE.Vector2(0.5 * sV * (tu - halfW), -0.5 * sV * (tv - halfH));

    // Corner pairs of the target grid (min/max u, v) and the pair nearest the centre.
    const byTarget = (fu, fv) =>
      pairs.reduce((best, pr) => (fu(pr[0]) + fv(pr[0]) < fu(best[0]) + fv(best[0]) ? pr : best));
    const c00 = byTarget((t) => t[0], (t) => t[1]);
    const c10 = byTarget((t) => -t[0], (t) => t[1]);
    const c01 = byTarget((t) => t[0], (t) => -t[1]);
    const c11 = byTarget((t) => -t[0], (t) => -t[1]);
    const tMid = new THREE.Vector2(
      0.5 * (c00[0][0] + c11[0][0]),
      0.5 * (c00[0][1] + c11[0][1])
    );
    const cMid = pairs.reduce((best, pr) =>
      Math.hypot(pr[0][0] - tMid.x, pr[0][1] - tMid.y) <
      Math.hypot(best[0][0] - tMid.x, best[0][1] - tMid.y)
        ? pr
        : best
    );

    // Rays from the layer's optical centre (the host rest camera) through the flat layer points.
    layer.mesh.updateWorldMatrix(true, false);
    const mw = layer.mesh.matrixWorld;
    const apex = new THREE.Vector3(
      u.uOpticalCenterLocalX.value,
      u.uOpticalCenterLocalY.value,
      Math.max(1.0, u.uDistBg.value)
    ).applyMatrix4(mw);
    const rayTo = ([hx, hy]) => {
      const lx = (hx / SCENE_GEOMETRY_SPEC.innerFrameWidthPx - 0.5) * params.width;
      const ly = (0.5 - hy / SCENE_GEOMETRY_SPEC.innerFrameHeightPx) * params.height;
      return new THREE.Vector3(lx, ly, 0.0).applyMatrix4(mw).sub(apex);
    };
    const r00 = rayTo(c00[1]);
    const r10 = rayTo(c10[1]);
    const r01 = rayTo(c01[1]);
    const r11 = rayTo(c11[1]);
    // Parallelogram: P00 + P11 = P10 + P01, with d00 = 1  ->  d11 r11 - d10 r10 - d01 r01 = -r00.
    const m = new THREE.Matrix3().set(
      r11.x, -r10.x, -r01.x,
      r11.y, -r10.y, -r01.y,
      r11.z, -r10.z, -r01.z
    );
    if (Math.abs(m.determinant()) < 1e-9) return null;
    const d = r00.clone().negate().applyMatrix3(m.clone().invert());
    const P00 = r00.clone();
    const P11 = r11.clone().multiplyScalar(d.x);
    const P10 = r10.clone().multiplyScalar(d.y);
    const P01 = r01.clone().multiplyScalar(d.z);

    // Pin the depth along the rays to the host layer's displaced surface at the photo centre.
    const surfMid = this._sampleLayerSurfaceWorldPoint(layer, cMid[1][0], cMid[1][1]);
    if (!surfMid) return null;
    const rMid = rayTo(cMid[1]);
    const nPlane = new THREE.Vector3().crossVectors(P10.clone().sub(P00), P01.clone().sub(P00));
    const kPlane = P00.dot(nPlane) / Math.max(1e-9, rMid.dot(nPlane));
    const kSurf = surfMid.clone().sub(apex).length() / Math.max(1e-9, rMid.length());
    const depthScale = kSurf / Math.max(1e-9, kPlane);
    for (const P of [P00, P10, P01, P11]) P.multiplyScalar(depthScale).add(apex);

    // Affine A0: target (x, y, z) -> host, with z measured from the target bg plane.
    const t00 = tOf(c00[0]);
    const t10 = tOf(c10[0]);
    const t01 = tOf(c01[0]);
    const U = P10.clone().sub(P00).multiplyScalar(1 / (t10.x - t00.x));
    const W = P01.clone().sub(P00).multiplyScalar(1 / (t01.y - t00.y));
    const lambdaP = 0.5 * (U.length() + W.length());
    const e1 = U.clone().normalize();
    const e2 = W.clone().addScaledVector(e1, -W.dot(e1)).normalize();
    const e3 = new THREE.Vector3().crossVectors(e1, e2);
    const N = e3.clone().multiplyScalar(lambdaP);
    const lift = wm.hostWallClearanceWorld || 0.0;
    // Host point for target (x, y, bgZ): P00 + U (x - t00.x) + W (y - t00.y).
    const origin = P00.clone()
      .addScaledVector(U, -t00.x)
      .addScaledVector(W, -t00.y)
      .addScaledVector(N, -targetObj.sceneBgZ)
      .addScaledVector(e3, lift);
    const restMatrix = new THREE.Matrix4().makeBasis(U, W, N).setPosition(origin);

    // Landing similarity S1: same centre, orthonormalized axes, uniform scale.
    const qDio = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(e1, e2, e3));
    const tc = new THREE.Vector3(
      0.5 * (t00.x + tOf(c11[0]).x),
      0.5 * (t00.y + tOf(c11[0]).y),
      targetObj.sceneBgZ
    );
    const hostCentroid = tc.clone().applyMatrix4(restMatrix);
    const tDio = hostCentroid.clone().sub(tc.clone().applyQuaternion(qDio).multiplyScalar(lambdaP));
    return { tDio, qDio, lambdaP, hostCentroid, restMatrix };
  }

  _applyDioramaInstanceTint(inst, rgbArr) {
    if (!inst || !Array.isArray(inst.dioramaTintMaterials)) return;
    const tint =
      Array.isArray(rgbArr) && rgbArr.length === 3
        ? rgbArr
        : SCENE_GEOMETRY_SPEC.neutralTintRgb;
    const r = Number(tint[0]);
    const g = Number(tint[1]);
    const b = Number(tint[2]);
    for (const mat of inst.dioramaTintMaterials) {
      if (!mat) continue;
      if (mat.uniforms && mat.uniforms.uTint && mat.uniforms.uTint.value) {
        mat.uniforms.uTint.value.setRGB(r, g, b);
      } else if (mat.color && typeof mat.color.setRGB === 'function') {
        mat.color.setRGB(r, g, b);
      }
    }
  }

  _syncSingleDioramaInstance(inst) {
    if (!inst || !inst.embedDiorama || !inst.dioramaGroup) return;
    const hostObj = this.sceneObjects && this.sceneObjects[inst.hostSceneId];
    const hostCfg = this.sceneConfigs && this.sceneConfigs[inst.hostSceneId];
    const targetObj = this.sceneObjects && this.sceneObjects[inst.targetSceneId];
    const targetCfg = this.sceneConfigs && this.sceneConfigs[inst.targetSceneId];
    if (!hostObj || !hostCfg || !targetObj || !targetCfg) return;
    const geom = inst.spec;

    inst.bgMesh.position.copy(targetObj.layer0.position);
    inst.bgMesh.rotation.copy(targetObj.layer0.rotation);
    inst.bgMesh.scale.copy(targetObj.layer0.scale);
    if (targetObj.layer0.material && targetObj.layer0.material.uniforms) {
      inst.bgMat.uniforms.uDistBg.value = targetObj.layer0.material.uniforms.uDistBg.value;
      inst.bgMat.uniforms.uOpticalCenterLocalY.value =
        targetObj.layer0.material.uniforms.uOpticalCenterLocalY.value;
    }

    inst.midMesh.position.copy(targetObj.layer1.position);
    inst.midMesh.rotation.copy(targetObj.layer1.rotation);
    inst.midMesh.scale.copy(targetObj.layer1.scale);
    if (
      inst.midMat &&
      inst.midMat.uniforms &&
      targetObj.layer1.material &&
      targetObj.layer1.material.uniforms
    ) {
      inst.midMat.uniforms.uDistBg.value = targetObj.layer1.material.uniforms.uDistBg.value;
      inst.midMat.uniforms.uOpticalCenterLocalY.value =
        targetObj.layer1.material.uniforms.uOpticalCenterLocalY.value;
    }

    inst.fgMesh.position.copy(targetObj.layer2.position);
    inst.fgMesh.rotation.copy(targetObj.layer2.rotation);
    inst.fgMesh.scale.copy(targetObj.layer2.scale);
    if (
      inst.fgMat &&
      inst.fgMat.uniforms &&
      targetObj.layer2.material &&
      targetObj.layer2.material.uniforms
    ) {
      inst.fgMat.uniforms.uDistBg.value = targetObj.layer2.material.uniforms.uDistBg.value;
      inst.fgMat.uniforms.uOpticalCenterLocalY.value =
        targetObj.layer2.material.uniforms.uOpticalCenterLocalY.value;
    }

    if (inst.refMesh && targetObj.reflectionMesh) {
      inst.refMesh.position.copy(targetObj.reflectionMesh.position);
      inst.refMesh.rotation.copy(targetObj.reflectionMesh.rotation);
      inst.refMesh.scale.copy(targetObj.reflectionMesh.scale);
    }

    for (const exItem of inst.extraMeshes) {
      exItem.mesh.position.copy(exItem.sourceExtra.mesh.position);
      exItem.mesh.rotation.copy(exItem.sourceExtra.mesh.rotation);
      exItem.mesh.scale.copy(exItem.sourceExtra.mesh.scale);
      if (
        exItem.mat &&
        exItem.mat.uniforms &&
        exItem.sourceExtra.mesh.material &&
        exItem.sourceExtra.mesh.material.uniforms
      ) {
        exItem.mat.uniforms.uDistBg.value =
          exItem.sourceExtra.mesh.material.uniforms.uDistBg.value;
        exItem.mat.uniforms.uOpticalCenterLocalX.value =
          exItem.sourceExtra.mesh.material.uniforms.uOpticalCenterLocalX.value;
        exItem.mat.uniforms.uOpticalCenterLocalY.value =
          exItem.sourceExtra.mesh.material.uniforms.uOpticalCenterLocalY.value;
        exItem.mat.uniforms.uClipUvMinX.value =
          exItem.sourceExtra.mesh.material.uniforms.uClipUvMinX.value;
        exItem.mat.uniforms.uClipUvMaxX.value =
          exItem.sourceExtra.mesh.material.uniforms.uClipUvMaxX.value;
      }
    }

    let lambdaP = 1.0;
    if (geom.wallMount) {
      // Wall-mounted dioramas (e.g. the vitrine hung rotated on Bar Giora's wall) sit flush on the
      // host wall plane with their in-plane roll; the flight camera arrives at the matching pose.
      const wmx = this._computeWallMountDioramaTransform(geom, inst.hostSceneId);
      if (!wmx) {
        // Never show an unplaced (identity, full-scale) diorama; retry on the next sync.
        inst.wallMountXform = null;
        inst.hiddenUnplaced = true;
        inst.dioramaGroup.visible = false;
        return;
      }
      inst.wallMountXform = wmx;
      if (inst.hiddenUnplaced) {
        inst.hiddenUnplaced = false;
        inst.dioramaGroup.visible = Boolean(targetObj.texturesReady);
      }
      if (geom.wallMount.flatAtRest) {
        this._applyMountReliefToMaterials(
          [inst.bgMat, inst.midMat, inst.fgMat],
          targetObj,
          this._mountReliefScale(inst)
        );
      }
      lambdaP = wmx.lambdaP;
      this._applyMountGroupTransform(inst.dioramaGroup, wmx, this._mountReliefScale(inst));
    } else {
      const targetRest = this._getSceneRestPose(inst.targetSceneId);
      const frontZ = Math.max(targetObj.sceneFgZ, 0.0);
      const restFrontDist = Math.max(40.0, targetRest.z - frontZ);
      const effFitZ = typeof geom.effectiveFitZ === 'number' ? geom.effectiveFitZ : geom.fitZ;
      lambdaP = (effFitZ + geom.recessDepth) / restFrontDist;
      const fitCamX = geom.worldX;
      const fitCamY = geom.worldY;
      const fitCamZ = geom.worldZ + effFitZ;

      const rollRad = THREE.MathUtils.degToRad(geom.dioramaRollDeg || 0.0);
      const cosR = Math.cos(rollRad);
      const sinR = Math.sin(rollRad);
      inst.dioramaGroup.scale.setScalar(lambdaP);
      inst.dioramaGroup.rotation.set(0, 0, rollRad);
      inst.dioramaGroup.position.set(
        fitCamX - lambdaP * (targetRest.x * cosR - targetRest.y * sinR),
        fitCamY - lambdaP * (targetRest.x * sinR + targetRest.y * cosR),
        fitCamZ - lambdaP * targetRest.z
      );
    }
    if (inst.dustMat && targetObj.layer3 && targetObj.layer3.material) {
      inst.dustMat.size = targetObj.layer3.material.size * lambdaP;
    }
    if (inst.dustMesh && targetObj.layer3) {
      inst.dustMesh.rotation.copy(targetObj.layer3.rotation);
    }

    if (Array.isArray(inst.childPortalMeshes)) {
      const tRest = this._getSceneRestPose(inst.targetSceneId);
      const tBoreY = targetCfg.boreSightAtRest ? tRest.y : 0.0;
      const tBoreZ = targetCfg.boreSightAtRest ? tRest.z : this.baseCameraZ;

      for (const cp of inst.childPortalMeshes) {
        const cs = cp.childSpec;
        const sourceChildInst = this._portalInstances.find(
          (p) => p.hostSceneId === inst.targetSceneId && p.spec.id === cs.id
        );
        if (cp.childWallMount) {
          this._syncNestedWallMount(cp.childWallMount, cs, inst.targetSceneId, sourceChildInst);
        }
        const cGeom = computePortalWorldGeometry(
          cs,
          targetObj.innerScale,
          tBoreY,
          tBoreZ,
          targetObj.sceneBgZ,
          this.camera.fov,
          this.camera.aspect
        );

        if (cp.childFrameLine) {
          cp.childFrameLine.position.set(
            cGeom.worldX,
            cGeom.worldY,
            cGeom.worldZ + SCENE_GEOMETRY_SPEC.portalFrameBevelOffsetZ
          );
          cp.childFrameLine.rotation.set(0, 0, THREE.MathUtils.degToRad(cs.rotationDeg || 0.0));
          cp.childFrameLine.scale.set(cGeom.width, cGeom.height, 1.0);
        }

        if (cp.childDioBgMesh) {
          const recessZ = cGeom.worldZ - cGeom.recessDepth;
          cp.childDioBgMesh.position.set(cGeom.worldX, cGeom.worldY, recessZ);
          cp.childDioBgMesh.scale.set(cGeom.width * 1.5, cGeom.height * 1.5, 1.0);
        }

        if (cp.childOccMesh && cs.occluder) {
          const childOccFactor = sourceChildInst
            ? THREE.MathUtils.clamp(
                typeof sourceChildInst.occluderAnimFactor === 'number'
                  ? sourceChildInst.occluderAnimFactor
                  : 1.0,
                0.0,
                1.0
              )
            : this._erasedPortalOccluders.has(cs.id)
              ? 0.0
              : 1.0;

          if (sourceChildInst && sourceChildInst.occluderMesh) {
            cp.childOccMesh.position.copy(sourceChildInst.occluderMesh.position);
            cp.childOccMesh.rotation.copy(sourceChildInst.occluderMesh.rotation);
            cp.childOccMesh.scale.copy(sourceChildInst.occluderMesh.scale);
            if (cp.childOccMat && cp.childOccMat.uniforms) {
              cp.childOccMat.uniforms.uDistBg.value = sourceChildInst.occluderBoreDist;
              cp.childOccMat.uniforms.uOpticalCenterLocalY.value =
                sourceChildInst.occluderOpticalCenterLocalY;
              cp.childOccMat.uniforms.uOpacity.value = childOccFactor;
            } else if (cp.childOccMat) {
              cp.childOccMat.opacity = childOccFactor;
            }
          } else {
            const occZ = cs.occluder.worldZ;
            const liftT = 1.0 - childOccFactor;
            const scaleBoost =
              1.0 + (SCENE_GEOMETRY_SPEC.portalOccluderLiftScale - 1.0) * liftT;
            if (cs.occluder.fullInnerCanvas) {
              const boreDistBg = Math.max(1.0, tBoreZ - targetObj.sceneBgZ);
              const ratio = (tBoreZ - occZ) / boreDistBg;
              const innerW = SCENE_GEOMETRY_SPEC.bgBaseWidth * targetObj.innerScale;
              const occScale = (innerW * ratio) / SCENE_GEOMETRY_SPEC.fgBaseWidth;
              cp.childOccMesh.position.set(
                0.0,
                tBoreY * (1.0 - ratio),
                occZ + SCENE_GEOMETRY_SPEC.portalOccluderLiftZ * liftT
              );
              cp.childOccMesh.scale.set(occScale * scaleBoost, occScale * scaleBoost, 1.0);
              if (cp.childOccMat && cp.childOccMat.uniforms) {
                cp.childOccMat.uniforms.uDistBg.value = Math.max(1.0, tBoreZ - occZ);
                cp.childOccMat.uniforms.uOpticalCenterLocalY.value =
                  (tBoreY * ratio) / Math.max(1e-5, occScale);
                cp.childOccMat.uniforms.uOpacity.value = childOccFactor;
              } else if (cp.childOccMat) {
                cp.childOccMat.opacity = childOccFactor;
              }
            } else {
              const occGeom = Array.isArray(cs.occluder.innerBoxPx)
                ? computePortalWorldGeometry(
                    { innerBoxPx: cs.occluder.innerBoxPx, worldZ: occZ },
                    targetObj.innerScale,
                    tBoreY,
                    tBoreZ,
                    targetObj.sceneBgZ,
                    this.camera.fov,
                    this.camera.aspect
                  )
                : cGeom;
              cp.childOccMesh.position.set(
                occGeom.worldX,
                occGeom.worldY,
                occZ + SCENE_GEOMETRY_SPEC.portalOccluderLiftZ * liftT
              );
              cp.childOccMesh.scale.set(
                occGeom.width * scaleBoost,
                occGeom.height * scaleBoost,
                1.0
              );
              if (cp.childOccMat) {
                cp.childOccMat.opacity = childOccFactor;
              }
            }
          }
          cp.childOccMesh.visible = childOccFactor > 0.005;
        }

        if (cp.childVeilMesh && cs.veil) {
          const veilZ = cs.veil.worldZ;
          if (sourceChildInst && sourceChildInst.veilMesh) {
            cp.childVeilMesh.position.copy(sourceChildInst.veilMesh.position);
            cp.childVeilMesh.rotation.copy(sourceChildInst.veilMesh.rotation);
            cp.childVeilMesh.scale.copy(sourceChildInst.veilMesh.scale);
          } else if (cs.veil.fullInnerCanvas) {
            const boreDistBg = Math.max(1.0, tBoreZ - targetObj.sceneBgZ);
            const veilRatio = (tBoreZ - veilZ) / boreDistBg;
            const innerW = SCENE_GEOMETRY_SPEC.bgBaseWidth * targetObj.innerScale;
            const veilScale = (innerW * veilRatio) / SCENE_GEOMETRY_SPEC.fgBaseWidth;
            cp.childVeilMesh.position.set(0.0, tBoreY * (1.0 - veilRatio), veilZ);
            cp.childVeilMesh.scale.set(veilScale, veilScale, 1.0);
          } else {
            cp.childVeilMesh.position.set(cGeom.worldX, cGeom.worldY, veilZ);
            cp.childVeilMesh.scale.set(cGeom.width, cGeom.height, 1.0);
          }
          const childVeilOp = sourceChildInst
            ? sourceChildInst.veilCurrentOpacity
            : typeof cs.veil.restOpacity === 'number'
              ? cs.veil.restOpacity
              : 1.0;
          if (cp.childVeilMat) {
            cp.childVeilMat.opacity = childVeilOp;
          }
          cp.childVeilMesh.visible = childVeilOp > 0.005;
        }
      }
    }
  }

  /**
   * Places a wall-mounted grandchild diorama (e.g. the vitrine inside the shelf cubby's copy of
   * Bar Giora) with the same flush wall transform the standalone host uses, expressed in the host
   * scene's own coordinates (which are the parent dioramaGroup's local coordinates).
   */
  _syncNestedWallMount(cwm, childSpec, hostSceneId, sourceChildInst) {
    const childTargetObj = this.sceneObjects && this.sceneObjects[childSpec.targetSceneId];
    const wmx = this._computeWallMountDioramaTransform(childSpec, hostSceneId);
    if (!wmx || !childTargetObj) {
      cwm.group.visible = false;
      return;
    }
    cwm.group.visible = true;
    const nestedSettle =
      childSpec.wallMount.flatAtRest && sourceChildInst ? this._mountReliefScale(sourceChildInst) : 0.0;
    this._applyMountGroupTransform(cwm.group, wmx, childSpec.wallMount.flatAtRest ? nestedSettle : 1.0);

    const pairs = [
      [cwm.bgMesh, cwm.bgMat, childTargetObj.layer0],
      [cwm.midMesh, cwm.midMat, childTargetObj.layer1],
      [cwm.fgMesh, cwm.fgMat, childTargetObj.layer2]
    ];
    for (const [mesh, mat, src] of pairs) {
      if (!mesh || !src) continue;
      mesh.position.copy(src.position);
      mesh.rotation.copy(src.rotation);
      mesh.scale.copy(src.scale);
      const srcU = src.material && src.material.uniforms;
      const dstU = mat && mat.uniforms;
      if (srcU && dstU) {
        for (const key of ['uDistBg', 'uOpticalCenterLocalX', 'uOpticalCenterLocalY']) {
          if (srcU[key] && dstU[key]) dstU[key].value = srcU[key].value;
        }
      }
    }
    if (cwm.bgMat && cwm.bgMat.uniforms && cwm.bgMat.uniforms.uOuterWallOpacity) {
      const srcOuter =
        sourceChildInst && sourceChildInst.bgMat && sourceChildInst.bgMat.uniforms
          ? sourceChildInst.bgMat.uniforms.uOuterWallOpacity.value
          : 0.0;
      cwm.bgMat.uniforms.uOuterWallOpacity.value = srcOuter;
    }
    if (childSpec.wallMount.flatAtRest) {
      const rs = sourceChildInst ? this._mountReliefScale(sourceChildInst) : 0.0;
      this._applyMountReliefToMaterials([cwm.bgMat, cwm.midMat, cwm.fgMat], childTargetObj, rs);
    }
  }

  /**
   * Depth-relief scale for `flatAtRest` mounts (the Purim photo reads as a flat print on the floor
   * at rest and gains its full 3D garden relief exactly at landing, so t=0 and t=1 both register).
   */
  _applyMountReliefToMaterials(mats, targetObj, reliefScale) {
    const [bgMat, midMat, fgMat] = mats;
    const set = (mat, amp) => {
      const u = mat && mat.uniforms;
      if (u && u.uDepthAmplitude && typeof amp === 'number') u.uDepthAmplitude.value = amp * reliefScale;
    };
    set(bgMat, targetObj.roomDepthAmplitude);
    set(midMat, targetObj.midDepthAmplitude);
    set(fgMat, targetObj.fgDepthAmplitude);
  }

  _mountReliefScale(inst) {
    const wm = inst && inst.spec && inst.spec.wallMount;
    if (!wm || !wm.flatAtRest) return 1.0;
    return typeof inst.reliefScale === 'number' ? inst.reliefScale : 0.0;
  }

  _setWallMountReliefScale(inst, reliefScale) {
    const wm = inst && inst.spec && inst.spec.wallMount;
    if (!wm || !wm.flatAtRest) return;
    inst.reliefScale = THREE.MathUtils.clamp(reliefScale, 0.0, 1.0);
    const targetObj = this.sceneObjects && this.sceneObjects[inst.targetSceneId];
    if (targetObj) {
      this._applyMountReliefToMaterials([inst.bgMat, inst.midMat, inst.fgMat], targetObj, inst.reliefScale);
    }
    if (inst.wallMountXform && inst.dioramaGroup) {
      this._applyMountGroupTransform(inst.dioramaGroup, inst.wallMountXform, inst.reliefScale);
    }
  }

  /**
   * Places a mount group. Surface mounts blend element-wise from the rest affine map (pixel-exact
   * on the print) to the landing similarity as `settle` goes 0 -> 1; other mounts use the similarity.
   */
  _applyMountGroupTransform(group, wmx, settle) {
    if (!wmx.restMatrix) {
      group.matrixAutoUpdate = true;
      group.position.copy(wmx.tDio);
      group.quaternion.copy(wmx.qDio);
      group.scale.setScalar(wmx.lambdaP);
      return;
    }
    const sim = new THREE.Matrix4().compose(
      wmx.tDio,
      wmx.qDio,
      new THREE.Vector3(wmx.lambdaP, wmx.lambdaP, wmx.lambdaP)
    );
    const k = THREE.MathUtils.clamp(settle, 0.0, 1.0);
    const a = wmx.restMatrix.elements;
    const b = sim.elements;
    const out = new THREE.Matrix4();
    for (let i = 0; i < 16; i++) out.elements[i] = a[i] * (1.0 - k) + b[i] * k;
    group.matrixAutoUpdate = false;
    group.matrix.copy(out);
    group.matrix.decompose(group.position, group.quaternion, group.scale);
    group.matrixWorldNeedsUpdate = true;
  }

  _setWallMountOuterWallOpacity(inst, opacity) {
    if (!inst || !inst.spec || !inst.spec.wallMount) return;
    inst.wallMountOuterOpacity = opacity;
    const u = inst.bgMat && inst.bgMat.uniforms;
    if (u && u.uOuterWallOpacity) u.uOuterWallOpacity.value = opacity;
    this._syncDioramaMirrorsForScene(inst.hostSceneId);
  }

  /** Forward/reverse outer-wall reveal for wall-mounted dioramas over `outerWallFadeRangeT`. */
  _wallMountOuterWallOpacity(spec, t, isReverse) {
    const range = Array.isArray(spec.outerWallFadeRangeT) ? spec.outerWallFadeRangeT : [0.78, 0.96];
    const fwdT = isReverse ? 1.0 - t : t;
    const u = THREE.MathUtils.clamp((fwdT - range[0]) / Math.max(1e-5, range[1] - range[0]), 0.0, 1.0);
    return u * u * (3.0 - 2.0 * u);
  }

  _syncDioramaMirrorsForScene(sceneId) {
    if (!sceneId || this._portalInstances.length === 0) return;
    for (const inst of this._portalInstances) {
      if (inst.embedDiorama && inst.dioramaGroup && inst.targetSceneId === sceneId) {
        this._syncSingleDioramaInstance(inst);
      }
    }
  }

  _updatePortalShelfTransforms() {
    for (const [hostSceneId, hostCfg] of Object.entries(this.sceneConfigs)) {
      if (!Array.isArray(hostCfg.portalShelves) || hostCfg.portalShelves.length === 0) continue;
      const hostObj = this.sceneObjects && this.sceneObjects[hostSceneId];
      if (!hostObj) continue;

      const hostRest = this._getSceneRestPose(hostSceneId);
      const hostBoreY = hostCfg.boreSightAtRest ? hostRest.y : 0.0;
      const hostBoreZ = hostCfg.boreSightAtRest ? hostRest.z : this.baseCameraZ;

      for (const spec of hostCfg.portalShelves) {
        const geom = computePortalWorldGeometry(
          spec,
          hostObj.innerScale,
          hostBoreY,
          hostBoreZ,
          hostObj.sceneBgZ,
          this.camera.fov,
          this.camera.aspect
        );
        const targetRest = this._getSceneRestPose(spec.targetSceneId);
        const effFitZ = computePortalEffectiveFitZ(
          geom,
          this.camera.fov,
          this.camera.aspect,
          targetRest.yawDeg,
          targetRest.pitchDeg,
          spec.flightRollDeg || 0.0,
          spec.rotationDeg || 0.0
        );
        spec.worldX = Number(geom.worldX.toFixed(3));
        spec.worldY = Number(geom.worldY.toFixed(3));
        spec.worldZ = Number(geom.worldZ.toFixed(3));
        spec.width = Number(geom.width.toFixed(3));
        spec.height = Number(geom.height.toFixed(3));
        spec.recessDepth = Number(geom.recessDepth.toFixed(3));
        spec.fitZ = Number(geom.fitZ.toFixed(3));
        spec.effectiveFitZ = Number(effFitZ.toFixed(3));
        spec.heightFitZ = Number(geom.heightFitZ.toFixed(3));
        spec.widthFitZ = Number(geom.widthFitZ.toFixed(3));
      }
    }

    if (this._portalInstances.length === 0) return;

    // Pass 1: Update all standalone portal frames, veils, occluders, and companion-fall transforms first
    // so every target scene's mutable state is current before Pass 2 mirrors it into parent dioramas.
    for (const inst of this._portalInstances) {
      const hostObj = this.sceneObjects[inst.hostSceneId];
      const hostCfg = this.sceneConfigs[inst.hostSceneId];
      if (!hostObj || !hostCfg) continue;
      const geom = inst.spec;

      if (inst.frameLine) {
        inst.frameLine.position.set(
          geom.worldX,
          geom.worldY,
          geom.worldZ + SCENE_GEOMETRY_SPEC.portalFrameBevelOffsetZ
        );
        inst.frameLine.rotation.set(0, 0, THREE.MathUtils.degToRad(geom.rotationDeg || 0.0));
        inst.frameLine.scale.set(geom.width, geom.height, 1.0);
      }

      if (
        inst.occluderMesh &&
        inst.spec.occluder &&
        (Array.isArray(inst.spec.occluder.innerBoxPx) || inst.spec.occluder.fullInnerCanvas)
      ) {
        const hostRest = this._getSceneRestPose(inst.hostSceneId);
        const hostBoreY = hostCfg.boreSightAtRest ? hostRest.y : 0.0;
        const hostBoreZ = hostCfg.boreSightAtRest ? hostRest.z : this.baseCameraZ;
        const occZ = inst.spec.occluder.worldZ;
        const occBox = Array.isArray(inst.spec.occluder.innerBoxPx)
          ? inst.spec.occluder.innerBoxPx
          : inst.spec.innerBoxPx;
        inst.occluderBaseGeom = computePortalWorldGeometry(
          {
            innerBoxPx: occBox,
            worldZ: occZ,
            recessDepth: 0
          },
          hostObj.innerScale,
          hostBoreY,
          hostBoreZ,
          hostObj.sceneBgZ,
          this.camera.fov,
          this.camera.aspect
        );
        if (inst.spec.occluder.fullInnerCanvas) {
          const boreDistBg = Math.max(1.0, hostBoreZ - hostObj.sceneBgZ);
          const occRatio = (hostBoreZ - occZ) / boreDistBg;
          const innerWorldWidthAtBg = SCENE_GEOMETRY_SPEC.bgBaseWidth * hostObj.innerScale;
          const occScale = (innerWorldWidthAtBg * occRatio) / SCENE_GEOMETRY_SPEC.fgBaseWidth;
          inst.occluderPlaneScale = occScale;
          inst.occluderPlaneCenterY = hostBoreY * (1.0 - occRatio);
          inst.occluderPlaneWorldW = SCENE_GEOMETRY_SPEC.fgBaseWidth * occScale;
          inst.occluderPlaneWorldH = SCENE_GEOMETRY_SPEC.fgBaseHeight * occScale;
          inst.occluderBoreDist = Math.max(1.0, hostBoreZ - occZ);
          inst.occluderOpticalCenterLocalY =
            (hostBoreY * occRatio) / Math.max(1e-5, occScale);
        }
        this._applyOccluderInstanceTransform(inst, true);
      }

      if (inst.veilMesh && inst.spec.veil) {
        const hostRest = this._getSceneRestPose(inst.hostSceneId);
        const hostBoreY = hostCfg.boreSightAtRest ? hostRest.y : 0.0;
        const hostBoreZ = hostCfg.boreSightAtRest ? hostRest.z : this.baseCameraZ;
        const veilZ = inst.spec.veil.worldZ;
        if (inst.spec.veil.fullInnerCanvas) {
          const boreDistBg = Math.max(1.0, hostBoreZ - hostObj.sceneBgZ);
          const veilRatio = (hostBoreZ - veilZ) / boreDistBg;
          const innerWorldWidthAtBg = SCENE_GEOMETRY_SPEC.bgBaseWidth * hostObj.innerScale;
          const veilScale = (innerWorldWidthAtBg * veilRatio) / SCENE_GEOMETRY_SPEC.fgBaseWidth;
          inst.veilMesh.position.set(0.0, hostBoreY * (1.0 - veilRatio), veilZ);
          inst.veilMesh.scale.set(veilScale, veilScale, 1.0);
        } else {
          inst.veilMesh.position.set(geom.worldX, geom.worldY, veilZ);
          inst.veilMesh.scale.set(geom.width, geom.height, 1.0);
        }
      }
    }

    // Pass 2: Mirror target scene transforms, companion-fall rotations, carpet rigs, and child portals into embedded dioramas.
    for (const inst of this._portalInstances) {
      if (inst.embedDiorama && inst.dioramaGroup) {
        this._syncSingleDioramaInstance(inst);
      }
    }
  }

  _applyOccluderInstanceTransform(inst, skipMirrorSync = false) {
    if (!inst || !inst.occluderMesh || !inst.occluderMat || !inst.occluderBaseGeom) return;
    const g = inst.occluderBaseGeom;
    const f = THREE.MathUtils.clamp(
      typeof inst.occluderAnimFactor === 'number' ? inst.occluderAnimFactor : 1.0,
      0.0,
      1.0
    );
    const liftT = 1.0 - f;
    const scaleBoost = 1.0 + (SCENE_GEOMETRY_SPEC.portalOccluderLiftScale - 1.0) * liftT;
    const hostObj = this.sceneObjects && this.sceneObjects[inst.hostSceneId];
    const activePid = hostObj ? hostObj.activeFlightPortalId : null;
    const applySurrounding = !activePid || activePid === inst.spec.id;
    const surroundingOp = applySurrounding
      ? hostObj && typeof hostObj.portalSurroundingOpacity === 'number'
        ? hostObj.portalSurroundingOpacity
        : inst.hostSceneId === 'shelf_portal'
          ? this._portalSurroundingOpacity
          : 1.0
      : 1.0;
    const effOpacity = f * surroundingOp;

    if (inst.spec.occluder && inst.spec.occluder.fullInnerCanvas) {
      const occZ = inst.spec.occluder.worldZ;
      inst.occluderMesh.position.set(
        0.0,
        inst.occluderPlaneCenterY,
        occZ + SCENE_GEOMETRY_SPEC.portalOccluderLiftZ * liftT
      );
      inst.occluderMesh.scale.set(
        inst.occluderPlaneScale * scaleBoost,
        inst.occluderPlaneScale * scaleBoost,
        1.0
      );
      if (inst.occluderMat.uniforms) {
        inst.occluderMat.uniforms.uDistBg.value = inst.occluderBoreDist;
        inst.occluderMat.uniforms.uOpticalCenterLocalY.value = inst.occluderOpticalCenterLocalY;
        inst.occluderMat.uniforms.uOpacity.value = effOpacity;
      } else {
        inst.occluderMat.opacity = effOpacity;
      }
    } else {
      inst.occluderMesh.position.set(
        g.worldX,
        g.worldY,
        g.worldZ + SCENE_GEOMETRY_SPEC.portalOccluderLiftZ * liftT
      );
      inst.occluderMesh.scale.set(g.width * scaleBoost, g.height * scaleBoost, 1.0);
      inst.occluderMat.opacity = effOpacity;
    }
    inst.occluderMesh.visible = effOpacity > 0.005;

    const cf = inst.spec && inst.spec.occluder && inst.spec.occluder.companionFall;
    if (cf && cf.extraLayerId && hostObj && Array.isArray(hostObj.extraLayers)) {
      const compExtra = hostObj.extraLayers.find((el) => el && el.id === cf.extraLayerId);
      if (compExtra && compExtra.mesh && typeof compExtra.planeWorldW === 'number') {
        const fallT = liftT;
        const theta = THREE.MathUtils.degToRad((cf.fallAngleDeg || 0.0) * fallT);
        const pivPx = Array.isArray(cf.pivotInnerPx) ? cf.pivotInnerPx : [750, 500];
        let localOffX =
          (pivPx[0] / SCENE_GEOMETRY_SPEC.innerFrameWidthPx - 0.5) * compExtra.planeWorldW;
        let localOffY =
          (0.5 - pivPx[1] / SCENE_GEOMETRY_SPEC.innerFrameHeightPx) * compExtra.planeWorldH;
        const meshBaseZ = compExtra.basePosZ !== undefined ? compExtra.basePosZ : compExtra.zPos;
        if (
          typeof cf.pivotWorldZ === 'number' &&
          compExtra.mesh.material &&
          compExtra.mesh.material.uniforms &&
          compExtra.mesh.material.uniforms.uDistBg
        ) {
          const uDistBg = Math.max(1.0, compExtra.mesh.material.uniforms.uDistBg.value);
          const uOptWorldY =
            (compExtra.mesh.material.uniforms.uOpticalCenterLocalY.value || 0.0) *
            (compExtra.baseScale || 1.0);
          const zDisp = cf.pivotWorldZ - meshBaseZ;
          const rayScale = (uDistBg - zDisp) / uDistBg;
          localOffX *= rayScale;
          localOffY = uOptWorldY + (localOffY - uOptWorldY) * rayScale;
        }
        const pivX = (compExtra.basePosX || 0) + localOffX;
        const pivY = (compExtra.basePosY || 0) + localOffY;
        const dx = -localOffX;
        const dy = -localOffY;
        const cosT = Math.cos(theta);
        const sinT = Math.sin(theta);
        compExtra.mesh.position.set(
          pivX + (dx * cosT - dy * sinT) + (cf.shiftWorldX || 0) * fallT,
          pivY + (dx * sinT + dy * cosT) + (cf.shiftWorldY || 0) * fallT,
          meshBaseZ
        );
        compExtra.mesh.rotation.set(0, 0, theta);
      }
    }

    if (!skipMirrorSync) {
      this._syncDioramaMirrorsForScene(inst.hostSceneId);
    }
  }

  _stepPortalOccluderAnimations(nowMs) {
    if (this._occluderAnimations.size === 0) return;
    for (const [portalId, anim] of this._occluderAnimations.entries()) {
      const inst = this._portalInstances.find((p) => p.spec.id === portalId);
      const rawT = THREE.MathUtils.clamp(
        (nowMs - anim.startTimeMs) / Math.max(1.0, anim.durationMs),
        0.0,
        1.0
      );
      const easeOut = 1.0 - Math.pow(1.0 - rawT, 3);
      if (inst) {
        if (anim.mode === 'erase') {
          inst.occluderAnimFactor = anim.fromOpacity * (1.0 - easeOut);
        } else {
          inst.occluderAnimFactor = anim.fromOpacity + (1.0 - anim.fromOpacity) * easeOut;
        }
        this._applyOccluderInstanceTransform(inst);
      }
      if (rawT >= 1.0) {
        this._occluderAnimations.delete(portalId);
      }
    }
  }

  _updatePortalClippingPlanes(unclipPortalId = null) {
    if (this._portalInstances.length === 0 || !this.camera) return;
    const camPos = this.camera.position;

    for (const inst of this._portalInstances) {
      if (!inst.embedDiorama || !inst.clipPlanes || !inst.dioramaGroup || !inst.dioramaGroup.visible) continue;
      if (unclipPortalId && inst.spec.id === unclipPortalId) {
        for (let i = 0; i < 4; i += 1) {
          inst.clipPlanes[i].set(Z_AXIS_UNIT, 1e6);
        }
        continue;
      }
      const s = inst.spec;
      const halfW = s.width * 0.5;
      const halfH = s.height * 0.5;
      const wz = s.worldZ;
      const rotRad = THREE.MathUtils.degToRad(s.rotationDeg || 0.0);
      const cosA = Math.cos(rotRad);
      const sinA = Math.sin(rotRad);
      const rotCorner = (lx, ly) =>
        new THREE.Vector3(
          s.worldX + lx * cosA - ly * sinA,
          s.worldY + lx * sinA + ly * cosA,
          wz
        );

      const tl = rotCorner(-halfW, +halfH);
      const tr = rotCorner(+halfW, +halfH);
      const br = rotCorner(+halfW, -halfH);
      const bl = rotCorner(-halfW, -halfH);

      const center = new THREE.Vector3(s.worldX, s.worldY, wz);
      const rayIntoCubby = center.clone().sub(camPos);
      if (rayIntoCubby.lengthSq() < 1e-6) {
        rayIntoCubby.set(0, 0, -1);
      } else {
        rayIntoCubby.normalize();
      }
      const interiorPoint = center.clone().addScaledVector(rayIntoCubby, s.recessDepth + 12.0);

      const edges = [
        [tl, tr],
        [tr, br],
        [br, bl],
        [bl, tl]
      ];
      for (let i = 0; i < 4; i += 1) {
        const plane = inst.clipPlanes[i];
        plane.setFromCoplanarPoints(camPos, edges[i][0], edges[i][1]);
        if (plane.distanceToPoint(interiorPoint) < 0) {
          plane.negate();
        }
      }
    }
  }

  _raycastPortalShelfAtClient(clientX, clientY) {
    if (!this.hasPortalShelves(this.currentSceneId) || this._portalInstances.length === 0) {
      return null;
    }
    for (const inst of this._portalInstances) {
      if (inst.hostSceneId !== this.currentSceneId) continue;
      const s = inst.spec;
      if (
        inst.occluderBaseGeom &&
        inst.occluderMesh &&
        inst.occluderMesh.visible &&
        !this._erasedPortalOccluders.has(s.id)
      ) {
        const og = inst.occluderBaseGeom;
        if (s.occluder && s.occluder.fullInnerCanvas && inst.occluderTex && inst.occluderTex.image) {
          const sampler = this._getTextureSampler(inst.occluderTex, 512);
          if (sampler) {
            const alphaMin =
              typeof s.occluder.alphaHitThreshold === 'number'
                ? s.occluder.alphaHitThreshold
                : SCENE_GEOMETRY_SPEC.portalOccluderDefaultAlphaHit;
            const depthAmp = typeof s.occluder.depthAmplitude === 'number' ? s.occluder.depthAmplitude : 0.0;
            const zOffsets = depthAmp > 0.0 ? [0.0, 0.45 * depthAmp, 0.85 * depthAmp] : [0.0];
            const hostRest = this._getSceneRestPose(inst.hostSceneId);
            const hostCfg = this.sceneConfigs[inst.hostSceneId];
            const hostBoreY = hostCfg && hostCfg.boreSightAtRest ? hostRest.y : 0.0;
            for (const zOff of zOffsets) {
              const occHit = this._intersectPlaneZ(clientX, clientY, og.worldZ + zOff);
              if (!occHit) continue;
              const scaleAtZ =
                zOff > 0.0
                  ? Math.max(0.2, (inst.occluderBoreDist - zOff) / Math.max(1.0, inst.occluderBoreDist))
                  : 1.0;
              const wx0 = occHit[0] / scaleAtZ;
              const wy0 = (occHit[1] - hostBoreY) / scaleAtZ + hostBoreY;
              const u = wx0 / Math.max(1e-5, inst.occluderPlaneWorldW) + 0.5;
              const v =
                (wy0 - inst.occluderPlaneCenterY) / Math.max(1e-5, inst.occluderPlaneWorldH) + 0.5;
              if (u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0) {
                const sample = sampler.sampleAtUv(u, v);
                if (sample.a >= alphaMin) {
                  return s;
                }
              }
            }
          }
        } else {
          const occHit = this._intersectPlaneZ(clientX, clientY, og.worldZ);
          if (occHit) {
            const odx = Math.abs(occHit[0] - og.worldX);
            const ody = Math.abs(occHit[1] - og.worldY);
            if (odx <= og.width * 0.5 && ody <= og.height * 0.5) {
              return s;
            }
          }
        }
      }
      if (!inst.embedDiorama && this.isPortalLocked(s)) {
        continue;
      }
      const hitPt = this._intersectPlaneZ(clientX, clientY, s.worldZ);
      if (!hitPt) continue;
      const rawDx = hitPt[0] - s.worldX;
      const rawDy = hitPt[1] - s.worldY;
      const rotRad = -THREE.MathUtils.degToRad(s.rotationDeg || 0.0);
      const localX = rawDx * Math.cos(rotRad) - rawDy * Math.sin(rotRad);
      const localY = rawDx * Math.sin(rotRad) + rawDy * Math.cos(rotRad);
      if (Math.abs(localX) <= s.width * 0.5 && Math.abs(localY) <= s.height * 0.5) {
        return s;
      }
    }
    return null;
  }

  _getCarpetUnrollState(sceneId) {
    const cfg = this.sceneConfigs && sceneId ? this.sceneConfigs[sceneId] : null;
    if (!cfg || !cfg.carpetUnrollRig) return null;
    let st = this._carpetUnrollStates.get(sceneId);
    if (!st) {
      st = {
        progress: 0.0,
        target: 0.0,
        animStartMs: 0.0,
        animFrom: 0.0,
        durationMs:
          typeof cfg.carpetUnrollRig.durationMs === 'number'
            ? cfg.carpetUnrollRig.durationMs
            : 1450
      };
      this._carpetUnrollStates.set(sceneId, st);
    }
    return st;
  }

  getCarpetUnrollProgress(sceneId = this.currentSceneId) {
    const st = this._getCarpetUnrollState(sceneId);
    return st ? st.progress : 0.0;
  }

  toggleCarpetUnroll(sceneId = this.currentSceneId, forceUnrolled = null) {
    const st = this._getCarpetUnrollState(sceneId);
    if (!st) return false;
    const nextTarget =
      typeof forceUnrolled === 'boolean' ? (forceUnrolled ? 1.0 : 0.0) : st.target > 0.5 ? 0.0 : 1.0;
    st.animFrom = st.progress;
    st.target = nextTarget;
    st.animStartMs = performance.now();
    if (window.__sendDiag) {
      window.__sendDiag('carpet_unroll_toggled', {
        scene: sceneId,
        unrolled: nextTarget > 0.5
      });
    }
    return nextTarget > 0.5;
  }

  _resolveCarpetMeshRecord(sceneId, layerId) {
    const obj = this.sceneObjects && sceneId ? this.sceneObjects[sceneId] : null;
    if (!obj || !layerId) return null;
    if (layerId === 'foreground' && obj.layer2) {
      return {
        mesh: obj.layer2,
        tex: obj.fgTex,
        basePosX: obj.fgBasePosX || 0,
        basePosY: obj.fgBasePosY || 0,
        basePosZ: obj.sceneFgZ,
        planeWorldW: obj.fgPlaneWorldW || SCENE_GEOMETRY_SPEC.fgBaseWidth,
        planeWorldH: obj.fgPlaneWorldH || SCENE_GEOMETRY_SPEC.fgBaseHeight
      };
    }
    if (layerId === 'midground' && obj.layer1) {
      return {
        mesh: obj.layer1,
        tex: obj.midTex,
        basePosX: obj.midBasePosX || 0,
        basePosY: obj.midBasePosY || 0,
        basePosZ: obj.sceneMidZ,
        planeWorldW: obj.midPlaneWorldW || SCENE_GEOMETRY_SPEC.midBaseWidth,
        planeWorldH: obj.midPlaneWorldH || SCENE_GEOMETRY_SPEC.midBaseHeight
      };
    }
    if (Array.isArray(obj.extraLayers)) {
      const extra = obj.extraLayers.find((el) => el && el.id === layerId);
      if (extra && extra.mesh) {
        return {
          mesh: extra.mesh,
          tex: extra.tex,
          mirrorLeftEdge: Boolean(extra.mirrorLeftEdge),
          basePosX: extra.basePosX || 0,
          basePosY: extra.basePosY || 0,
          basePosZ: extra.basePosZ !== undefined ? extra.basePosZ : extra.zPos,
          planeWorldW: extra.planeWorldW || SCENE_GEOMETRY_SPEC.fgBaseWidth,
          planeWorldH: extra.planeWorldH || SCENE_GEOMETRY_SPEC.fgBaseHeight
        };
      }
    }
    return null;
  }

  _applyCarpetUnrollTransform(sceneId, skipMirrorSync = false) {
    const cfg = this.sceneConfigs && sceneId ? this.sceneConfigs[sceneId] : null;
    if (!cfg || !cfg.carpetUnrollRig) return;
    const st = this._getCarpetUnrollState(sceneId);
    if (!st) return;
    const rig = cfg.carpetUnrollRig;
    const u = THREE.MathUtils.clamp(st.progress, 0.0, 1.0);
    const initMax = typeof rig.initialClipUvMaxX === 'number' ? rig.initialClipUvMaxX : 0.25;
    const fullMax = typeof rig.unrolledClipUvMaxX === 'number' ? rig.unrolledClipUvMaxX : 1.0;
    const minX = typeof rig.clipUvMinX === 'number' ? rig.clipUvMinX : 0.0;
    const clipMaxX = THREE.MathUtils.lerp(initMax, fullMax, u);

    for (const tLayerId of [rig.trailLayerId, rig.reflectedTrailLayerId]) {
      if (!tLayerId) continue;
      const trailRec = this._resolveCarpetMeshRecord(sceneId, tLayerId);
      if (
        trailRec &&
        trailRec.mesh &&
        trailRec.mesh.material &&
        trailRec.mesh.material.uniforms &&
        trailRec.mesh.material.uniforms.uClipUvMaxX
      ) {
        trailRec.mesh.material.uniforms.uClipUvMinX.value = minX;
        trailRec.mesh.material.uniforms.uClipUvMaxX.value = clipMaxX;
      }
    }

    const shiftPxX = typeof rig.rollShiftInnerPxX === 'number' ? rig.rollShiftInnerPxX : 0.0;
    const shiftPxY = typeof rig.rollShiftInnerPxY === 'number' ? rig.rollShiftInnerPxY : 0.0;
    const shiftZ = typeof rig.rollShiftWorldZ === 'number' ? rig.rollShiftWorldZ : 0.0;

    for (const rLayerId of [rig.rollLayerId, rig.reflectedRollLayerId]) {
      if (!rLayerId) continue;
      const rollRec = this._resolveCarpetMeshRecord(sceneId, rLayerId);
      if (!rollRec || !rollRec.mesh) continue;
      let boreRayScale = 1.0;
      if (cfg.boreSightAtRest && typeof rig.rollTrackWorldZ === 'number') {
        const restPose = this._getSceneRestPose(sceneId);
        const distPlane = Math.max(1.0, restPose.z - rollRec.basePosZ);
        boreRayScale = THREE.MathUtils.clamp(
          (restPose.z - rig.rollTrackWorldZ) / distPlane,
          0.2,
          1.5
        );
      }
      const xSign = rollRec.mirrorLeftEdge ? -1.0 : 1.0;
      const worldDx =
        xSign *
        (shiftPxX / SCENE_GEOMETRY_SPEC.innerFrameWidthPx) *
        rollRec.planeWorldW *
        boreRayScale *
        u;
      const worldDy =
        -(shiftPxY / SCENE_GEOMETRY_SPEC.innerFrameHeightPx) *
        rollRec.planeWorldH *
        boreRayScale *
        u;
      rollRec.mesh.position.set(
        rollRec.basePosX + worldDx,
        rollRec.basePosY + worldDy,
        rollRec.basePosZ + shiftZ * u
      );
    }

    if (!skipMirrorSync) {
      this._syncDioramaMirrorsForScene(sceneId);
    }
  }

  _stepCarpetUnrollAnimations(nowMs) {
    for (const [sceneId, st] of this._carpetUnrollStates.entries()) {
      if (Math.abs(st.progress - st.target) < 1e-4) {
        if (st.progress !== st.target) {
          st.progress = st.target;
          this._applyCarpetUnrollTransform(sceneId);
        }
        continue;
      }
      const rawT = THREE.MathUtils.clamp(
        (nowMs - st.animStartMs) / Math.max(1.0, st.durationMs),
        0.0,
        1.0
      );
      const ease = rawT * rawT * (3.0 - 2.0 * rawT);
      st.progress = THREE.MathUtils.lerp(st.animFrom, st.target, ease);
      this._applyCarpetUnrollTransform(sceneId);
    }
  }

  _raycastCarpetUnrollAtClient(clientX, clientY) {
    const cfg = this.sceneConfigs && this.sceneConfigs[this.currentSceneId];
    if (!cfg || !cfg.carpetUnrollRig || !this.canvas || !this.camera) return false;
    const rig = cfg.carpetUnrollRig;
    const st = this._getCarpetUnrollState(this.currentSceneId);
    const u = st ? st.progress : 0.0;
    const initMax = typeof rig.initialClipUvMaxX === 'number' ? rig.initialClipUvMaxX : 0.25;
    const fullMax = typeof rig.unrolledClipUvMaxX === 'number' ? rig.unrolledClipUvMaxX : 1.0;
    const minUvX = typeof rig.clipUvMinX === 'number' ? rig.clipUvMinX : 0.0;
    const maxUvX = THREE.MathUtils.lerp(initMax, fullMax, u);

    const rect = this.canvas.getBoundingClientRect();
    const normX = THREE.MathUtils.clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const normY = THREE.MathUtils.clamp((clientY - rect.top) / Math.max(1, rect.height), 0, 1);
    const ndc = new THREE.Vector2(normX * 2 - 1, -(normY * 2 - 1));
    this.camera.updateMatrixWorld();
    this._raycaster.setFromCamera(ndc, this.camera);

    const rollRec = this._resolveCarpetMeshRecord(this.currentSceneId, rig.rollLayerId);
    const reflRollRec = rig.reflectedRollLayerId
      ? this._resolveCarpetMeshRecord(this.currentSceneId, rig.reflectedRollLayerId)
      : null;
    const trailRec = this._resolveCarpetMeshRecord(this.currentSceneId, rig.trailLayerId);
    const reflTrailRec = rig.reflectedTrailLayerId
      ? this._resolveCarpetMeshRecord(this.currentSceneId, rig.reflectedTrailLayerId)
      : null;
    const rollTrackZ = typeof rig.rollTrackWorldZ === 'number' ? rig.rollTrackWorldZ : null;
    const trailTrackZ = typeof rig.trailTrackWorldZ === 'number' ? rig.trailTrackWorldZ : null;
    const candidates = [
      rollRec ? { rec: rollRec, clipped: false, trackZ: rollTrackZ } : null,
      reflRollRec ? { rec: reflRollRec, clipped: false, trackZ: rollTrackZ } : null,
      trailRec ? { rec: trailRec, clipped: true, trackZ: trailTrackZ } : null,
      reflTrailRec ? { rec: reflTrailRec, clipped: true, trackZ: trailTrackZ } : null
    ].filter(Boolean);

    for (const { rec, clipped, trackZ } of candidates) {
      if (!rec.mesh || !rec.mesh.visible || !rec.tex) continue;
      const sampler = this._getTextureSampler(rec.tex, 512);
      if (!sampler) continue;

      if (
        cfg.boreSightAtRest &&
        typeof trackZ === 'number' &&
        rec.planeWorldW > 1e-4 &&
        rec.planeWorldH > 1e-4
      ) {
        const restPose = this._getSceneRestPose(this.currentSceneId);
        const distPlane = Math.max(1.0, restPose.z - rec.basePosZ);
        const boreRayScale = THREE.MathUtils.clamp(
          (restPose.z - trackZ) / distPlane,
          0.2,
          1.5
        );
        const planeHit = this._intersectPlaneZ(clientX, clientY, trackZ);
        if (planeHit) {
          const optY = cfg.boreSightAtRest ? restPose.y - rec.basePosY : 0.0;
          const shiftDx = rec.mesh.position.x - rec.basePosX;
          const localWorldX = (planeHit[0] - shiftDx) / boreRayScale;
          const localWorldY =
            optY + (planeHit[1] - rec.mesh.position.y - optY) / boreRayScale;
          const uvX = rec.mirrorLeftEdge
            ? -0.5 - localWorldX / rec.planeWorldW
            : localWorldX / rec.planeWorldW + 0.5;
          const uvY = localWorldY / rec.planeWorldH + 0.5;
          const inClipRange = !clipped || (uvX >= minUvX && uvX <= maxUvX);
          if (inClipRange && uvX >= 0.0 && uvX <= 1.0 && uvY >= 0.0 && uvY <= 1.0) {
            const sample = sampler.sampleAtUv(uvX, uvY);
            if (sample.a >= 24) {
              return true;
            }
          }
        }
      }

      const hits = this._raycaster.intersectObject(rec.mesh, false);
      if (hits.length > 0 && hits[0].uv) {
        const uvX = hits[0].uv.x;
        const uvY = hits[0].uv.y;
        if (clipped && (uvX < minUvX || uvX > maxUvX)) continue;
        const sample = sampler.sampleAtUv(uvX, uvY);
        if (sample.a >= 24) {
          return true;
        }
      }
    }
    return false;
  }

  _updatePortalHoverState(clientX, clientY, targetEl = null) {
    const overUi =
      targetEl &&
      typeof targetEl.closest === 'function' &&
      (targetEl.closest('.bottom-dock') ||
        targetEl.closest('.top-bar') ||
        targetEl.closest('#spatial-sidebar') ||
        targetEl.closest('#spatial-nav-bar') ||
        targetEl.closest('#info-drawer') ||
        targetEl.closest('#beta-drawer'));

    const carpetHovered =
      !overUi &&
      !this.interactionCaptureActive &&
      !this.isTransitioning &&
      !this._portalFlight &&
      this._raycastCarpetUnrollAtClient(clientX, clientY);
    this._hoveredCarpetSceneId = carpetHovered ? this.currentSceneId : null;

    if (
      !this.hasPortalShelves(this.currentSceneId) ||
      this.interactionCaptureActive ||
      this.isTransitioning ||
      this._portalFlight
    ) {
      if (this.hoveredPortalId !== null) {
        this.hoveredPortalId = null;
        for (const inst of this._portalInstances) {
          if (inst.frameMat) {
            inst.frameMat.opacity =
              typeof inst.spec.restFrameOpacity === 'number' ? inst.spec.restFrameOpacity : 0.24;
          }
          if (inst.spec.veil) {
            inst.veilTargetOpacity = inst.spec.veil.restOpacity;
          }
        }
        this.onPortalHover(null, clientX, clientY);
      }
      if (this.canvas && !this._isMouseDragging) {
        this.canvas.style.cursor = carpetHovered ? 'pointer' : 'grab';
      }
      return;
    }

    const hitSpec = overUi ? null : this._raycastPortalShelfAtClient(clientX, clientY);
    const nextHoverId = hitSpec ? hitSpec.id : null;

    if (nextHoverId !== this.hoveredPortalId) {
      this.hoveredPortalId = nextHoverId;
      for (const inst of this._portalInstances) {
        if (inst.hostSceneId === this.currentSceneId) {
          if (inst.frameMat) {
            const restOp =
              typeof inst.spec.restFrameOpacity === 'number' ? inst.spec.restFrameOpacity : 0.24;
            inst.frameMat.opacity = inst.spec.id === nextHoverId ? 0.92 : restOp;
          }
          if (inst.spec.veil) {
            inst.veilTargetOpacity =
              inst.spec.id === nextHoverId && !this.isPortalLocked(inst.spec)
                ? inst.spec.veil.hoverOpacity
                : inst.spec.veil.restOpacity;
          }
        }
      }
      if (hitSpec && hitSpec.targetSceneId) {
        const hoverTargetObj = this.sceneObjects[hitSpec.targetSceneId];
        if (hoverTargetObj && hoverTargetObj.texturesReady && !hoverTargetObj.gpuPrewarmed) {
          this._prewarmSceneGpuResources(hitSpec.targetSceneId);
        }
      }
    }
    if (this.canvas && !this._isMouseDragging) {
      this.canvas.style.cursor = hitSpec || carpetHovered ? 'pointer' : 'grab';
    }
    this.onPortalHover(hitSpec, clientX, clientY);
  }

  getPortalShelves(sceneId = this.currentSceneId) {
    const targetId = sceneId || 'shelf_portal';
    const cfg = this.sceneConfigs && this.sceneConfigs[targetId];
    return cfg && Array.isArray(cfg.portalShelves) ? cfg.portalShelves : [];
  }

  _getAllPortalShelves() {
    const all = [];
    for (const cfg of Object.values(this.sceneConfigs || {})) {
      if (cfg && Array.isArray(cfg.portalShelves)) {
        for (const s of cfg.portalShelves) {
          all.push(s);
        }
      }
    }
    return all;
  }

  _findPortalSpec(portalIdOrSpec) {
    if (portalIdOrSpec && typeof portalIdOrSpec === 'object') {
      return portalIdOrSpec;
    }
    if (typeof portalIdOrSpec !== 'string') {
      return null;
    }
    const activeShelves = this.getPortalShelves(this.currentSceneId);
    const inActive = activeShelves.find(
      (s) => s.id === portalIdOrSpec || s.targetSceneId === portalIdOrSpec
    );
    if (inActive) return inActive;
    return (
      this._getAllPortalShelves().find(
        (s) => s.id === portalIdOrSpec || s.targetSceneId === portalIdOrSpec
      ) || null
    );
  }

  isPortalLocked(portalIdOrSpec) {
    const spec = this._findPortalSpec(portalIdOrSpec);
    if (!spec || !spec.occluder || !spec.occluder.locksPortal) return false;
    return !this._erasedPortalOccluders.has(spec.id);
  }

  getErasedPortalOccluderCount(sceneId = this.currentSceneId) {
    if (!sceneId) return this._erasedPortalOccluders.size;
    const shelves = this.getPortalShelves(sceneId);
    let count = 0;
    for (const s of shelves) {
      if (this._erasedPortalOccluders.has(s.id)) {
        count += 1;
      }
    }
    return count;
  }

  _clearOccluderRegenTimer(portalId) {
    if (!this._occluderRegenTimers || !portalId) return;
    const existing = this._occluderRegenTimers.get(portalId);
    if (existing) {
      clearTimeout(existing);
      this._occluderRegenTimers.delete(portalId);
    }
  }

  _clearAllOccluderRegenTimers(onlyIds = null) {
    if (!this._occluderRegenTimers) return;
    if (Array.isArray(onlyIds)) {
      for (const pid of onlyIds) {
        this._clearOccluderRegenTimer(pid);
      }
      return;
    }
    for (const timerId of this._occluderRegenTimers.values()) {
      clearTimeout(timerId);
    }
    this._occluderRegenTimers.clear();
  }

  _scheduleOccluderRegen(spec, customDelayMs = null) {
    if (!spec || !spec.id) return;
    this._clearOccluderRegenTimer(spec.id);
    const delayMs =
      typeof customDelayMs === 'number'
        ? customDelayMs
        : this._rigSpec && typeof this._rigSpec.portalOccluderRegenDelayMs === 'number'
          ? this._rigSpec.portalOccluderRegenDelayMs
          : SCENE_GEOMETRY_SPEC.portalOccluderRegenDelayMs;
    if (!(delayMs > 0)) return;

    const timerId = setTimeout(() => {
      this._occluderRegenTimers.delete(spec.id);
      if (this._disposed || !this._erasedPortalOccluders.has(spec.id)) return;
      const activeFlightOnSamePortal =
        (this._portalFlight &&
          this._portalFlight.portalSpec &&
          this._portalFlight.portalSpec.id === spec.id) ||
        this._pendingPortalFlightId === spec.id;
      const activeVisitorInsideTarget =
        (spec.targetSceneId && this.currentSceneId === spec.targetSceneId) ||
        this._portalOriginStack.some(
          (e) =>
            e &&
            (e.portalId === spec.id ||
              (spec.targetSceneId && e.hostSceneId === spec.targetSceneId))
        );
      if (activeFlightOnSamePortal || activeVisitorInsideTarget) {
        this._scheduleOccluderRegen(spec, SCENE_GEOMETRY_SPEC.occluderActiveVisitorDeferMs);
        return;
      }
      this.restoreSinglePortalOccluder(spec.id, { autoRegenerated: true });
    }, delayMs);
    this._occluderRegenTimers.set(spec.id, timerId);
  }

  _ensurePortalOpenForEmergence(portalIdOrSpec) {
    const spec = this._findPortalSpec(portalIdOrSpec);
    if (!spec || !spec.occluder) return;
    const wasLocked = !this._erasedPortalOccluders.has(spec.id);
    this._erasedPortalOccluders.add(spec.id);
    this._occluderAnimations.delete(spec.id);
    this._scheduleOccluderRegen(spec);
    const inst = this._portalInstances.find((p) => p.spec.id === spec.id);
    if (inst) {
      inst.occluderAnimFactor = 0.0;
      this._applyOccluderInstanceTransform(inst);
    }
    if (wasLocked) {
      const hostSceneId = spec.hostSceneId || this.currentSceneId;
      const hostErasedCount = this.getErasedPortalOccluderCount(hostSceneId);
      this.onPortalUnlock(spec, hostErasedCount);
    }
  }

  erasePortalOccluder(portalIdOrSpec) {
    const spec = this._findPortalSpec(portalIdOrSpec);
    if (!spec || !spec.occluder || this._erasedPortalOccluders.has(spec.id)) {
      return false;
    }
    this._erasedPortalOccluders.add(spec.id);
    this._scheduleOccluderRegen(spec);
    const centerHint = document.getElementById('center-hint');
    if (centerHint) {
      centerHint.classList.add('hidden');
    }
    const inst = this._portalInstances.find((p) => p.spec.id === spec.id);
    if (inst && inst.occluderMesh && inst.occluderMat) {
      this._occluderAnimations.set(spec.id, {
        mode: 'erase',
        startTimeMs: performance.now(),
        durationMs: SCENE_GEOMETRY_SPEC.portalOccluderDissolveMs,
        fromOpacity: inst.occluderAnimFactor
      });
    }
    if (spec.targetSceneId && this.sceneObjects && this.sceneObjects[spec.targetSceneId]) {
      const targetObj = this.sceneObjects[spec.targetSceneId];
      if (!targetObj.texturesReady && !targetObj.loadPromise) {
        this._loadSceneTextures(spec.targetSceneId, false).catch(() => {});
      }
    }
    const hostSceneId = spec.hostSceneId || this.currentSceneId;
    const hostErasedCount = this.getErasedPortalOccluderCount(hostSceneId);
    if (window.__sendDiag) {
      window.__sendDiag('portal_occluder_erased', {
        portal: spec.id,
        hostScene: hostSceneId,
        erasedCount: hostErasedCount
      });
    }
    this.onPortalUnlock(spec, hostErasedCount);
    return true;
  }

  restoreSinglePortalOccluder(portalIdOrSpec, options = {}) {
    const spec = this._findPortalSpec(portalIdOrSpec);
    if (!spec || !this._erasedPortalOccluders.has(spec.id)) {
      return false;
    }
    this._clearOccluderRegenTimer(spec.id);
    this._erasedPortalOccluders.delete(spec.id);
    const inst = this._portalInstances.find((p) => p.spec.id === spec.id);
    if (inst && inst.occluderMesh && inst.occluderMat) {
      inst.occluderMesh.visible = true;
      this._occluderAnimations.set(spec.id, {
        mode: 'restore',
        startTimeMs: performance.now(),
        durationMs: SCENE_GEOMETRY_SPEC.portalOccluderDissolveMs,
        fromOpacity: inst.occluderAnimFactor
      });
    }
    const hostSceneId = spec.hostSceneId || this.currentSceneId;
    const hostErasedCount = this.getErasedPortalOccluderCount(hostSceneId);
    if (window.__sendDiag) {
      window.__sendDiag('portal_occluder_regenerated', {
        portal: spec.id,
        hostScene: hostSceneId,
        auto: Boolean(options.autoRegenerated),
        erasedCount: hostErasedCount
      });
    }
    this.onPortalUnlock(spec, hostErasedCount);
    return true;
  }

  restoreAllPortalOccluders(sceneId = this.currentSceneId) {
    const targetSpecs = sceneId ? this.getPortalShelves(sceneId) : this._getAllPortalShelves();
    const restoredIds = [];
    for (const spec of targetSpecs) {
      if (this._erasedPortalOccluders.has(spec.id)) {
        this._clearOccluderRegenTimer(spec.id);
        this._erasedPortalOccluders.delete(spec.id);
        restoredIds.push(spec.id);
      }
    }
    if (restoredIds.length === 0) return 0;
    const nowMs = performance.now();
    for (const pid of restoredIds) {
      const inst = this._portalInstances.find((p) => p.spec.id === pid);
      if (inst && inst.occluderMesh && inst.occluderMat) {
        inst.occluderMesh.visible = true;
        this._occluderAnimations.set(pid, {
          mode: 'restore',
          startTimeMs: nowMs,
          durationMs: SCENE_GEOMETRY_SPEC.portalOccluderDissolveMs,
          fromOpacity: inst.occluderAnimFactor
        });
      }
    }
    if (window.__sendDiag) {
      window.__sendDiag('portal_occluders_restored', {
        hostScene: sceneId || 'all',
        count: restoredIds.length
      });
    }
    this.onPortalUnlock(null, 0);
    return restoredIds.length;
  }

  _computePortalFlightRimAndTint(spec, camZ, ease, isReverse = false) {
    const effFitZ =
      typeof spec.effectiveFitZ === 'number'
        ? spec.effectiveFitZ
        : typeof spec.fitZ === 'number'
          ? spec.fitZ
          : 80.0;
    let apertureFillRatio = effFitZ / Math.max(1e-4, Math.max(effFitZ, camZ - spec.worldZ));
    const wmInst = spec.wallMount
      ? this._portalInstances.find((p) => p.spec.id === spec.id && p.wallMountXform)
      : null;
    if (wmInst) {
      // Tilted, rolled wall mount: measure fill by true 3D distance to the mounted box rather than Z.
      const wmx = wmInst.wallMountXform;
      const dArrival = wmx.arrivalCamPos.distanceTo(wmx.hostAnchor);
      const dCam = this.camera.position.distanceTo(wmx.hostAnchor);
      apertureFillRatio = dArrival / Math.max(1e-4, Math.max(dArrival, dCam));
    }
    const fillStart = SCENE_GEOMETRY_SPEC.portalRimDissolveFillStart;
    const fillT = THREE.MathUtils.clamp(
      (apertureFillRatio - fillStart) / Math.max(1e-5, 1.0 - fillStart),
      0.0,
      1.0
    );
    const rimDissolve = fillT * fillT * (3.0 - 2.0 * fillT);
    const rimOpacity = 1.0 - rimDissolve;

    const restTint =
      Array.isArray(spec.dioramaRestTintRgb) && spec.dioramaRestTintRgb.length === 3
        ? spec.dioramaRestTintRgb
        : SCENE_GEOMETRY_SPEC.neutralTintRgb;
    const neutral = SCENE_GEOMETRY_SPEC.neutralTintRgb;
    const chromaProgress = isReverse ? 1.0 - ease : ease;
    const tintRgb = [
      THREE.MathUtils.lerp(restTint[0], neutral[0], chromaProgress),
      THREE.MathUtils.lerp(restTint[1], neutral[1], chromaProgress),
      THREE.MathUtils.lerp(restTint[2], neutral[2], chromaProgress)
    ];
    return { rimOpacity, tintRgb, apertureFillRatio };
  }

  _setHostPortalSurroundingOpacity(hostSceneId, opacity, activePortalId = null) {
    const hostObj = this.sceneObjects && this.sceneObjects[hostSceneId];
    if (!hostObj) return;
    const clampedOp = THREE.MathUtils.clamp(opacity, 0.0, 1.0);
    hostObj.portalSurroundingOpacity = clampedOp;
    hostObj.activeFlightPortalId = activePortalId;
    if (hostSceneId === 'shelf_portal') {
      this._portalSurroundingOpacity = clampedOp;
    }
    if (
      hostObj.layer0 &&
      hostObj.layer0.material &&
      hostObj.layer0.material.uniforms &&
      hostObj.layer0.material.uniforms.uFadeToClear
    ) {
      // Keep hostObj.layer0 opaque and at full natural luminance (uFadeToClear = 0.0):
      // never darken the host scene toward clearColorHex (0x121314) during portal flight.
      hostObj.layer0.material.transparent = false;
      hostObj.layer0.material.uniforms.uOpacity.value = 1.0;
      hostObj.layer0.material.uniforms.uFadeToClear.value = 0.0;
    }
    if (hostObj.layer1 && hostObj.layer1.material) {
      if (hostObj.layer1.material.uniforms && hostObj.layer1.material.uniforms.uOpacity) {
        hostObj.layer1.material.transparent =
          hostObj.midAlphaCutout > 0.0 ? clampedOp < 0.999 : true;
        hostObj.layer1.material.uniforms.uOpacity.value = clampedOp;
      } else {
        hostObj.layer1.material.opacity = clampedOp;
      }
      hostObj.layer1.visible = clampedOp > 0.005;
    }
    if (hostObj.layer2 && hostObj.layer2.material) {
      if (hostObj.layer2.material.uniforms && hostObj.layer2.material.uniforms.uOpacity) {
        hostObj.layer2.material.transparent =
          hostObj.fgAlphaCutout > 0.0 ? clampedOp < 0.999 : true;
        hostObj.layer2.material.uniforms.uOpacity.value = clampedOp;
      } else {
        hostObj.layer2.material.opacity = clampedOp;
      }
      hostObj.layer2.visible = clampedOp > 0.005;
    }
    if (Array.isArray(hostObj.extraLayers)) {
      const camZ = this.camera ? this.camera.position.z : this.baseCameraZ;
      const fadeStart = SCENE_GEOMETRY_SPEC.nearPassFadeStartDist;
      const fadeFull = SCENE_GEOMETRY_SPEC.nearPassFadeFullDist;
      for (const extra of hostObj.extraLayers) {
        if (!extra || !extra.mesh || !extra.mesh.material) continue;
        const exBase = typeof extra.baseOpacity === 'number' ? extra.baseOpacity : 1.0;
        let nearFade = 1.0;
        if (!(extra.alphaCutout > 0.0)) {
          const dist = camZ - extra.zPos;
          const nt = THREE.MathUtils.clamp(
            (dist - fadeStart) / Math.max(1e-5, fadeFull - fadeStart),
            0.0,
            1.0
          );
          nearFade = dist <= fadeStart ? 0.0 : nt * nt * (3.0 - 2.0 * nt);
        }
        const nextOp = exBase * clampedOp * nearFade;
        extra.mesh.visible = nextOp > 0.005;
        if (extra.mesh.material.uniforms && extra.mesh.material.uniforms.uOpacity) {
          extra.mesh.material.transparent = extra.alphaCutout > 0.0 ? nextOp < 0.999 : true;
          extra.mesh.material.uniforms.uOpacity.value = nextOp;
        } else {
          extra.mesh.material.opacity = nextOp;
        }
      }
    }
    if (hostObj.layer3 && hostObj.layer3.material) {
      hostObj.layer3.material.opacity = 0.28 * clampedOp;
    }
    for (const inst of this._portalInstances) {
      if (inst.hostSceneId !== hostSceneId) continue;
      const isSelected = Boolean(activePortalId && inst.spec.id === activePortalId);
      const inSpecificFlight = Boolean(activePortalId);
      const restFrameOp =
        typeof inst.spec.restFrameOpacity === 'number' ? inst.spec.restFrameOpacity : 0.24;
      if (inst.dioramaGroup) {
        const targetReady = Boolean(
          this.sceneObjects[inst.targetSceneId] &&
            this.sceneObjects[inst.targetSceneId].texturesReady
        );
        const placed = !inst.spec.wallMount || Boolean(inst.wallMountXform);
        const allowVisible = placed && (inst.hostSceneId === 'shelf_portal' ? true : targetReady);
        inst.dioramaGroup.visible = placed && isSelected
          ? true
          : inSpecificFlight
            ? allowVisible
            : allowVisible && clampedOp > 0.02;
      }
      const frameScale = isSelected || !inSpecificFlight ? clampedOp : 1.0;
      if (inst.frameLine) {
        inst.frameLine.visible = frameScale > 0.02 && (isSelected || restFrameOp > 0.001);
      }
      if (inst.frameMat) {
        inst.frameMat.opacity = (isSelected ? 0.92 : restFrameOp) * frameScale;
      }
      if (inst.dustMat) {
        inst.dustMat.opacity = isSelected || inSpecificFlight ? 0.28 : 0.28 * clampedOp;
      }
      if (inst.occluderMesh && inst.occluderMat) {
        this._applyOccluderInstanceTransform(inst, true);
      }
      if (inst.veilMesh && inst.veilMat && inst.spec.veil) {
        const veilScale = isSelected || !inSpecificFlight ? clampedOp : 1.0;
        const effVeilOp = inst.veilCurrentOpacity * veilScale;
        inst.veilMat.opacity = effVeilOp;
        inst.veilMesh.visible = effVeilOp > 0.005;
      }
    }
  }

  _setShelfPortalSurroundingOpacity(opacity, activePortalId = null) {
    this._setHostPortalSurroundingOpacity('shelf_portal', opacity, activePortalId);
  }

  _syncPortalOriginFromStack() {
    const top =
      this._portalOriginStack.length > 0
        ? this._portalOriginStack[this._portalOriginStack.length - 1]
        : null;
    this.enteredViaPortalId = top ? top.portalId : null;
    this.enteredFromPortalSceneId = top ? top.hostSceneId : null;
  }

  getPortalAncestryStack() {
    return this._portalOriginStack.map((entry) => ({
      hostSceneId: entry.hostSceneId,
      portalId: entry.portalId
    }));
  }

  setPortalAncestryStack(rawStack = []) {
    if (!Array.isArray(rawStack)) {
      this._portalOriginStack = [];
    } else {
      this._portalOriginStack = rawStack
        .filter((e) => e && typeof e.hostSceneId === 'string' && this.sceneConfigs[e.hostSceneId])
        .map((e) => ({
          hostSceneId: e.hostSceneId,
          portalId: e.portalId || null
        }));
    }
    this._syncPortalOriginFromStack();
  }

  returnToPortalAncestor(stackIdx) {
    if (this.isTransitioning || this._portalFlight) return false;
    if (
      typeof stackIdx !== 'number' ||
      stackIdx < 0 ||
      stackIdx >= this._portalOriginStack.length
    ) {
      return false;
    }
    if (stackIdx === this._portalOriginStack.length - 1) {
      return this.returnToPortalHub();
    }
    const targetAncestor = this._portalOriginStack[stackIdx];
    this._portalOriginStack.length = stackIdx;
    this._syncPortalOriginFromStack();
    return this.transitionTo(targetAncestor.hostSceneId, { keepPortalOrigin: true });
  }

  _recordPortalHop(hostSceneId, portalId, targetSceneId) {
    const existingTargetIdx = this._portalOriginStack.findIndex(
      (entry) => entry.hostSceneId === targetSceneId
    );
    if (existingTargetIdx >= 0) {
      this._portalOriginStack.length = existingTargetIdx;
    } else {
      const existingHostIdx = this._portalOriginStack.findIndex(
        (entry) => entry.hostSceneId === hostSceneId
      );
      if (existingHostIdx >= 0) {
        this._portalOriginStack.length = existingHostIdx;
      }
      this._portalOriginStack.push({ hostSceneId, portalId });
    }
    this._syncPortalOriginFromStack();
  }

  flyIntoPortalShelf(portalIdOrSpec) {
    if (this.isTransitioning || this._portalFlight) return false;
    const spec = this._findPortalSpec(portalIdOrSpec);
    if (!spec || !spec.targetSceneId || !this.sceneConfigs[spec.targetSceneId]) {
      return false;
    }

    const hostSceneId = spec.hostSceneId || 'shelf_portal';
    if (this.currentSceneId !== hostSceneId) {
      this._recordPortalHop(hostSceneId, spec.id, spec.targetSceneId);
      return this.transitionTo(spec.targetSceneId, { keepPortalOrigin: true });
    }

    if (this.isPortalLocked(spec)) {
      this.erasePortalOccluder(spec);
      return false;
    }

    if (spec.portalMode === 'reverse_transition') {
      this._recordPortalHop(hostSceneId, spec.id, spec.targetSceneId);
      return this.transitionTo(spec.targetSceneId, { keepPortalOrigin: true });
    }

    this._updatePortalShelfTransforms();
    this.hoveredPortalId = null;
    this.onPortalHover(null, 0, 0);

    // Seamless reveal contract: the camera flight only starts once EVERY target layer texture
    // (bg, depth, mid, fg, reflection, extra layers + their depth maps) is decoded, uploaded to the
    // GPU and the target scene's shader programs are compiled, so nothing pops in mid-flight or at landing.
    const targetObj = this.sceneObjects[spec.targetSceneId];
    if (targetObj && targetObj.texturesReady) {
      this._startPortalFlight(spec);
      return true;
    }
    if (this._pendingPortalFlightId) return false;
    this._pendingPortalFlightId = spec.id;
    if (this.canvas) {
      this.canvas.style.cursor = 'progress';
    }
    this._loadSceneTextures(spec.targetSceneId, false).then(
      () => {
        const stillValid =
          this._pendingPortalFlightId === spec.id &&
          this.currentSceneId === hostSceneId &&
          !this.isTransitioning &&
          !this._portalFlight;
        this._pendingPortalFlightId = null;
        if (stillValid) {
          this._startPortalFlight(spec);
        }
      },
      (err) => {
        this._pendingPortalFlightId = null;
        console.error(
          `Portal flight aborted: required textures for scene "${spec.targetSceneId}" failed to load`,
          err
        );
        if (window.__sendDiag) {
          window.__sendDiag('portal_flight_load_error', {
            portal: spec.id,
            to: spec.targetSceneId,
            err: String(err && (err.message || err))
          });
        }
        if (this.canvas) {
          this.canvas.style.cursor = 'grab';
        }
      }
    );
    return true;
  }

  /**
   * Uploads every texture of a fully loaded scene to the GPU and compiles its shader programs
   * (the portal diorama uses clipping-plane program variants, so the target's own programs would
   * otherwise compile on the landing frame and cause a visible hitch).
   */
  _prewarmSceneGpuResources(sceneId) {
    const obj = this.sceneObjects[sceneId];
    if (!obj || !obj.texturesReady) {
      throw new Error(`Cannot prewarm GPU resources for scene "${sceneId}" before its textures load`);
    }
    if (obj.gpuPrewarmed) return;
    if (this.hasPortalShelves(sceneId)) {
      this._ensurePortalShelvesInitialized(sceneId);
    }
    const textures = [
      obj.bgTex,
      obj.depthTex,
      obj.midTex,
      obj.midDepthTex,
      obj.fgTex,
      obj.fgDepthTex,
      obj.refTex
    ];
    for (const extra of obj.extraLayers || []) {
      textures.push(extra.tex, extra.depthTex);
    }
    for (const inst of this._portalInstances) {
      if (inst.hostSceneId === sceneId) {
        textures.push(inst.occluderTex, inst.occluderDepthTex, inst.veilTex);
        for (const cp of inst.childPortalMeshes || []) {
          textures.push(cp.childOccTex, cp.childOccDepthTex, cp.childVeilTex);
        }
      }
    }
    for (const tex of textures) {
      if (tex && tex.image) {
        this.renderer.initTexture(tex);
      }
    }
    const prevClip = this.renderer.localClippingEnabled;
    const needsClip = this._sceneNeedsLocalClipping(sceneId);
    this.renderer.localClippingEnabled = needsClip;
    this.renderer.compile(obj.scene, this.camera);
    this.renderer.localClippingEnabled = prevClip;
    obj.gpuPrewarmed = true;
  }

  _computeDirectPortalTargetPose(spec, hostSceneId) {
    const flightRollDeg = typeof spec.flightRollDeg === 'number' ? spec.flightRollDeg : 0.0;
    const rawFitDist = Math.max(
      SCENE_GEOMETRY_SPEC.cameraNearPlane + 6.0,
      SCENE_GEOMETRY_SPEC.directPortalMinFitDistance,
      (typeof spec.fitZ === 'number' ? spec.fitZ : 110.0) * SCENE_GEOMETRY_SPEC.directPortalFitScale
    );
    const hostObj = this.sceneObjects && hostSceneId ? this.sceneObjects[hostSceneId] : null;
    const hostCfg = this.sceneConfigs && hostSceneId ? this.sceneConfigs[hostSceneId] : null;
    const sceneBgZ =
      hostCfg && typeof hostCfg.bgZ === 'number' ? hostCfg.bgZ : SCENE_GEOMETRY_SPEC.bgZ;
    const layer0Scale =
      hostObj && typeof hostObj.layer0Scale === 'number' ? hostObj.layer0Scale : this._layer0Scale;
    const targetZ = Math.max(
      sceneBgZ + SCENE_GEOMETRY_SPEC.nearPassFadeFullDist + 4.0,
      spec.worldZ + rawFitDist
    );

    const distBg = Math.max(1.0, targetZ - sceneBgZ);
    const fovDeg = this.camera ? this.camera.fov : SCENE_GEOMETRY_SPEC.desktopFovDeg;
    const aspect = this.camera ? this.camera.aspect : this.width / Math.max(1, this.height);
    const tanY = Math.tan(THREE.MathUtils.degToRad(fovDeg * 0.5));
    const tanX = tanY * aspect;
    const rollRad = THREE.MathUtils.degToRad(flightRollDeg);
    const absCos = Math.abs(Math.cos(rollRad));
    const absSin = Math.abs(Math.sin(rollRad));
    const rotHalfVisW = distBg * (tanX * absCos + tanY * absSin);
    const rotHalfVisH = distBg * (tanX * absSin + tanY * absCos);
    const maxHalfW =
      0.5 * SCENE_GEOMETRY_SPEC.bgBaseWidth * layer0Scale * SCENE_GEOMETRY_SPEC.frustumEdgeSafetyRatio;
    const maxHalfH =
      0.5 * SCENE_GEOMETRY_SPEC.bgBaseHeight * layer0Scale * SCENE_GEOMETRY_SPEC.frustumEdgeSafetyRatio;
    const safeSpanX = Math.max(0.0, maxHalfW - rotHalfVisW);
    const safeSpanY = Math.max(0.0, maxHalfH - rotHalfVisH);

    return {
      x: THREE.MathUtils.clamp(spec.worldX, -safeSpanX, safeSpanX),
      y: THREE.MathUtils.clamp(spec.worldY, -safeSpanY, safeSpanY),
      z: targetZ,
      yawDeg: 0.0,
      pitchDeg: 0.0,
      rollDeg: flightRollDeg
    };
  }

  _startPortalFlight(spec) {
    this._prewarmSceneGpuResources(spec.targetSceneId);
    if (this.hasPortalShelves(spec.targetSceneId)) {
      this._ensurePortalShelvesInitialized(spec.targetSceneId);
      this._setHostPortalSurroundingOpacity(spec.targetSceneId, 1.0, null);
    }
    this._updatePortalShelfTransforms();
    if (this.canvas) {
      this.canvas.style.cursor = 'default';
    }

    this._foldActiveGyroIntoPose();
    this._hoverNorm.targetX = 0;
    this._hoverNorm.targetY = 0;
    this.pointer.targetX = 0;
    this.pointer.targetY = 0;

    const hostSceneId = spec.hostSceneId || this.currentSceneId;
    const embedDiorama = spec.embedDiorama !== false;
    const flightRollDeg = typeof spec.flightRollDeg === 'number' ? spec.flightRollDeg : 0.0;
    const targetRest = this._getSceneRestPose(spec.targetSceneId);
    const effFitZ = typeof spec.effectiveFitZ === 'number' ? spec.effectiveFitZ : spec.fitZ;
    let toPose = embedDiorama
      ? {
          x: spec.worldX,
          y: spec.worldY,
          z: spec.worldZ + effFitZ,
          yawDeg: targetRest.yawDeg,
          pitchDeg: targetRest.pitchDeg,
          rollDeg: flightRollDeg
        }
      : this._computeDirectPortalTargetPose(spec, hostSceneId);

    const flightInst = this._portalInstances.find((p) => p.spec.id === spec.id);
    let quatPath = null;
    if (embedDiorama && spec.wallMount && flightInst && flightInst.wallMountXform) {
      // Land exactly on the camera pose that sees the flush, rolled diorama as the standalone
      // target scene sees itself at rest; orientation slerps alongside position (no Euler lerp).
      const wmx = flightInst.wallMountXform;
      toPose = this._poseFromPositionQuat(wmx.arrivalCamPos, wmx.arrivalCamQuat);
      quatPath = { from: this.camera.quaternion.clone(), to: wmx.arrivalCamQuat.clone() };
    }
    if (flightInst && flightInst.spec.veil) {
      flightInst.veilFlightStartOpacity = flightInst.veilCurrentOpacity;
    }
    if (embedDiorama && flightInst) {
      const initRimAndTint = this._computePortalFlightRimAndTint(
        spec,
        this.camera.position.z,
        0.0,
        false
      );
      this._applyDioramaInstanceTint(flightInst, initRimAndTint.tintRgb);
    }

    const flightDurationMs = embedDiorama
      ? this._rigSpec.portalFlightDurationMs
      : SCENE_GEOMETRY_SPEC.directPortalFlightDurationMs;

    this._portalFlight = {
      reverse: false,
      portalSpec: spec,
      hostSceneId,
      targetSceneId: spec.targetSceneId,
      embedDiorama,
      flightRollDeg,
      startTimeMs: performance.now(),
      durationMs: flightDurationMs,
      fromPose: {
        x: this.camera.position.x,
        y: this.camera.position.y,
        z: this.camera.position.z,
        yawDeg: window.__PROTOTYPE_STATE__.cameraRotationDeg.yaw,
        pitchDeg: window.__PROTOTYPE_STATE__.cameraRotationDeg.pitch,
        rollDeg: window.__PROTOTYPE_STATE__.cameraRotationDeg.roll || 0
      },
      toPose,
      quatPath
    };
    window.__PROTOTYPE_STATE__.isTransitioning = true;
    if (window.__sendDiag) {
      window.__sendDiag('portal_flight_start', {
        portal: spec.id,
        from: hostSceneId,
        to: spec.targetSceneId,
        fitZ: effFitZ,
        toZ: Number(toPose.z.toFixed(2)),
        rollDeg: flightRollDeg,
        embedDiorama
      });
    }
    return true;
  }

  _startReversePortalFlight(spec, returnHost) {
    const hostObj = this.sceneObjects && this.sceneObjects[returnHost];
    const childObj = this.sceneObjects && this.sceneObjects[spec.targetSceneId];
    if (!hostObj || !hostObj.texturesReady || !childObj || !childObj.texturesReady) {
      return false;
    }

    this._ensurePortalShelvesInitialized(returnHost);
    const embedDiorama = spec.embedDiorama !== false;
    const inst = this._portalInstances.find(
      (p) => p.hostSceneId === returnHost && p.spec.id === spec.id
    );
    if (embedDiorama && (!inst || !inst.embedDiorama || !inst.dioramaGroup)) {
      return false;
    }

    this._prewarmSceneGpuResources(returnHost);
    this._ensurePortalOpenForEmergence(spec);
    this._updatePortalShelfTransforms();

    this._foldActiveGyroIntoPose();
    this._hoverNorm.targetX = 0;
    this._hoverNorm.targetY = 0;
    this.pointer.targetX = 0;
    this.pointer.targetY = 0;

    const flightRollDeg = typeof spec.flightRollDeg === 'number' ? spec.flightRollDeg : 0.0;
    const hostRest = this._getSceneRestPose(returnHost);

    if (!embedDiorama) {
      const fromPose = this._computeDirectPortalTargetPose(spec, returnHost);
      const childStartPose = {
        x: this.camera.position.x,
        y: this.camera.position.y,
        z: this.camera.position.z,
        yawDeg: window.__PROTOTYPE_STATE__.cameraRotationDeg.yaw,
        pitchDeg: window.__PROTOTYPE_STATE__.cameraRotationDeg.pitch,
        rollDeg: window.__PROTOTYPE_STATE__.cameraRotationDeg.roll || 0
      };
      this.currentSceneId = returnHost;
      this._innerScale = hostObj.innerScale;
      this._layer0Scale = hostObj.layer0Scale;
      window.__PROTOTYPE_STATE__.currentScene = returnHost;
      window.__PROTOTYPE_STATE__.innerLayerScale = this._innerScale;
      window.__PROTOTYPE_STATE__.layer0Scale = this._layer0Scale;
      window.__PROTOTYPE_STATE__.isTransitioning = true;

      this._applyPoseToCameraObject(fromPose);
      this._portalFlight = {
        reverse: true,
        portalSpec: spec,
        hostSceneId: returnHost,
        targetSceneId: spec.targetSceneId,
        embedDiorama: false,
        flightRollDeg,
        childStartPose,
        startTimeMs: performance.now(),
        durationMs: SCENE_GEOMETRY_SPEC.directPortalFlightDurationMs,
        fromPose,
        toPose: { ...hostRest }
      };
      if (window.__sendDiag) {
        window.__sendDiag('portal_reverse_flight_start', {
          portal: spec.id,
          from: spec.targetSceneId,
          to: returnHost,
          embedDiorama: false
        });
      }
      this.onSceneChange(returnHost);
      return true;
    }

    // Map user's current camera pose inside childScene 1:1 into hostScene's dioramaGroup coordinates
    // so frame t = 0 has zero visual discontinuity even if the visitor dollied or rotated inside childScene.
    const childRest = this._getSceneRestPose(spec.targetSceneId);
    const frontZ = Math.max(childObj.sceneFgZ, 0.0);
    const restFrontDist = Math.max(40.0, childRest.z - frontZ);
    const effFitZ = typeof spec.effectiveFitZ === 'number' ? spec.effectiveFitZ : spec.fitZ;
    const lambdaP = (effFitZ + spec.recessDepth) / restFrontDist;
    const fitCamX = spec.worldX;
    const fitCamY = spec.worldY;
    const fitCamZ = spec.worldZ + effFitZ;

    const mappedFromZ = Math.max(
      spec.worldZ + SCENE_GEOMETRY_SPEC.cameraNearPlane + 3.0,
      fitCamZ + lambdaP * (this.camera.position.z - childRest.z)
    );
    const rollRad = THREE.MathUtils.degToRad(spec.dioramaRollDeg || 0.0);
    const cosR = Math.cos(rollRad);
    const sinR = Math.sin(rollRad);
    const dxChild = this.camera.position.x - childRest.x;
    const dyChild = this.camera.position.y - childRest.y;
    let fromPose = {
      x: fitCamX + lambdaP * (dxChild * cosR - dyChild * sinR),
      y: fitCamY + lambdaP * (dxChild * sinR + dyChild * cosR),
      z: mappedFromZ,
      yawDeg: window.__PROTOTYPE_STATE__.cameraRotationDeg.yaw,
      pitchDeg: window.__PROTOTYPE_STATE__.cameraRotationDeg.pitch,
      rollDeg: flightRollDeg
    };
    let quatPath = null;
    if (spec.wallMount && inst.wallMountXform) {
      // Map the visitor's live child-scene camera through the flush wall-mount similarity so
      // t = 0 shows exactly what they were looking at, then slerp back to the host rest pose.
      const wmx = inst.wallMountXform;
      const fromPos = this.camera.position
        .clone()
        .applyQuaternion(wmx.qDio)
        .multiplyScalar(wmx.lambdaP)
        .add(wmx.tDio);
      const fromQuat = wmx.qDio.clone().multiply(this.camera.quaternion);
      fromPose = this._poseFromPositionQuat(fromPos, fromQuat);
      const toQuat = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(
          THREE.MathUtils.degToRad(hostRest.pitchDeg || 0.0),
          THREE.MathUtils.degToRad(hostRest.yawDeg || 0.0),
          THREE.MathUtils.degToRad(hostRest.rollDeg || 0.0),
          'YXZ'
        )
      );
      quatPath = { from: fromQuat, to: toQuat };
    }

    if (inst.spec.veil) {
      inst.veilCurrentOpacity = 0.0;
      inst.veilTargetOpacity = inst.spec.veil.restOpacity;
    }

    this.currentSceneId = returnHost;
    this._innerScale = hostObj.innerScale;
    this._layer0Scale = hostObj.layer0Scale;
    window.__PROTOTYPE_STATE__.currentScene = returnHost;
    window.__PROTOTYPE_STATE__.innerLayerScale = this._innerScale;
    window.__PROTOTYPE_STATE__.layer0Scale = this._layer0Scale;
    window.__PROTOTYPE_STATE__.isTransitioning = true;

    this._applyPoseToCameraObject(fromPose);
    if (quatPath) {
      this._applyFlightCameraQuat(quatPath, 0.0);
      this._setWallMountReliefScale(inst, 1.0);
      this._setWallMountOuterWallOpacity(inst, 1.0);
    }
    const initRimAndTint = this._computePortalFlightRimAndTint(spec, fromPose.z, 0.0, true);
    this._applyDioramaInstanceTint(inst, initRimAndTint.tintRgb);
    this._setHostPortalSurroundingOpacity(returnHost, initRimAndTint.rimOpacity, spec.id);
    this._updatePortalClippingPlanes(null);

    this._portalFlight = {
      reverse: true,
      portalSpec: spec,
      hostSceneId: returnHost,
      targetSceneId: spec.targetSceneId,
      embedDiorama: true,
      flightRollDeg,
      startTimeMs: performance.now(),
      durationMs: this._rigSpec.portalFlightDurationMs,
      fromPose,
      toPose: { ...hostRest },
      quatPath
    };
    if (window.__sendDiag) {
      window.__sendDiag('portal_reverse_flight_start', {
        portal: spec.id,
        from: spec.targetSceneId,
        to: returnHost
      });
    }
    this.onSceneChange(returnHost);
    return true;
  }

  getPortalReturnInfo() {
    if (!this.enteredViaPortalId) return null;
    const hostSceneId = this.enteredFromPortalSceneId || 'shelf_portal';
    if (this.currentSceneId === hostSceneId) return null;
    if (hostSceneId === 'fried_yard_installation') {
      return {
        hostSceneId,
        buttonLabelHe: '← חזרה לחצר ולמיצב',
        ariaLabelHe: 'חזרה אל חצר בית פריד והמיצב ׳מרחב לא מוגן 7׳'
      };
    }
    if (hostSceneId === 'fried_house_railing') {
      return {
        hostSceneId,
        buttonLabelHe: '← חזרה למעקה בבית פריד',
        ariaLabelHe: 'חזרה אל המעקה בבית פריד בהרצליה'
      };
    }
    if (hostSceneId === 'denya_living_room') {
      return {
        hostSceneId,
        buttonLabelHe: '← חזרה לסלון בית דניה',
        ariaLabelHe: 'חזרה אל סלון בית דניה בחיפה'
      };
    }
    if (hostSceneId === 'denya_couch_guitar') {
      return {
        hostSceneId,
        buttonLabelHe: '← חזרה לספה והגיטרה בדניה',
        ariaLabelHe: 'חזרה אל הספה והגיטרה בסלון בית דניה'
      };
    }
    if (hostSceneId === 'wandering_drawing_1') {
      return {
        hostSceneId,
        buttonLabelHe: '← חזרה החוצה ל׳שיטוט 1׳',
        ariaLabelHe: 'חזרה מן המגילה אל מרחב ׳שיטוט ברישום 1׳'
      };
    }
    if (hostSceneId === 'tzofen_brick_table' || hostSceneId === 'tzofen_brick_table_v2') {
      return {
        hostSceneId,
        buttonLabelHe: '← חזרה אל ׳צופן׳ והשולחן',
        ariaLabelHe: 'חזרה מצמרת העץ דרך החלון אל המיצב ׳צופן׳'
      };
    }
    if (hostSceneId === 'bene_beitcha_scroll_foot') {
      return {
        hostSceneId,
        buttonLabelHe: '← חזרה ל׳מגילה וכף רגל׳',
        ariaLabelHe: 'חזרה אל מיצב המגילה וכף הרגל בבנה ביתך'
      };
    }
    if (hostSceneId === 'bar_giora_bedroom') {
      return {
        hostSceneId,
        buttonLabelHe: '← חזרה לחדר בבר גיורא',
        ariaLabelHe: 'חזרה אל חדר השינה בבר גיורא בחיפה'
      };
    }
    if (hostSceneId === 'paris_cite_white_table') {
      return {
        hostSceneId,
        buttonLabelHe: '← חזרה לסטודיו בפריז',
        ariaLabelHe: 'חזרה אל השולחן הלבן בסטודיו Cité des Arts בפריז'
      };
    }
    return {
      hostSceneId,
      buttonLabelHe: '← חזרה לקיר המדפים',
      ariaLabelHe: 'חזרה אל קיר המדפים התלת-ממדי'
    };
  }

  returnToPortalHub() {
    if (this.isTransitioning || this._portalFlight) return false;
    let returnHost = this.enteredFromPortalSceneId || 'shelf_portal';
    let returnPortalId = this.enteredViaPortalId || null;
    if (this._portalOriginStack.length > 0) {
      const popped = this._portalOriginStack.pop();
      returnHost = popped.hostSceneId;
      returnPortalId = popped.portalId || returnPortalId;
      this._syncPortalOriginFromStack();
    } else {
      this.enteredViaPortalId = null;
      this.enteredFromPortalSceneId = null;
    }
    if (this.currentSceneId === returnHost) {
      this.resetCameraPose(false);
      return true;
    }
    if (returnPortalId) {
      const hostShelves = this.getPortalShelves(returnHost);
      const portalSpec = hostShelves.find((s) => s.id === returnPortalId);
      if (portalSpec && portalSpec.targetSceneId === this.currentSceneId) {
        this._ensurePortalOpenForEmergence(portalSpec);
        if (this._startReversePortalFlight(portalSpec, returnHost)) {
          return true;
        }
      }
    }
    return this.transitionTo(returnHost, { keepPortalOrigin: true });
  }

  applyDynamicWidescreenScaling() {
    const aspect = this.camera.aspect;
    const isPortrait = aspect < 1.0;

    this.camera.fov = isPortrait
      ? SCENE_GEOMETRY_SPEC.portraitFovDeg
      : SCENE_GEOMETRY_SPEC.desktopFovDeg;
    this.camera.updateProjectionMatrix();

    const distBg = this.baseCameraZ - SCENE_GEOMETRY_SPEC.bgZ;
    const halfFovRad = THREE.MathUtils.degToRad(this.camera.fov / 2);
    const visibleHeightAtBg = 2 * distBg * Math.tan(halfFovRad);
    const visibleWidthAtBg = visibleHeightAtBg * aspect;

    // Reference inner 1500x1000 scale ensures wide viewports stay generously framed, while Layer 0 scales
    // by 1.80x around the exact optical center so bore-sight alignment across all depth planes is 1:1 exact.
    const reqInnerScaleX =
      (visibleWidthAtBg * SCENE_GEOMETRY_SPEC.frustumInnerCoverageFactor) /
      SCENE_GEOMETRY_SPEC.bgBaseWidth;
    const reqInnerScaleY =
      (visibleHeightAtBg * SCENE_GEOMETRY_SPEC.frustumInnerCoverageFactor) /
      SCENE_GEOMETRY_SPEC.bgBaseHeight;
    const baseInnerScale = Math.max(SCENE_GEOMETRY_SPEC.minInnerScale, reqInnerScaleX, reqInnerScaleY);

    for (const key in this.sceneObjects) {
      const obj = this.sceneObjects[key];
      const cfg = this.sceneConfigs[key];
      if (!obj || !cfg) continue;

      const sceneBgZ = typeof cfg.bgZ === 'number' ? cfg.bgZ : SCENE_GEOMETRY_SPEC.bgZ;
      const restPose = this._getSceneRestPose(key);
      const boreY = cfg.boreSightAtRest ? restPose.y : 0.0;
      const boreZ = cfg.boreSightAtRest ? restPose.z : this.baseCameraZ;
      const depthStretch = cfg.boreSightAtRest
        ? (boreZ - sceneBgZ) / Math.max(1.0, boreZ - SCENE_GEOMETRY_SPEC.bgZ)
        : 1.0;

      const restDistBg = restPose.z - sceneBgZ;
      const restVisHeightBg = 2 * restDistBg * Math.tan(halfFovRad);
      const restVisWidthBg = restVisHeightBg * aspect;
      const expFactor =
        typeof cfg.bgExpansionFactor === 'number'
          ? cfg.bgExpansionFactor
          : SCENE_GEOMETRY_SPEC.bgExpansionFactor;
      const minScaleForRestW =
        restVisWidthBg / (SCENE_GEOMETRY_SPEC.bgBaseWidth * expFactor * 0.91);
      const minScaleForRestH =
        (restVisHeightBg + 2 * Math.abs(restPose.y)) /
        (SCENE_GEOMETRY_SPEC.bgBaseHeight * expFactor * 0.91);

      const sceneMinInner =
        typeof cfg.minInnerScale === 'number'
          ? Math.max(baseInnerScale, cfg.minInnerScale) * depthStretch
          : baseInnerScale * depthStretch;
      const sceneInnerScale = Math.max(
        sceneMinInner,
        minScaleForRestW,
        minScaleForRestH
      );
      const sceneLayer0Scale = sceneInnerScale * expFactor;
      obj.innerScale = sceneInnerScale;
      obj.layer0Scale = sceneLayer0Scale;

      const boreDistBg = boreZ - sceneBgZ;
      const innerWorldWidthAtBg = SCENE_GEOMETRY_SPEC.bgBaseWidth * sceneInnerScale;

      if (obj.layer0) {
        obj.layer0.position.set(0, 0, sceneBgZ);
        obj.layer0.scale.set(sceneLayer0Scale, sceneLayer0Scale, 1.0);
        if (obj.layer0.material && obj.layer0.material.uniforms) {
          obj.layer0.material.uniforms.uDistBg.value = boreDistBg;
          obj.layer0.material.uniforms.uOpticalCenterLocalY.value = boreY / sceneLayer0Scale;
        }
      }

      if (obj.layer1) {
        const midRatio = (boreZ - obj.sceneMidZ) / boreDistBg;
        const midScale = (innerWorldWidthAtBg * midRatio) / SCENE_GEOMETRY_SPEC.midBaseWidth;
        const midCenterY = boreY * (1.0 - midRatio);
        obj.midBasePosX = 0;
        obj.midBasePosY = midCenterY;
        obj.midPlaneWorldW = SCENE_GEOMETRY_SPEC.midBaseWidth * midScale;
        obj.midPlaneWorldH = SCENE_GEOMETRY_SPEC.midBaseHeight * midScale;
        obj.layer1.scale.set(midScale, midScale, 1.0);
        obj.layer1.position.set(0, midCenterY, obj.sceneMidZ);
        if (obj.layer1.material && obj.layer1.material.uniforms) {
          obj.layer1.material.uniforms.uDistBg.value = Math.max(1.0, boreZ - obj.sceneMidZ);
          obj.layer1.material.uniforms.uOpticalCenterLocalY.value =
            (boreY * midRatio) / Math.max(1e-5, midScale);
        }
      }

      if (obj.layer2) {
        const fgRatio = (boreZ - obj.sceneFgZ) / boreDistBg;
        const fgExpMult = cfg.fgExpanded ? SCENE_GEOMETRY_SPEC.bgExpansionFactor : 1.0;
        const fgScale =
          ((innerWorldWidthAtBg * fgRatio) / SCENE_GEOMETRY_SPEC.fgBaseWidth) * fgExpMult;
        const fgCenterY = boreY * (1.0 - fgRatio);
        obj.fgBasePosX = 0;
        obj.fgBasePosY = fgCenterY;
        obj.fgPlaneWorldW = SCENE_GEOMETRY_SPEC.fgBaseWidth * fgScale;
        obj.fgPlaneWorldH = SCENE_GEOMETRY_SPEC.fgBaseHeight * fgScale;
        obj.layer2.scale.set(fgScale, fgScale, 1.0);
        obj.layer2.position.set(0, fgCenterY, obj.sceneFgZ);
        if (obj.layer2.material && obj.layer2.material.uniforms) {
          obj.layer2.material.uniforms.uDistBg.value = Math.max(1.0, boreZ - obj.sceneFgZ);
          obj.layer2.material.uniforms.uOpticalCenterLocalY.value =
            (boreY * fgRatio) / Math.max(1e-5, fgScale);
        }
        if (cfg.fgBleedEdges) {
          const fgHalfW =
            0.5 * SCENE_GEOMETRY_SPEC.fgBaseWidth * fgScale * SCENE_GEOMETRY_SPEC.frustumEdgeSafetyRatio;
          const fgHalfH =
            0.5 * SCENE_GEOMETRY_SPEC.fgBaseHeight * fgScale * SCENE_GEOMETRY_SPEC.frustumEdgeSafetyRatio;
          obj.fgBleedBounds = {
            z: obj.sceneFgZ,
            guardMinZ:
              typeof cfg.fgBleedEdges.guardMinZ === 'number' ? cfg.fgBleedEdges.guardMinZ : 180,
            minX: cfg.fgBleedEdges.left ? -fgHalfW : null,
            maxX: cfg.fgBleedEdges.right ? fgHalfW : null,
            minY: cfg.fgBleedEdges.bottom ? fgCenterY - fgHalfH : null,
            maxY: cfg.fgBleedEdges.top ? fgCenterY + fgHalfH : null
          };
        } else {
          obj.fgBleedBounds = null;
        }
      }

      if (obj.reflectionMesh) {
        const refRatio = (boreZ - SCENE_GEOMETRY_SPEC.reflectionZ) / boreDistBg;
        const refScale = (innerWorldWidthAtBg * refRatio) / SCENE_GEOMETRY_SPEC.fgBaseWidth;
        obj.reflectionMesh.scale.set(refScale, refScale, 1.0);
        obj.reflectionMesh.position.set(0, boreY * (1.0 - refRatio), SCENE_GEOMETRY_SPEC.reflectionZ);
      }

      if (Array.isArray(obj.extraLayers)) {
        for (const extra of obj.extraLayers) {
          const exRatio = (boreZ - extra.zPos) / boreDistBg;
          const exExpMult = extra.expanded ? SCENE_GEOMETRY_SPEC.bgExpansionFactor : 1.0;
          const exScale =
            ((innerWorldWidthAtBg * exRatio) / SCENE_GEOMETRY_SPEC.fgBaseWidth) * exExpMult;
          const exCenterY = boreY * (1.0 - exRatio);
          extra.planeWorldW = SCENE_GEOMETRY_SPEC.fgBaseWidth * exScale;
          extra.planeWorldH = SCENE_GEOMETRY_SPEC.fgBaseHeight * exScale;
          extra.basePosX = extra.mirrorLeftEdge ? -extra.planeWorldW : 0;
          extra.basePosY = exCenterY;
          extra.basePosZ = extra.zPos;
          extra.baseScale = exScale;
          extra.mesh.scale.set(extra.mirrorLeftEdge ? -exScale : exScale, exScale, 1.0);
          extra.mesh.position.set(extra.basePosX, exCenterY, extra.zPos);
          extra.mesh.rotation.set(0, 0, 0);
          if (extra.mesh.material && extra.mesh.material.uniforms) {
            extra.mesh.material.uniforms.uDistBg.value = Math.max(1.0, boreZ - extra.zPos);
            extra.mesh.material.uniforms.uOpticalCenterLocalX.value = extra.mirrorLeftEdge
              ? -SCENE_GEOMETRY_SPEC.fgBaseWidth
              : 0.0;
            extra.mesh.material.uniforms.uOpticalCenterLocalY.value =
              (boreY * exRatio) / Math.max(1e-5, exScale);
          }
        }
      }
      if (cfg.carpetUnrollRig) {
        this._applyCarpetUnrollTransform(key, true);
      }
    }

    this._updatePortalShelfTransforms();

    const activeObj = this.sceneObjects[this.currentSceneId];
    this._innerScale = activeObj ? activeObj.innerScale : baseInnerScale;
    this._layer0Scale =
      activeObj ? activeObj.layer0Scale : baseInnerScale * SCENE_GEOMETRY_SPEC.bgExpansionFactor;

    // Re-clamp active camera pose to updated viewport frustum
    const reclamped = this._clampPose(this.poseTarget);
    this.poseTarget.x = reclamped.x;
    this.poseTarget.y = reclamped.y;
    this.poseTarget.z = reclamped.z;
    this.poseTarget.yawDeg = reclamped.yawDeg;
    this.poseTarget.pitchDeg = reclamped.pitchDeg;

    window.__PROTOTYPE_STATE__.innerLayerScale = this._innerScale;
    window.__PROTOTYPE_STATE__.layer0Scale = this._layer0Scale;
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
    const dpr = this.renderer ? this.renderer.getPixelRatio() : 1;
    const rtW = Math.max(1, Math.round(this.width * dpr));
    const rtH = Math.max(1, Math.round(this.height * dpr));
    this.rtCurrent = new THREE.WebGLRenderTarget(rtW, rtH);
    this.rtNext = new THREE.WebGLRenderTarget(rtW, rtH);

    this.transitionCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.transitionScene = new THREE.Scene();

    this.transitionMaterial = new THREE.ShaderMaterial({
      vertexShader: TRANSITION_VERTEX_SHADER,
      fragmentShader: TRANSITION_FRAGMENT_SHADER,
      uniforms: {
        uTexCurrent: { value: null },
        uTexNext: { value: null },
        uProgress: { value: 0.0 }
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
    window.addEventListener('orientationchange', this._onScreenOrientationChange);
    if (
      typeof window !== 'undefined' &&
      window.screen &&
      window.screen.orientation &&
      typeof window.screen.orientation.addEventListener === 'function'
    ) {
      window.screen.orientation.addEventListener('change', this._onScreenOrientationChange);
    }
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this._onVisibilityChange);
    }
    window.addEventListener('resize', this._onResize);
    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
    window.addEventListener('blur', this._onWindowBlur);
  }

  switchScene(nextSceneId) {
    return this.transitionTo(nextSceneId);
  }

  transitionToScene(nextSceneId, options = {}) {
    return this.transitionTo(nextSceneId, options);
  }

  _beginShaderTransition(nextSceneId) {
    if (this.currentSceneId === nextSceneId || this.isTransitioning) return;

    this._transitionFromPose = {
      x: this.camera.position.x,
      y: this.camera.position.y,
      z: this.camera.position.z,
      yawDeg: window.__PROTOTYPE_STATE__.cameraRotationDeg.yaw,
      pitchDeg: window.__PROTOTYPE_STATE__.cameraRotationDeg.pitch
    };
    const nextRest = this._getSceneRestPose(nextSceneId);
    this._transitionToPose = { ...nextRest };

    const nextObj = this.sceneObjects[nextSceneId];
    if (nextObj && nextObj.texturesReady) {
      this._prewarmSceneGpuResources(nextSceneId);
    }

    if (this.hasPortalShelves(nextSceneId)) {
      this._ensurePortalShelvesInitialized(nextSceneId);
      this._setHostPortalSurroundingOpacity(nextSceneId, 1.0, null);
      this._updatePortalShelfTransforms();
      this._warmPortalTargetScenes(nextSceneId);
    }

    this.isTransitioning = true;
    this.targetSceneId = nextSceneId;
    this.transitionStartTime = performance.now();
    this.transitionProgress = 0.0;
    this.setCameraPose(nextRest, false);

    if (window.__sendDiag) {
      window.__sendDiag('scene_transition', { from: this.currentSceneId, to: nextSceneId });
    }

    window.__PROTOTYPE_STATE__.isTransitioning = true;
  }

  transitionTo(nextSceneId, options = {}) {
    if (!nextSceneId) return false;
    if (this.currentSceneId === nextSceneId && !this._pendingTransitionSceneId) return false;
    if (!this.sceneConfigs[nextSceneId]) return false;

    if (this.currentSceneId === nextSceneId && this._pendingTransitionSceneId) {
      this._pendingTransitionSceneId = null;
      window.__PROTOTYPE_STATE__.isTransitioning = false;
      this.onSceneChange(this.currentSceneId);
      return true;
    }
    if (
      this.isTransitioning ||
      this._portalFlight ||
      this._pendingPortalFlightId ||
      this._pendingTransitionSceneId
    ) {
      return false;
    }

    if (this.hoveredPortalId !== null) {
      this.hoveredPortalId = null;
      this.onPortalHover(null, 0, 0);
    }

    const handleAsyncBundleFailure = () => {
      if (this._pendingTransitionSceneId === nextSceneId) {
        this._pendingTransitionSceneId = null;
        window.__PROTOTYPE_STATE__.isTransitioning = false;
        this.onSceneChange(this.currentSceneId);
        const hint = document.getElementById('center-hint');
        if (hint) {
          hint.textContent = 'המרחב עדיין בבנייה ברקע — נסו שוב בעוד רגע';
          hint.classList.remove('hidden');
        }
      }
    };

    const originSceneId = this.currentSceneId;
    const allowPortalFlight = options.forceCrossDissolve !== true;
    const fwdSpec = allowPortalFlight
      ? this.getPortalShelves(originSceneId).find(
          (s) => s && s.portalMode !== 'reverse_transition' && s.targetSceneId === nextSceneId
        )
      : null;

    if (fwdSpec) {
      if (this.isPortalLocked(fwdSpec)) {
        this.erasePortalOccluder(fwdSpec);
      }
      this.onSceneChange(nextSceneId);
      if (this._isSceneBundleReady(nextSceneId)) {
        this._updatePortalShelfTransforms();
        return this._startPortalFlight(fwdSpec);
      }
      this._pendingTransitionSceneId = nextSceneId;
      window.__PROTOTYPE_STATE__.isTransitioning = true;
      this._loadSceneBundleReady(nextSceneId, false)
        .then(() => {
          if (
            this._pendingTransitionSceneId === nextSceneId &&
            this.currentSceneId === originSceneId &&
            !this.isTransitioning &&
            !this._portalFlight
          ) {
            this._pendingTransitionSceneId = null;
            this._updatePortalShelfTransforms();
            this._startPortalFlight(fwdSpec);
          }
        })
        .catch(handleAsyncBundleFailure);
      return true;
    }

    const revSpec = allowPortalFlight
      ? this.getPortalShelves(nextSceneId).find(
          (s) => s && s.portalMode !== 'reverse_transition' && s.targetSceneId === originSceneId
        )
      : null;

    if (revSpec) {
      const ancIdx = this._portalOriginStack.findIndex((e) => e.hostSceneId === nextSceneId);
      if (ancIdx >= 0) {
        this._portalOriginStack.length = ancIdx;
      } else if (!options.keepPortalOrigin && !options.preservePortalStack) {
        this._portalOriginStack.length = 0;
      }
      this._syncPortalOriginFromStack();
      this._ensurePortalOpenForEmergence(revSpec);

      const hostObj = this.sceneObjects[nextSceneId];
      const originObj = this.sceneObjects[originSceneId];
      if (hostObj && hostObj.texturesReady && originObj && originObj.texturesReady) {
        this._warmPortalTargetScenes(nextSceneId);
        if (this._startReversePortalFlight(revSpec, nextSceneId)) {
          return true;
        }
      }
      this.onSceneChange(nextSceneId);
      this._pendingTransitionSceneId = nextSceneId;
      window.__PROTOTYPE_STATE__.isTransitioning = true;
      Promise.all([
        this._loadSceneBundleReady(nextSceneId, false),
        this._loadSceneTextures(originSceneId, false)
      ])
        .then(() => {
          if (
            this._pendingTransitionSceneId === nextSceneId &&
            this.currentSceneId === originSceneId &&
            !this.isTransitioning &&
            !this._portalFlight
          ) {
            this._pendingTransitionSceneId = null;
            this._ensurePortalOpenForEmergence(revSpec);
            if (!this._startReversePortalFlight(revSpec, nextSceneId)) {
              this._beginShaderTransition(nextSceneId);
            }
          }
        })
        .catch(handleAsyncBundleFailure);
      return true;
    }

    if (!options.keepPortalOrigin && !options.preservePortalStack) {
      this._portalOriginStack.length = 0;
      this._syncPortalOriginFromStack();
    }
    this.onSceneChange(nextSceneId);

    // Full Bundle Preloading (#27): only begin revealing nextSceneId once ALL of its
    // own layers AND every embedded portal diorama inside it are decoded and GPU-prewarmed.
    if (this._isSceneBundleReady(nextSceneId)) {
      this._pendingTransitionSceneId = null;
      this._beginShaderTransition(nextSceneId);
      return true;
    }

    this._pendingTransitionSceneId = nextSceneId;
    window.__PROTOTYPE_STATE__.isTransitioning = true;
    this._loadSceneBundleReady(nextSceneId, false)
      .then(() => {
        if (this._pendingTransitionSceneId === nextSceneId && !this.isTransitioning) {
          this._pendingTransitionSceneId = null;
          this._beginShaderTransition(nextSceneId);
        }
      })
      .catch(handleAsyncBundleFailure);
    return true;
  }

  disableGyroscope(callback) {
    if (this._gyroNoEventsTimer !== null) {
      clearTimeout(this._gyroNoEventsTimer);
      this._gyroNoEventsTimer = null;
    }
    this._foldActiveGyroIntoPose();
    this.gyroActive = false;
    window.__PROTOTYPE_STATE__.gyroActive = false;
    const gyroBtn = document.getElementById('gyro-toggle');
    if (gyroBtn) {
      gyroBtn.classList.remove('active');
      gyroBtn.textContent = '⟲ מצב הטיה: עכבר / ג׳יירו';
    }
    if (window.__sendDiag) {
      window.__sendDiag('gyro_toggled', { active: false, permission: this.gyroPermissionState });
    }
    if (typeof callback === 'function') callback(false);
    return Promise.resolve(false);
  }

  enableGyroscope(callback) {
    if (this._gyroPermissionPending) {
      return this._gyroPermissionPending;
    }

    const syncProtoGyroState = () => {
      window.__PROTOTYPE_STATE__.gyroActive = this.gyroActive;
      window.__PROTOTYPE_STATE__.gyroPermissionState = this.gyroPermissionState;
    };

    const activateAndMonitor = (active) => {
      if (this._gyroNoEventsTimer !== null) {
        clearTimeout(this._gyroNoEventsTimer);
        this._gyroNoEventsTimer = null;
      }
      this._gyroReferenceQuat = null;
      this._gyroLastValidAlphaDeg = null;
      this._gyroTargetYawDeg = 0;
      this._gyroTargetPitchDeg = 0;
      this._gyroYawDeg = 0;
      this._gyroPitchDeg = 0;
      this._lastScreenAngleDeg = getScreenOrientationAngleDeg();
      syncProtoGyroState();
      if (active) {
        const beforeCount = this._gyroEventsCount;
        const timeoutMs = this._rigSpec.gyroNoEventsTimeoutMs;
        this._gyroNoEventsTimer = setTimeout(() => {
          this._gyroNoEventsTimer = null;
          if (this.gyroActive && this._gyroEventsCount === beforeCount) {
            if (window.__sendDiag) {
              window.__sendDiag('gyro_no_events', {
                msg: `No deviceorientation events received within ${timeoutMs}ms`
              });
            }
          }
        }, timeoutMs);
      }
      if (window.__sendDiag) {
        window.__sendDiag('gyro_toggled', {
          active: active,
          permission: this.gyroPermissionState
        });
      }
    };

    const updateUI = (active) => {
      const gyroBtn = document.getElementById('gyro-toggle');
      if (gyroBtn) {
        if (active) {
          gyroBtn.classList.add('active');
          gyroBtn.textContent = '✓ ג׳יירו פעיל';
        } else if (this.gyroPermissionState === 'denied') {
          gyroBtn.classList.remove('active');
          gyroBtn.textContent = '⚠ הרשאת ג׳יירו נדחתה';
        } else if (this.gyroPermissionState === 'unsupported') {
          gyroBtn.classList.remove('active');
          gyroBtn.textContent = '⚠ חיישן הטיה לא זמין';
        } else {
          gyroBtn.classList.remove('active');
          gyroBtn.textContent = '⟲ מצב הטיה: עכבר / ג׳יירו';
        }
      }
      activateAndMonitor(active);
      if (typeof callback === 'function') callback(active);
    };

    if (this.gyroActive) {
      return this.disableGyroscope(callback);
    }

    if (typeof DeviceOrientationEvent === 'undefined') {
      this.gyroPermissionState = 'unsupported';
      this.gyroActive = false;
      updateUI(false);
      return Promise.resolve(false);
    }

    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
      this._gyroPermissionPending = DeviceOrientationEvent.requestPermission()
        .then((permissionState) => {
          this._gyroPermissionPending = null;
          const granted = permissionState === 'granted';
          this.gyroPermissionState = granted ? 'granted' : 'denied';
          this.gyroActive = granted;
          updateUI(granted);
          return granted;
        })
        .catch((err) => {
          this._gyroPermissionPending = null;
          console.error('DeviceOrientation permission error:', err);
          this.gyroPermissionState = 'denied';
          this.gyroActive = false;
          updateUI(false);
          return false;
        });
      return this._gyroPermissionPending;
    }

    this.gyroPermissionState = 'granted';
    this.gyroActive = true;
    updateUI(true);
    return Promise.resolve(true);
  }

  _stepKeyboardNavigation(dtSec) {
    if (this._keysHeld.size === 0) return;

    const activeCfg = this.sceneConfigs[this.targetSceneId || this.currentSceneId] || null;
    const speedMult =
      activeCfg && typeof activeCfg.dollySpeedMult === 'number' ? activeCfg.dollySpeedMult : 1.0;
    const moveDist = this._rigSpec.keyMoveSpeedPerSec * speedMult * dtSec;
    const elevDist = this._rigSpec.keyElevationSpeedPerSec * speedMult * dtSec;
    let forward = 0;
    let right = 0;
    let up = 0;

    if (KEYBOARD_NAV_BINDINGS.forward.some((c) => this._keysHeld.has(c))) forward += moveDist;
    if (KEYBOARD_NAV_BINDINGS.backward.some((c) => this._keysHeld.has(c))) forward -= moveDist;
    if (KEYBOARD_NAV_BINDINGS.right.some((c) => this._keysHeld.has(c))) right += moveDist;
    if (KEYBOARD_NAV_BINDINGS.left.some((c) => this._keysHeld.has(c))) right -= moveDist;
    if (KEYBOARD_NAV_BINDINGS.up.some((c) => this._keysHeld.has(c))) up += elevDist;
    if (KEYBOARD_NAV_BINDINGS.down.some((c) => this._keysHeld.has(c))) up -= elevDist;

    if (forward === 0 && right === 0 && up === 0) return;

    const yawRad = THREE.MathUtils.degToRad(this.poseTarget.yawDeg + this._gyroYawDeg);
    const dollyVec = this._computeTunnelBoreDollyDelta(forward, yawRad);
    const prevX = this.poseTarget.x;
    const prevY = this.poseTarget.y;
    const prevZ = this.poseTarget.z;
    const nextClamped = this._clampPose({
      x: this.poseTarget.x + dollyVec.dx + Math.cos(yawRad) * right,
      y: this.poseTarget.y + dollyVec.dy + up,
      z: this.poseTarget.z + dollyVec.dz - Math.sin(yawRad) * right,
      yawDeg: this.poseTarget.yawDeg,
      pitchDeg: this.poseTarget.pitchDeg,
      rollDeg: this.poseTarget.rollDeg || 0
    });
    const actualMove = Math.hypot(
      nextClamped.x - prevX,
      nextClamped.y - prevY,
      nextClamped.z - prevZ
    );
    this.poseTarget.x = nextClamped.x;
    this.poseTarget.y = nextClamped.y;
    this.poseTarget.z = nextClamped.z;

    if (window.__sendDiag) {
      const now = Date.now();
      if (actualMove < 0.05) {
        if (now - this._lastDiagKbClampTs > 1500) {
          this._lastDiagKbClampTs = now;
          window.__sendDiag('keyboard_clamp_saturated', {
            keys: Array.from(this._keysHeld),
            scene: this.currentSceneId,
            pose: {
              x: Math.round(this.poseTarget.x),
              y: Math.round(this.poseTarget.y),
              z: Math.round(this.poseTarget.z),
              yaw: Number(this.poseTarget.yawDeg.toFixed(1)),
              pitch: Number(this.poseTarget.pitchDeg.toFixed(1))
            },
            desired: {
              forward: Number(forward.toFixed(2)),
              right: Number(right.toFixed(2)),
              up: Number(up.toFixed(2))
            }
          });
        }
      } else if (now - this._lastDiagKbMoveTs > 1500) {
        this._lastDiagKbMoveTs = now;
        window.__sendDiag('keyboard_nav_move', {
          keys: Array.from(this._keysHeld),
          scene: this.currentSceneId,
          pose: {
            x: Math.round(this.poseTarget.x),
            y: Math.round(this.poseTarget.y),
            z: Math.round(this.poseTarget.z)
          }
        });
      }
    }
  }

  _poseFromPositionQuat(pos, quat) {
    const e = new THREE.Euler().setFromQuaternion(quat, 'YXZ');
    return {
      x: pos.x,
      y: pos.y,
      z: pos.z,
      yawDeg: THREE.MathUtils.radToDeg(e.y),
      pitchDeg: THREE.MathUtils.radToDeg(e.x),
      rollDeg: THREE.MathUtils.radToDeg(e.z)
    };
  }

  _applyFlightCameraQuat(quatPath, ease) {
    this.camera.quaternion.slerpQuaternions(quatPath.from, quatPath.to, ease);
    this.camera.updateMatrixWorld();
    const rot = window.__PROTOTYPE_STATE__.cameraRotationDeg;
    rot.yaw = THREE.MathUtils.radToDeg(this.camera.rotation.y);
    rot.pitch = THREE.MathUtils.radToDeg(this.camera.rotation.x);
    rot.roll = THREE.MathUtils.radToDeg(this.camera.rotation.z);
  }

  _applyPoseToCameraObject(pose) {
    const rollDeg = typeof pose.rollDeg === 'number' ? pose.rollDeg : 0;
    this.camera.position.set(pose.x, pose.y, pose.z);
    this.camera.rotation.set(
      THREE.MathUtils.degToRad(pose.pitchDeg),
      THREE.MathUtils.degToRad(pose.yawDeg),
      THREE.MathUtils.degToRad(rollDeg)
    );
    this.camera.updateMatrixWorld();

    window.__PROTOTYPE_STATE__.cameraPosition.x = this.camera.position.x;
    window.__PROTOTYPE_STATE__.cameraPosition.y = this.camera.position.y;
    window.__PROTOTYPE_STATE__.cameraPosition.z = this.camera.position.z;
    window.__PROTOTYPE_STATE__.cameraRotationDeg.yaw = pose.yawDeg;
    window.__PROTOTYPE_STATE__.cameraRotationDeg.pitch = pose.pitchDeg;
    window.__PROTOTYPE_STATE__.cameraRotationDeg.roll = rollDeg;
  }

  _syncCameraTransform() {
    // Combine damped 6-DOF pose + quaternion gyro + subtle hover + programmatic pointer excursion,
    // then run the analytical ray-frustum guard so the final rendered camera pose NEVER exposes borders.
    const gyroYaw = this.gyroActive ? this._gyroYawDeg : 0;
    const gyroPitch = this.gyroActive ? this._gyroPitchDeg : 0;
    const gyroNormX = gyroYaw / Math.max(1e-4, this._rigSpec.gyroMaxYawDeg);
    const gyroNormY = gyroPitch / Math.max(1e-4, this._rigSpec.gyroMaxPitchDeg);

    const combinedPose = this._clampPose({
      x:
        this.poseCurrent.x -
        gyroNormX * this._rigSpec.gyroParallaxShiftX +
        this._hoverNorm.x * SCENE_GEOMETRY_SPEC.hoverAmbientShiftX +
        this.pointer.x * SCENE_GEOMETRY_SPEC.pointerExcursionShiftX,
      y:
        this.poseCurrent.y +
        gyroNormY * this._rigSpec.gyroParallaxShiftY +
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
        this.pointer.y * SCENE_GEOMETRY_SPEC.pointerExcursionPitchDeg,
      rollDeg: this.poseCurrent.rollDeg || 0
    });

    this._applyPoseToCameraObject(combinedPose);
  }

  _updateScenePlaneVisibilitiesAndOpacities(sceneKey) {
    const obj = this.sceneObjects ? this.sceneObjects[sceneKey] : null;
    const cfg = this.sceneConfigs ? this.sceneConfigs[sceneKey] : null;
    if (!obj || !this.camera) return;

    const camZ = this.camera.position.z;
    const viewExcursion = Math.min(
      1.0,
      Math.abs(window.__PROTOTYPE_STATE__.cameraRotationDeg.yaw) / 20.0 +
        Math.abs(this.camera.position.x) / 140.0
    );
    const fadeStart = SCENE_GEOMETRY_SPEC.nearPassFadeStartDist;
    const fadeFull = SCENE_GEOMETRY_SPEC.nearPassFadeFullDist;
    const setMeshOpacity = (mesh, op, alphaCutout = 0.0) => {
      if (!mesh || !mesh.material) return;
      if (mesh.material.uniforms && mesh.material.uniforms.uOpacity) {
        mesh.material.transparent = alphaCutout > 0.0 ? op < 0.999 : true;
        mesh.material.uniforms.uOpacity.value = op;
      } else {
        mesh.material.opacity = op;
      }
    };
    const applyPlaneFade = (mesh, planeZ, baseOpacity, alphaCutout = 0.0) => {
      if (!mesh || !mesh.material) return;
      const dist = camZ - planeZ;
      if (dist <= fadeStart) {
        mesh.visible = false;
        setMeshOpacity(mesh, 0.0, alphaCutout);
        return;
      }
      mesh.visible = baseOpacity > 0.005;
      const t = THREE.MathUtils.clamp((dist - fadeStart) / Math.max(1e-5, fadeFull - fadeStart), 0.0, 1.0);
      const nearFade = t * t * (3.0 - 2.0 * t);
      setMeshOpacity(mesh, baseOpacity * nearFade, alphaCutout);
    };

    const surroundingOp =
      typeof obj.portalSurroundingOpacity === 'number' ? obj.portalSurroundingOpacity : 1.0;
    applyPlaneFade(obj.layer2, obj.sceneFgZ, surroundingOp, obj.fgAlphaCutout || 0.0);
    applyPlaneFade(obj.layer1, obj.sceneMidZ, surroundingOp, obj.midAlphaCutout || 0.0);

    if (obj.reflectionMesh) {
      const baseOp =
        cfg && typeof cfg.reflectionBaseOpacity === 'number' ? cfg.reflectionBaseOpacity : 0.35;
      const boostOp =
        cfg && typeof cfg.reflectionExcursionBoost === 'number'
          ? cfg.reflectionExcursionBoost
          : 0.45;
      applyPlaneFade(
        obj.reflectionMesh,
        SCENE_GEOMETRY_SPEC.reflectionZ,
        Math.min(1.0, baseOp + boostOp * viewExcursion)
      );
    }

    if (Array.isArray(obj.extraLayers)) {
      for (const extra of obj.extraLayers) {
        const exBase = typeof extra.baseOpacity === 'number' ? extra.baseOpacity : 1.0;
        const exBoost = typeof extra.excursionBoost === 'number' ? extra.excursionBoost : 0.0;
        const targetOp = Math.min(1.0, exBase + exBoost * viewExcursion) * surroundingOp;
        if (extra.depthTex && extra.mesh.material && extra.mesh.material.uniforms && extra.alphaCutout > 0.0) {
          extra.mesh.visible = targetOp > 0.005;
          extra.mesh.material.transparent = targetOp < 0.999;
          extra.mesh.material.uniforms.uOpacity.value = targetOp;
          continue;
        }
        applyPlaneFade(
          extra.mesh,
          extra.zPos,
          targetOp,
          extra.alphaCutout || 0.0
        );
      }
    }
  }

  animate() {
    this.animId = requestAnimationFrame(() => this.animate());

    if (!this.layersReady) {
      return;
    }

    const flightTargetClipping =
      this._portalFlight && this._sceneNeedsLocalClipping(this._portalFlight.targetSceneId);
    const needsLocalClipping =
      this._sceneNeedsLocalClipping(this.currentSceneId) ||
      this._sceneNeedsLocalClipping(this.targetSceneId) ||
      Boolean(flightTargetClipping);
    if (this.renderer.localClippingEnabled !== needsLocalClipping) {
      this.renderer.localClippingEnabled = needsLocalClipping;
    }

    const nowMs = performance.now();
    const rawDtSec = Math.max(0.001, (nowMs - this._lastFrameTimeMs) * 0.001);
    const dtSec = Math.min(0.05, rawDtSec);
    const dampingFrameScale =
      Math.min(SCENE_GEOMETRY_SPEC.dampingMaxDtSec, rawDtSec) *
      SCENE_GEOMETRY_SPEC.dampingReferenceHz;
    this._lastFrameTimeMs = nowMs;

    this._stepPortalOccluderAnimations(nowMs);
    this._stepCarpetUnrollAnimations(nowMs);

    if (this.gyroActive) {
      const gyroAlpha = 1.0 - Math.exp(-dtSec * this._rigSpec.gyroSmoothingRate);
      this._gyroYawDeg += (this._gyroTargetYawDeg - this._gyroYawDeg) * gyroAlpha;
      this._gyroPitchDeg += (this._gyroTargetPitchDeg - this._gyroPitchDeg) * gyroAlpha;
    } else {
      this._gyroYawDeg = 0;
      this._gyroPitchDeg = 0;
    }

    if (this._portalFlight) {
      const flight = this._portalFlight;
      const rawT =
        typeof flight.frozenProgress === 'number'
          ? flight.frozenProgress
          : (nowMs - flight.startTimeMs) / flight.durationMs;
      const t = Math.min(1.0, Math.max(0.0, rawT));
      const ease = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);

      const flightPose = {
        x: THREE.MathUtils.lerp(flight.fromPose.x, flight.toPose.x, ease),
        y: THREE.MathUtils.lerp(flight.fromPose.y, flight.toPose.y, ease),
        z: THREE.MathUtils.lerp(flight.fromPose.z, flight.toPose.z, ease),
        yawDeg: THREE.MathUtils.lerp(flight.fromPose.yawDeg, flight.toPose.yawDeg, ease),
        pitchDeg: THREE.MathUtils.lerp(flight.fromPose.pitchDeg, flight.toPose.pitchDeg, ease),
        rollDeg: THREE.MathUtils.lerp(flight.fromPose.rollDeg || 0, flight.toPose.rollDeg || 0, ease)
      };
      this._applyPoseToCameraObject(flightPose);
      if (flight.quatPath) {
        this._applyFlightCameraQuat(flight.quatPath, ease);
      }

      const targetObj = this.sceneObjects[flight.targetSceneId];
      const flightInst = this._portalInstances.find((p) => p.spec.id === flight.portalSpec.id);
      if (!this.isPaused && targetObj && targetObj.layer3) {
        targetObj.layer3.rotation.y += 0.0004;
        targetObj.layer3.rotation.x += 0.0002;
        if (flightInst && flightInst.dustMesh) {
          flightInst.dustMesh.rotation.copy(targetObj.layer3.rotation);
        }
      }

      const completePortalFlight = () => {
        if (flight.embedDiorama) {
          if (flightInst) {
            this._applyDioramaInstanceTint(flightInst, flightInst.spec.dioramaRestTintRgb);
            this._setWallMountReliefScale(flightInst, 0.0);
            this._setWallMountOuterWallOpacity(flightInst, 0.0);
          }
          if (flightInst && flightInst.spec.veil) {
            flightInst.veilCurrentOpacity = flightInst.spec.veil.restOpacity;
            flightInst.veilTargetOpacity = flightInst.spec.veil.restOpacity;
          }
          this._setHostPortalSurroundingOpacity(flight.hostSceneId, 1.0, null);
        }
        const landedSceneId = flight.reverse ? flight.hostSceneId : flight.targetSceneId;
        if (!flight.reverse) {
          this._recordPortalHop(flight.hostSceneId, flight.portalSpec.id, flight.targetSceneId);
        }
        this.currentSceneId = landedSceneId;
        this._portalFlight = null;
        if (this._pendingPortalShelfRebuild) {
          const rebuildHost =
            typeof this._pendingPortalShelfRebuild === 'string'
              ? this._pendingPortalShelfRebuild
              : null;
          this._pendingPortalShelfRebuild = false;
          this._initPortalShelves(rebuildHost);
          this._updatePortalShelfTransforms();
        }
        if (this.hasPortalShelves(this.currentSceneId)) {
          this._ensurePortalShelvesInitialized(this.currentSceneId);
          this._setHostPortalSurroundingOpacity(this.currentSceneId, 1.0, null);
          this._updatePortalShelfTransforms();
          this._warmPortalTargetScenes(this.currentSceneId);
        }
        const landedObj = this.sceneObjects[this.currentSceneId];
        if (landedObj) {
          this._innerScale = landedObj.innerScale;
          this._layer0Scale = landedObj.layer0Scale;
          window.__PROTOTYPE_STATE__.innerLayerScale = this._innerScale;
          window.__PROTOTYPE_STATE__.layer0Scale = this._layer0Scale;
        }
        this._gyroReferenceQuat = null;
        this._gyroTargetYawDeg = 0;
        this._gyroTargetPitchDeg = 0;
        this._gyroYawDeg = 0;
        this._gyroPitchDeg = 0;
        this.pointer.targetX = 0;
        this.pointer.targetY = 0;
        this.pointer.x = 0;
        this.pointer.y = 0;
        this._hoverNorm.targetX = 0;
        this._hoverNorm.targetY = 0;
        this._hoverNorm.x = 0;
        this._hoverNorm.y = 0;
        this.setCameraPose(this._getSceneRestPose(this.currentSceneId), true);
        this._updateScenePlaneVisibilitiesAndOpacities(this.currentSceneId);
        if (this._sceneNeedsLocalClipping(this.currentSceneId)) {
          this._updatePortalClippingPlanes();
        }
        window.__PROTOTYPE_STATE__.currentScene = this.currentSceneId;
        window.__PROTOTYPE_STATE__.isTransitioning = false;
        if (!flight.reverse) {
          this.onSceneChange(this.currentSceneId);
        }
      };

      if (flight.embedDiorama) {
        const { rimOpacity, tintRgb } = this._computePortalFlightRimAndTint(
          flight.portalSpec,
          flightPose.z,
          ease,
          flight.reverse
        );
        if (flightInst) {
          this._applyDioramaInstanceTint(flightInst, tintRgb);
          this._setWallMountReliefScale(flightInst, flight.reverse ? 1.0 - ease : ease);
          this._setWallMountOuterWallOpacity(
            flightInst,
            this._wallMountOuterWallOpacity(flightInst.spec, t, flight.reverse)
          );
          if (flightInst.spec.veil) {
            if (flight.reverse) {
              const veilStart = 1.0 - flightInst.spec.veil.dissolveEndProgress;
              const vt = THREE.MathUtils.clamp(
                (t - veilStart) / Math.max(1e-5, 1.0 - veilStart),
                0.0,
                1.0
              );
              flightInst.veilCurrentOpacity =
                flightInst.spec.veil.restOpacity * (vt * vt * (3.0 - 2.0 * vt));
            } else {
              const veilEnd = Math.max(0.08, flightInst.spec.veil.dissolveEndProgress);
              const vt = THREE.MathUtils.clamp(t / veilEnd, 0.0, 1.0);
              const veilEase = vt * vt * (3.0 - 2.0 * vt);
              flightInst.veilCurrentOpacity =
                flightInst.veilFlightStartOpacity * (1.0 - veilEase);
            }
          }
        }
        this._setHostPortalSurroundingOpacity(
          flight.hostSceneId,
          rimOpacity,
          flight.portalSpec.id
        );
        this._updatePortalClippingPlanes(null);

        if (t >= 1.0) {
          completePortalFlight();
        }

        this.renderer.setRenderTarget(null);
        this.renderer.render(this.sceneObjects[this.currentSceneId].scene, this.camera);
      } else {
        if (t >= 1.0) {
          completePortalFlight();
          this.renderer.setRenderTarget(null);
          this.renderer.render(this.sceneObjects[this.currentSceneId].scene, this.camera);
        } else if (flight.reverse) {
          const revBlendEnd = SCENE_GEOMETRY_SPEC.directPortalReverseBlendEndT;
          const blendProgress = THREE.MathUtils.clamp(
            t / Math.max(1e-5, revBlendEnd),
            0.0,
            1.0
          );
          if (blendProgress >= 0.9999) {
            this._updateScenePlaneVisibilitiesAndOpacities(flight.hostSceneId);
            if (this._sceneNeedsLocalClipping(flight.hostSceneId)) {
              this._updatePortalClippingPlanes();
            }
            this.renderer.setRenderTarget(null);
            this.renderer.render(this.sceneObjects[flight.hostSceneId].scene, this.camera);
          } else {
            this._updateScenePlaneVisibilitiesAndOpacities(flight.hostSceneId);
            if (this._sceneNeedsLocalClipping(flight.hostSceneId)) {
              this._updatePortalClippingPlanes();
            }
            this.renderer.setRenderTarget(this.rtNext);
            this.renderer.render(this.sceneObjects[flight.hostSceneId].scene, this.camera);

            const childBase =
              flight.childStartPose || this._getSceneRestPose(flight.targetSceneId);
            const childRollDeg = -(flight.flightRollDeg || 0.0) * ease;
            const childOutPose = {
              ...this._clampPoseForScene(flight.targetSceneId, {
                x: childBase.x,
                y: childBase.y,
                z: childBase.z,
                yawDeg: childBase.yawDeg * (1.0 - ease),
                pitchDeg: childBase.pitchDeg * (1.0 - ease),
                rollDeg: childRollDeg
              }),
              rollDeg: childRollDeg
            };
            this._applyPoseToCameraObject(childOutPose);
            this._updateScenePlaneVisibilitiesAndOpacities(flight.targetSceneId);
            if (this._sceneNeedsLocalClipping(flight.targetSceneId)) {
              this._updatePortalClippingPlanes();
            }
            this.renderer.setRenderTarget(this.rtCurrent);
            this.renderer.render(this.sceneObjects[flight.targetSceneId].scene, this.camera);

            this._applyPoseToCameraObject(flightPose);
            this.renderer.setRenderTarget(null);
            this.transitionMaterial.uniforms.uTexCurrent.value = this.rtCurrent.texture;
            this.transitionMaterial.uniforms.uTexNext.value = this.rtNext.texture;
            this.transitionMaterial.uniforms.uProgress.value = blendProgress;
            this.renderer.render(this.transitionScene, this.transitionCamera);
          }
        } else {
          const blendStart =
            typeof flight.portalSpec.fadeStartProgress === 'number'
              ? flight.portalSpec.fadeStartProgress
              : SCENE_GEOMETRY_SPEC.directPortalBlendStartT;
          const blendProgress = THREE.MathUtils.clamp(
            (t - blendStart) / Math.max(1e-5, 1.0 - blendStart),
            0.0,
            1.0
          );
          if (blendProgress <= 1e-4) {
            this._updateScenePlaneVisibilitiesAndOpacities(flight.hostSceneId);
            if (this._sceneNeedsLocalClipping(flight.hostSceneId)) {
              this._updatePortalClippingPlanes();
            }
            this.renderer.setRenderTarget(null);
            this.renderer.render(this.sceneObjects[flight.hostSceneId].scene, this.camera);
          } else {
            this._updateScenePlaneVisibilitiesAndOpacities(flight.hostSceneId);
            if (this._sceneNeedsLocalClipping(flight.hostSceneId)) {
              this._updatePortalClippingPlanes();
            }
            this.renderer.setRenderTarget(this.rtCurrent);
            this.renderer.render(this.sceneObjects[flight.hostSceneId].scene, this.camera);

            const targetRest = this._getSceneRestPose(flight.targetSceneId);
            const inRollDeg = -(flight.flightRollDeg || 0.0) * (1.0 - ease);
            const inPose = {
              ...this._clampPoseForScene(flight.targetSceneId, {
                x: targetRest.x,
                y: targetRest.y,
                z: targetRest.z,
                yawDeg: targetRest.yawDeg,
                pitchDeg: targetRest.pitchDeg,
                rollDeg: inRollDeg
              }),
              rollDeg: inRollDeg
            };
            this._applyPoseToCameraObject(inPose);
            this._updateScenePlaneVisibilitiesAndOpacities(flight.targetSceneId);
            if (this._sceneNeedsLocalClipping(flight.targetSceneId)) {
              this._updatePortalClippingPlanes();
            }
            this.renderer.setRenderTarget(this.rtNext);
            this.renderer.render(this.sceneObjects[flight.targetSceneId].scene, this.camera);

            this._applyPoseToCameraObject(flightPose);

            this.renderer.setRenderTarget(null);
            this.transitionMaterial.uniforms.uTexCurrent.value = this.rtCurrent.texture;
            this.transitionMaterial.uniforms.uTexNext.value = this.rtNext.texture;
            this.transitionMaterial.uniforms.uProgress.value = blendProgress;
            this.renderer.render(this.transitionScene, this.transitionCamera);
          }
        }
      }

      if (this._frameListeners.size > 0) {
        for (const listener of this._frameListeners) {
          listener(this.currentSceneId, this.camera, Boolean(this._portalFlight));
        }
      }
      return;
    }

    this._stepKeyboardNavigation(dtSec);

    const activeDrag = this._isTouching || this._isMouseDragging;
    const basePoseDamping = activeDrag
      ? SCENE_GEOMETRY_SPEC.poseDragDamping60Hz
      : SCENE_GEOMETRY_SPEC.poseIdleDamping60Hz;
    const poseDamping = 1.0 - Math.pow(1.0 - basePoseDamping, dampingFrameScale);
    const pointerDamping =
      1.0 - Math.pow(1.0 - SCENE_GEOMETRY_SPEC.pointerDamping60Hz, dampingFrameScale);
    const hoverDamping =
      1.0 - Math.pow(1.0 - SCENE_GEOMETRY_SPEC.hoverDamping60Hz, dampingFrameScale);

    this.pointer.x += (this.pointer.targetX - this.pointer.x) * pointerDamping;
    this.pointer.y += (this.pointer.targetY - this.pointer.y) * pointerDamping;
    this._hoverNorm.x += (this._hoverNorm.targetX - this._hoverNorm.x) * hoverDamping;
    this._hoverNorm.y += (this._hoverNorm.targetY - this._hoverNorm.y) * hoverDamping;

    this.poseCurrent.x += (this.poseTarget.x - this.poseCurrent.x) * poseDamping;
    this.poseCurrent.y += (this.poseTarget.y - this.poseCurrent.y) * poseDamping;
    this.poseCurrent.z += (this.poseTarget.z - this.poseCurrent.z) * poseDamping;
    this.poseCurrent.yawDeg += (this.poseTarget.yawDeg - this.poseCurrent.yawDeg) * poseDamping;
    this.poseCurrent.pitchDeg +=
      (this.poseTarget.pitchDeg - this.poseCurrent.pitchDeg) * poseDamping;
    this.poseCurrent.rollDeg =
      (this.poseCurrent.rollDeg || 0) +
      ((this.poseTarget.rollDeg || 0) - (this.poseCurrent.rollDeg || 0)) * poseDamping;

    this._syncCameraTransform();

    for (const key in this.sceneObjects) {
      this._updateScenePlaneVisibilitiesAndOpacities(key);
    }

    // Smoothly damp portal veil opacity (e.g., scroll pencil drawing breathing open on hover)
    if (this._portalInstances.length > 0) {
      const veilDamping = 1.0 - Math.pow(1.0 - 0.14, dampingFrameScale);
      for (const inst of this._portalInstances) {
        if (!inst.veilMesh || !inst.veilMat || !inst.spec.veil) continue;
        const prevVeilOp = inst.veilCurrentOpacity;
        inst.veilCurrentOpacity +=
          (inst.veilTargetOpacity - inst.veilCurrentOpacity) * veilDamping;
        const hostObj = this.sceneObjects[inst.hostSceneId];
        const surroundingOp =
          hostObj && typeof hostObj.portalSurroundingOpacity === 'number'
            ? hostObj.portalSurroundingOpacity
            : 1.0;
        const effOp = inst.veilCurrentOpacity * surroundingOp;
        inst.veilMat.opacity = effOp;
        inst.veilMesh.visible = effOp > 0.005;
        if (Math.abs(inst.veilCurrentOpacity - prevVeilOp) > 1e-4) {
          this._syncDioramaMirrorsForScene(inst.hostSceneId);
        }
      }
    }

    if (!this.isPaused) {
      for (const key in this.sceneObjects) {
        const particles = this.sceneObjects[key].layer3;
        if (particles) {
          particles.rotation.y += 0.0004;
          particles.rotation.x += 0.0002;
        }
      }
      if (this.hasPortalShelves(this.currentSceneId)) {
        for (const inst of this._portalInstances) {
          if (inst.hostSceneId !== this.currentSceneId) continue;
          const tObj = this.sceneObjects[inst.targetSceneId];
          if (inst.dustMesh && tObj && tObj.layer3) {
            inst.dustMesh.rotation.copy(tObj.layer3.rotation);
          }
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
      const t = this.transitionProgress;
      const ease = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);

      const fromBase = this._transitionFromPose || this._getSceneRestPose(this.currentSceneId);
      const toBase = this._transitionToPose || this._getSceneRestPose(this.targetSceneId);

      // Outgoing scene: gentle forward depth glide along camera bore axis (-Z)
      const outPose = this._clampPoseForScene(this.currentSceneId, {
        x: fromBase.x,
        y: fromBase.y,
        z: fromBase.z - SCENE_GEOMETRY_SPEC.transitionGlideOutDollyZ * ease,
        yawDeg: fromBase.yawDeg,
        pitchDeg: fromBase.pitchDeg
      });
      this._applyPoseToCameraObject(outPose);
      this._updateScenePlaneVisibilitiesAndOpacities(this.currentSceneId);
      const curNeedsClip = this._sceneNeedsLocalClipping(this.currentSceneId);
      this.renderer.localClippingEnabled = curNeedsClip;
      if (curNeedsClip) {
        this._updatePortalClippingPlanes();
      }
      this.renderer.setRenderTarget(this.rtCurrent);
      this.renderer.render(this.sceneObjects[this.currentSceneId].scene, this.camera);

      // Incoming scene: glide cleanly forward from slightly behind rest pose into exact rest pose
      const gyroYaw = this.gyroActive ? this._gyroYawDeg : 0;
      const gyroPitch = this.gyroActive ? this._gyroPitchDeg : 0;
      const gyroNormX = gyroYaw / SCENE_GEOMETRY_SPEC.gyroMaxYawDeg;
      const gyroNormY = gyroPitch / SCENE_GEOMETRY_SPEC.gyroMaxPitchDeg;
      const inPose = this._clampPoseForScene(this.targetSceneId, {
        x:
          toBase.x +
          (-gyroNormX * SCENE_GEOMETRY_SPEC.gyroParallaxShiftX +
            this._hoverNorm.x * SCENE_GEOMETRY_SPEC.hoverAmbientShiftX +
            this.pointer.x * SCENE_GEOMETRY_SPEC.pointerExcursionShiftX) *
            ease,
        y:
          toBase.y +
          (gyroNormY * SCENE_GEOMETRY_SPEC.gyroParallaxShiftY +
            this._hoverNorm.y * SCENE_GEOMETRY_SPEC.hoverAmbientShiftY +
            this.pointer.y * SCENE_GEOMETRY_SPEC.pointerExcursionShiftY) *
            ease,
        z: toBase.z + SCENE_GEOMETRY_SPEC.transitionGlideInOffsetZ * (1.0 - ease),
        yawDeg:
          toBase.yawDeg +
          (gyroYaw -
            this._hoverNorm.x * SCENE_GEOMETRY_SPEC.hoverAmbientYawDeg -
            this.pointer.x * SCENE_GEOMETRY_SPEC.pointerExcursionYawDeg) *
            ease,
        pitchDeg:
          toBase.pitchDeg +
          (gyroPitch +
            this._hoverNorm.y * SCENE_GEOMETRY_SPEC.hoverAmbientPitchDeg +
            this.pointer.y * SCENE_GEOMETRY_SPEC.pointerExcursionPitchDeg) *
            ease
      });
      this._applyPoseToCameraObject(inPose);
      this._updateScenePlaneVisibilitiesAndOpacities(this.targetSceneId);
      const nextNeedsClip = this._sceneNeedsLocalClipping(this.targetSceneId);
      this.renderer.localClippingEnabled = nextNeedsClip;
      if (nextNeedsClip) {
        this._updatePortalClippingPlanes();
      }
      this.renderer.setRenderTarget(this.rtNext);
      this.renderer.render(this.sceneObjects[this.targetSceneId].scene, this.camera);

      this.renderer.setRenderTarget(null);
      this.transitionMaterial.uniforms.uTexCurrent.value = this.rtCurrent.texture;
      this.transitionMaterial.uniforms.uTexNext.value = this.rtNext.texture;
      this.transitionMaterial.uniforms.uProgress.value = this.transitionProgress;
      this.renderer.render(this.transitionScene, this.transitionCamera);

      if (this.transitionProgress >= 1.0) {
        this.isTransitioning = false;
        this.currentSceneId = this.targetSceneId;
        this.targetSceneId = null;
        this._transitionFromPose = null;
        const settledObj = this.sceneObjects[this.currentSceneId];
        if (settledObj) {
          this._innerScale = settledObj.innerScale;
          this._layer0Scale = settledObj.layer0Scale;
          window.__PROTOTYPE_STATE__.innerLayerScale = this._innerScale;
          window.__PROTOTYPE_STATE__.layer0Scale = this._layer0Scale;
        }
        this.setCameraPose(toBase, true);
        this._transitionToPose = null;
        window.__PROTOTYPE_STATE__.currentScene = this.currentSceneId;
        window.__PROTOTYPE_STATE__.isTransitioning = false;
      }
    } else {
      const needsClip = this._sceneNeedsLocalClipping(this.currentSceneId);
      this.renderer.localClippingEnabled = needsClip;
      if (needsClip) {
        this._updatePortalClippingPlanes();
      }
      this.renderer.setRenderTarget(null);
      this.renderer.render(this.sceneObjects[this.currentSceneId].scene, this.camera);
    }

    if (this._frameListeners.size > 0) {
      for (const listener of this._frameListeners) {
        listener(this.currentSceneId, this.camera, this.isTransitioning);
      }
    }
  }

  setInteractionCapture(captured) {
    this.interactionCaptureActive = !!captured;
    if (this.interactionCaptureActive) {
      this._isMouseDragging = false;
      this._isTouching = false;
      this._touchMode = 'none';
      this._keysHeld.clear();
      this._hoverNorm.targetX = 0;
      this._hoverNorm.targetY = 0;
      if (this.canvas) {
        this.canvas.style.cursor = 'crosshair';
      }
    } else if (this.canvas) {
      this.canvas.style.cursor = 'grab';
    }
  }

  addFrameListener(fn) {
    if (typeof fn === 'function') {
      this._frameListeners.add(fn);
    }
    return () => this._frameListeners.delete(fn);
  }

  removeFrameListener(fn) {
    this._frameListeners.delete(fn);
  }

  getCameraPoseSnapshot() {
    return {
      x: Number(this.camera.position.x.toFixed(3)),
      y: Number(this.camera.position.y.toFixed(3)),
      z: Number(this.camera.position.z.toFixed(3)),
      yaw_deg: Number(window.__PROTOTYPE_STATE__.cameraRotationDeg.yaw.toFixed(3)),
      pitch_deg: Number(window.__PROTOTYPE_STATE__.cameraRotationDeg.pitch.toFixed(3)),
      fov_deg: Number(this.camera.fov.toFixed(3))
    };
  }

  flyToCameraPose(sceneId, pose, immediate = false) {
    if (sceneId && this.sceneConfigs[sceneId] && sceneId !== this.currentSceneId) {
      this.transitionTo(sceneId);
    }
    if (!pose || typeof pose !== 'object') {
      return null;
    }
    this._hoverNorm.targetX = 0;
    this._hoverNorm.targetY = 0;
    this.pointer.targetX = 0;
    this.pointer.targetY = 0;
    return this.setCameraPose(
      {
        x: typeof pose.x === 'number' ? pose.x : this.poseTarget.x,
        y: typeof pose.y === 'number' ? pose.y : this.poseTarget.y,
        z: typeof pose.z === 'number' ? pose.z : this.poseTarget.z,
        yawDeg: typeof pose.yaw_deg === 'number' ? pose.yaw_deg : this.poseTarget.yawDeg,
        pitchDeg: typeof pose.pitch_deg === 'number' ? pose.pitch_deg : this.poseTarget.pitchDeg
      },
      immediate
    );
  }

  _getTextureSampler(tex, maxDim = 512) {
    if (!tex || !tex.image) return null;
    const img = tex.image;
    const origW = img.naturalWidth || img.videoWidth || img.width || 0;
    const origH = img.naturalHeight || img.videoHeight || img.height || 0;
    if (origW <= 0 || origH <= 0) return null;

    const cached = this._textureSamplers.get(tex);
    if (cached && cached.maxDim === maxDim) {
      return cached;
    }

    const scale = Math.min(1.0, maxDim / Math.max(origW, origH));
    const sw = Math.max(1, Math.round(origW * scale));
    const sh = Math.max(1, Math.round(origH * scale));
    const offscreen = document.createElement('canvas');
    offscreen.width = sw;
    offscreen.height = sh;
    const ctx = offscreen.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, sw, sh);
    const pixels = ctx.getImageData(0, 0, sw, sh).data;

    const sampler = {
      origW,
      origH,
      sw,
      sh,
      maxDim,
      pixels,
      sampleAtUv(u, v) {
        const cu = THREE.MathUtils.clamp(u, 0, 1);
        const cv = THREE.MathUtils.clamp(v, 0, 1);
        const sx = Math.min(sw - 1, Math.max(0, Math.round(cu * (sw - 1))));
        const sy = Math.min(sh - 1, Math.max(0, Math.round((1.0 - cv) * (sh - 1))));
        const idx = (sy * sw + sx) * 4;
        return {
          r: pixels[idx],
          g: pixels[idx + 1],
          b: pixels[idx + 2],
          a: pixels[idx + 3],
          pxX: Math.round(cu * (origW - 1)),
          pxY: Math.round((1.0 - cv) * (origH - 1))
        };
      }
    };
    this._textureSamplers.set(tex, sampler);
    return sampler;
  }

  _intersectPlaneZ(clientX, clientY, planeZ) {
    const rect = this.canvas.getBoundingClientRect();
    const normX = THREE.MathUtils.clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const normY = THREE.MathUtils.clamp((clientY - rect.top) / Math.max(1, rect.height), 0, 1);
    const ndc = new THREE.Vector2(normX * 2 - 1, -(normY * 2 - 1));
    this.camera.updateMatrixWorld();
    this._raycaster.setFromCamera(ndc, this.camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -planeZ);
    const target = new THREE.Vector3();
    const hit = this._raycaster.ray.intersectPlane(plane, target);
    return hit ? [Number(hit.x.toFixed(3)), Number(hit.y.toFixed(3)), Number(hit.z.toFixed(3))] : null;
  }

  pickSpatialTarget(clientX, clientY, pickingCfg) {
    const alphaThreshold = pickingCfg.alpha_hit_threshold;
    const sampleMaxDim = pickingCfg.alpha_sample_max_dim;
    const rect = this.canvas.getBoundingClientRect();
    const normX = THREE.MathUtils.clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const normY = THREE.MathUtils.clamp((clientY - rect.top) / Math.max(1, rect.height), 0, 1);
    const ndc = new THREE.Vector2(normX * 2 - 1, -(normY * 2 - 1));

    this.camera.updateMatrixWorld();
    this._raycaster.setFromCamera(ndc, this.camera);

    const sceneObj = this.sceneObjects[this.currentSceneId];
    if (!sceneObj) return null;

    const alphaCandidates = [
      {
        mesh: sceneObj.layer2,
        tex: sceneObj.fgTex,
        zPos: sceneObj.sceneFgZ,
        layerId: 'foreground',
        layerLabelHe: 'שכבה קדמית (מיצב / דמות)'
      },
      {
        mesh: sceneObj.layer1,
        tex: sceneObj.midTex,
        zPos: sceneObj.sceneMidZ,
        layerId: 'midground',
        layerLabelHe: 'שכבת ביניים (פיסול / עומק אמצעי)'
      },
      {
        mesh: sceneObj.reflectionMesh,
        tex: sceneObj.refTex,
        zPos: SCENE_GEOMETRY_SPEC.reflectionZ,
        layerId: 'reflection',
        layerLabelHe: 'השתקפות מראה (מרחב עליון)'
      }
    ];
    if (Array.isArray(sceneObj.extraLayers)) {
      for (const extra of sceneObj.extraLayers) {
        alphaCandidates.push({
          mesh: extra.mesh,
          tex: extra.tex,
          zPos: extra.zPos,
          layerId: extra.id,
          layerLabelHe: extra.labelHe
        });
      }
    }
    if (this.hasPortalShelves(this.currentSceneId)) {
      for (const inst of this._portalInstances) {
        if (inst.hostSceneId !== this.currentSceneId) continue;
        if (
          inst.occluderMesh &&
          inst.occluderMesh.visible &&
          inst.occluderTex &&
          !this._erasedPortalOccluders.has(inst.spec.id)
        ) {
          const occZ =
            inst.spec.occluder && typeof inst.spec.occluder.worldZ === 'number'
              ? inst.spec.occluder.worldZ
              : SCENE_GEOMETRY_SPEC.portalOccluderWorldZ;
          alphaCandidates.push({
            mesh: inst.occluderMesh,
            tex: inst.occluderTex,
            zPos: occZ,
            layerId: `occluder_${inst.spec.id}`,
            layerLabelHe: `חפץ מסתיר: ${
              (inst.spec.occluder && inst.spec.occluder.labelHe) || inst.spec.labelHe
            }`
          });
        }
      }
    }
    alphaCandidates.sort((a, b) => b.zPos - a.zPos);

    for (const cand of alphaCandidates) {
      if (!cand.mesh || !cand.mesh.visible || !cand.tex) continue;
      const hits = this._raycaster.intersectObject(cand.mesh, false);
      if (hits.length > 0 && hits[0].uv) {
        const hitUvX = hits[0].uv.x;
        const uniforms = cand.mesh.material && cand.mesh.material.uniforms;
        if (uniforms) {
          const clipMinX = uniforms.uClipUvMinX ? uniforms.uClipUvMinX.value : 0.0;
          const clipMaxX = uniforms.uClipUvMaxX ? uniforms.uClipUvMaxX.value : 1.0;
          if (hitUvX < clipMinX || hitUvX > clipMaxX) continue;
        }
        const sampler = this._getTextureSampler(cand.tex, sampleMaxDim);
        if (sampler) {
          const sample = sampler.sampleAtUv(hits[0].uv.x, hits[0].uv.y);
          if (sample.a >= alphaThreshold) {
            const pt = hits[0].point;
            return {
              layerId: cand.layerId,
              layerLabelHe: cand.layerLabelHe,
              worldPoint: [Number(pt.x.toFixed(3)), Number(pt.y.toFixed(3)), Number(pt.z.toFixed(3))],
              textureUv: [Number(hits[0].uv.x.toFixed(5)), Number(hits[0].uv.y.toFixed(5))],
              texturePixel: [sample.pxX, sample.pxY],
              pinScreenNorm: [Number(normX.toFixed(5)), Number(normY.toFixed(5))]
            };
          }
        }
      }
    }

    if (this.hasPortalShelves(this.currentSceneId)) {
      const portalHit = this._raycastPortalShelfAtClient(clientX, clientY);
      if (portalHit) {
        const portalPlaneZ = portalHit.worldZ - portalHit.recessDepth;
        const portalWorld = this._intersectPlaneZ(clientX, clientY, portalPlaneZ) || [
          portalHit.worldX,
          portalHit.worldY,
          Number(portalPlaneZ.toFixed(3))
        ];
        return {
          layerId: `portal_${portalHit.id}`,
          layerLabelHe: `שער מעבר תלת-ממדי: ${portalHit.labelHe}`,
          worldPoint: portalWorld,
          textureUv: null,
          texturePixel: null,
          pinScreenNorm: [Number(normX.toFixed(5)), Number(normY.toFixed(5))]
        };
      }
    }

    // Background Layer 0 (with Depth Anything V2 relief displacement compensation)
    const bgHits = this._raycaster.intersectObject(sceneObj.layer0, false);
    if (bgHits.length > 0 && bgHits[0].uv) {
      const uv = bgHits[0].uv;
      let zDisp = 0.0;
      let pxCoord = null;
      const depthSampler = this._getTextureSampler(sceneObj.depthTex, sampleMaxDim);
      const bgSampler = this._getTextureSampler(sceneObj.bgTex, sampleMaxDim);
      if (depthSampler) {
        const ds = depthSampler.sampleAtUv(uv.x, uv.y);
        const rawDepth = ds.r / 255.0;
        const margin = SCENE_GEOMETRY_SPEC.roomPinMarginUv;
        const smoothstep = (e0, e1, x) => {
          const t = THREE.MathUtils.clamp((x - e0) / Math.max(1e-5, e1 - e0), 0, 1);
          return t * t * (3 - 2 * t);
        };
        const pinX = smoothstep(0.0, margin, uv.x) * smoothstep(0.0, margin, 1.0 - uv.x);
        const pinY = smoothstep(0.0, margin, uv.y) * smoothstep(0.0, margin, 1.0 - uv.y);
        const reliefAmp =
          typeof sceneObj.roomDepthAmplitude === 'number'
            ? sceneObj.roomDepthAmplitude
            : SCENE_GEOMETRY_SPEC.roomDepthReliefAmplitude;
        zDisp = rawDepth * reliefAmp * pinX * pinY;
      }
      if (bgSampler) {
        const bs = bgSampler.sampleAtUv(uv.x, uv.y);
        pxCoord = [bs.pxX, bs.pxY];
      }
      const sceneBgZ =
        typeof sceneObj.sceneBgZ === 'number' ? sceneObj.sceneBgZ : SCENE_GEOMETRY_SPEC.bgZ;
      const displacedZ = sceneBgZ + zDisp;
      const displacedWorld =
        this._intersectPlaneZ(clientX, clientY, displacedZ) || [
          Number(bgHits[0].point.x.toFixed(3)),
          Number(bgHits[0].point.y.toFixed(3)),
          Number(displacedZ.toFixed(3))
        ];

      return {
        layerId: 'background',
        layerLabelHe: 'רקע המרחב המורחב (חדר / חצר)',
        worldPoint: displacedWorld,
        textureUv: [Number(uv.x.toFixed(5)), Number(uv.y.toFixed(5))],
        texturePixel: pxCoord,
        pinScreenNorm: [Number(normX.toFixed(5)), Number(normY.toFixed(5))]
      };
    }

    const fallbackBgZ =
      typeof sceneObj.sceneBgZ === 'number' ? sceneObj.sceneBgZ : SCENE_GEOMETRY_SPEC.bgZ;
    const fallbackWorld = this._intersectPlaneZ(clientX, clientY, fallbackBgZ);
    return {
      layerId: 'background',
      layerLabelHe: 'רקע המרחב המורחב (חדר / חצר)',
      worldPoint: fallbackWorld,
      textureUv: null,
      texturePixel: null,
      pinScreenNorm: [Number(normX.toFixed(5)), Number(normY.toFixed(5))]
    };
  }

  pickSpatialRegion(boxClientPx, pickingCfg) {
    const canvasRect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, canvasRect.width);
    const h = Math.max(1, canvasRect.height);

    const x0 = Math.min(boxClientPx.x0, boxClientPx.x1);
    const y0 = Math.min(boxClientPx.y0, boxClientPx.y1);
    const x1 = Math.max(boxClientPx.x0, boxClientPx.x1);
    const y1 = Math.max(boxClientPx.y0, boxClientPx.y1);

    const cx = 0.5 * (x0 + x1);
    const cy = 0.5 * (y0 + y1);
    const centerHit = this.pickSpatialTarget(cx, cy, pickingCfg);

    let chosenHit = centerHit;
    if (centerHit && centerHit.layerId === 'background') {
      const gridN = Math.max(2, pickingCfg.region_grid_samples_per_axis);
      const activeObj = this.sceneObjects[this.currentSceneId];
      const extraOrder =
        activeObj && Array.isArray(activeObj.extraLayers)
          ? [...activeObj.extraLayers].sort((a, b) => b.zPos - a.zPos).map((e) => e.id)
          : [];
      const priorityOrder = ['foreground', ...extraOrder, 'midground', 'reflection'];
      const bucket = {};
      for (const k of priorityOrder) {
        bucket[k] = [];
      }
      for (let gx = 0; gx < gridN; gx += 1) {
        for (let gy = 0; gy < gridN; gy += 1) {
          const sx = x0 + ((gx + 0.5) / gridN) * (x1 - x0);
          const sy = y0 + ((gy + 0.5) / gridN) * (y1 - y0);
          const sampleHit = this.pickSpatialTarget(sx, sy, pickingCfg);
          if (sampleHit && bucket[sampleHit.layerId]) {
            bucket[sampleHit.layerId].push(sampleHit);
          }
        }
      }
      for (const layerKey of priorityOrder) {
        if (bucket[layerKey].length >= 2) {
          const midIdx = Math.floor(bucket[layerKey].length / 2);
          chosenHit = bucket[layerKey][midIdx];
          break;
        }
      }
    }

    const anchorWorld = chosenHit ? chosenHit.worldPoint : null;
    const planeZ = anchorWorld ? anchorWorld[2] : SCENE_GEOMETRY_SPEC.fgZ;
    const c0 = this._intersectPlaneZ(x0, y0, planeZ);
    const c1 = this._intersectPlaneZ(x1, y0, planeZ);
    const c2 = this._intersectPlaneZ(x1, y1, planeZ);
    const c3 = this._intersectPlaneZ(x0, y1, planeZ);
    const worldBoxCorners = c0 && c1 && c2 && c3 ? [c0, c1, c2, c3] : null;

    const screenRectNorm = {
      x0: Number(THREE.MathUtils.clamp((x0 - canvasRect.left) / w, 0, 1).toFixed(5)),
      y0: Number(THREE.MathUtils.clamp((y0 - canvasRect.top) / h, 0, 1).toFixed(5)),
      x1: Number(THREE.MathUtils.clamp((x1 - canvasRect.left) / w, 0, 1).toFixed(5)),
      y1: Number(THREE.MathUtils.clamp((y1 - canvasRect.top) / h, 0, 1).toFixed(5))
    };
    const pinNorm = [
      Number((0.5 * (screenRectNorm.x0 + screenRectNorm.x1)).toFixed(5)),
      Number((0.5 * (screenRectNorm.y0 + screenRectNorm.y1)).toFixed(5))
    ];

    return {
      mode: 'box',
      layer_id: chosenHit ? chosenHit.layerId : 'background',
      layer_label_he: chosenHit ? chosenHit.layerLabelHe : 'רקע המרחב המורחב (חדר / חצר)',
      screen_rect_norm: screenRectNorm,
      pin_screen_norm: pinNorm,
      world_point: anchorWorld,
      world_box_corners: worldBoxCorners,
      texture_uv: chosenHit ? chosenHit.textureUv : null,
      texture_pixel: chosenHit ? chosenHit.texturePixel : null,
      ui_selector: null
    };
  }

  projectWorldPointToViewport(worldPoint, nearClipZ = 5.0) {
    if (!worldPoint || worldPoint.length !== 3 || !this.camera) return null;
    const vec = new THREE.Vector3(worldPoint[0], worldPoint[1], worldPoint[2]);
    const camSpace = vec.clone().applyMatrix4(this.camera.matrixWorldInverse);
    if (-camSpace.z < nearClipZ) {
      return { visible: false, clientX: 0, clientY: 0, normX: 0, normY: 0 };
    }
    vec.project(this.camera);
    const rect = this.canvas.getBoundingClientRect();
    const normX = (vec.x + 1) * 0.5;
    const normY = (1 - vec.y) * 0.5;
    const inFrustum =
      vec.z >= -1.0 && vec.z <= 1.0 && normX >= -0.08 && normX <= 1.08 && normY >= -0.08 && normY <= 1.08;
    return {
      visible: inFrustum,
      normX,
      normY,
      clientX: rect.left + normX * rect.width,
      clientY: rect.top + normY * rect.height
    };
  }

  projectWorldBoxToViewport(worldBoxCorners, nearClipZ = 5.0) {
    if (!worldBoxCorners || worldBoxCorners.length !== 4) return null;
    const projected = [];
    for (const corner of worldBoxCorners) {
      const p = this.projectWorldPointToViewport(corner, nearClipZ);
      if (!p || !p.visible) return null;
      projected.push(p);
    }
    const xs = projected.map((p) => p.clientX);
    const ys = projected.map((p) => p.clientY);
    return {
      left: Math.min(...xs),
      top: Math.min(...ys),
      right: Math.max(...xs),
      bottom: Math.max(...ys)
    };
  }

  renderCleanFrame() {
    if (!this.renderer || !this.camera) return false;
    const sceneObj = this.sceneObjects[this.currentSceneId];
    if (!sceneObj || !sceneObj.scene) return false;
    this._updateScenePlaneVisibilitiesAndOpacities(this.currentSceneId);
    if (this.hasPortalShelves(this.currentSceneId)) {
      this._ensurePortalShelvesInitialized(this.currentSceneId);
      if (this._sceneNeedsLocalClipping(this.currentSceneId)) {
        this.renderer.localClippingEnabled = true;
        this._updatePortalClippingPlanes();
      }
    }
    this.renderer.setRenderTarget(null);
    this.renderer.render(sceneObj.scene, this.camera);
    return true;
  }

  captureAnnotatedSnapshot(highlightRecord, pickingCfg) {
    const sceneObj = this.sceneObjects[this.currentSceneId];
    if (!sceneObj || !this.renderer || !this.canvas) return null;

    // Render one clean synchronous pass to default framebuffer before drawImage
    this._updateScenePlaneVisibilitiesAndOpacities(this.currentSceneId);
    if (this.hasPortalShelves(this.currentSceneId)) {
      this._ensurePortalShelvesInitialized(this.currentSceneId);
      if (this._sceneNeedsLocalClipping(this.currentSceneId)) {
        this.renderer.localClippingEnabled = true;
        this._updatePortalClippingPlanes();
      }
    }
    this.renderer.setRenderTarget(null);
    this.renderer.render(sceneObj.scene, this.camera);

    const srcW = Math.max(1, this.canvas.width);
    const srcH = Math.max(1, this.canvas.height);
    const maxW = pickingCfg.snapshot_max_width_px;
    const scale = Math.min(1.0, maxW / srcW);
    const outW = Math.max(16, Math.round(srcW * scale));
    const outH = Math.max(16, Math.round(srcH * scale));

    const snapCanvas = document.createElement('canvas');
    snapCanvas.width = outW;
    snapCanvas.height = outH;
    const ctx = snapCanvas.getContext('2d');
    ctx.drawImage(this.canvas, 0, 0, outW, outH);

    if (highlightRecord && highlightRecord.screen_rect_norm) {
      const r = highlightRecord.screen_rect_norm;
      const rx = r.x0 * outW;
      const ry = r.y0 * outH;
      const rw = Math.max(12, (r.x1 - r.x0) * outW);
      const rh = Math.max(12, (r.y1 - r.y0) * outH);

      // Subtle dim outside highlighted region so the target pops immediately
      ctx.save();
      ctx.fillStyle = 'rgba(10, 12, 16, 0.36)';
      ctx.beginPath();
      ctx.rect(0, 0, outW, outH);
      ctx.rect(rx, ry, rw, rh);
      ctx.fill('evenodd');

      // Warm gold highlight border + glow
      ctx.strokeStyle = '#E6B980';
      ctx.lineWidth = 2.5;
      ctx.shadowColor = 'rgba(230, 185, 128, 0.75)';
      ctx.shadowBlur = 10;
      ctx.strokeRect(rx, ry, rw, rh);

      // Center pin reticle
      if (highlightRecord.pin_screen_norm) {
        const px = highlightRecord.pin_screen_norm[0] * outW;
        const py = highlightRecord.pin_screen_norm[1] * outH;
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#C49B71';
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#121314';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.restore();
    }

    return snapCanvas.toDataURL('image/jpeg', pickingCfg.snapshot_jpeg_quality);
  }

  dispose() {
    this._disposed = true;
    this._frameListeners.clear();
    if (this.animId !== null) {
      cancelAnimationFrame(this.animId);
      this.animId = null;
    }

    if (this._gyroNoEventsTimer !== null) {
      clearTimeout(this._gyroNoEventsTimer);
      this._gyroNoEventsTimer = null;
    }
    this._clearAllOccluderRegenTimers();

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
    window.removeEventListener('orientationchange', this._onScreenOrientationChange);
    if (
      typeof window !== 'undefined' &&
      window.screen &&
      window.screen.orientation &&
      typeof window.screen.orientation.removeEventListener === 'function'
    ) {
      window.screen.orientation.removeEventListener('change', this._onScreenOrientationChange);
    }
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this._onVisibilityChange);
    }
    window.removeEventListener('resize', this._onResize);
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    window.removeEventListener('blur', this._onWindowBlur);

    this._disposePortalShelves();

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

    if (this._sharedUrlTextures) {
      for (const entry of this._sharedUrlTextures.values()) {
        if (entry && entry.tex && typeof entry.tex.dispose === 'function') {
          entry.tex.dispose();
        }
      }
      this._sharedUrlTextures.clear();
    }
    if (this._sharedPlaneGeos) {
      for (const geo of this._sharedPlaneGeos.values()) {
        if (geo && typeof geo.dispose === 'function') {
          geo.dispose();
        }
      }
      this._sharedPlaneGeos.clear();
    }

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
