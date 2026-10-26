// assets/shaders/warped_reflection_transition.glsl
// Warped Reflection Dissolve Transition Shader for Tamar "Tami" Sheaffer 70th Birthday Spatial Journey
// Implements dual-wave ripple & directional warp echoing curved mirror sculptures from "בנה ביתך" (Ramat Gan Museum)
// Includes M4 Reviewer 2 Finding 2 fix: max(0.0, 1.0 - dist) preventing negative color falloff at screen corners.

precision highp float;

uniform sampler2D uTexCurrent;   // Scene 1 render texture
uniform sampler2D uTexNext;      // Scene 2 render texture
uniform float uProgress;         // Transition progress [0.0 -> 1.0]
uniform float uAspect;           // Viewport aspect ratio (width / height)
uniform vec2 uPointer;           // Normalized pointer coordinates [-1, 1]

varying vec2 vUv;

#define PI 3.14159265359

void main() {
    vec2 uv = vUv;
    
    // Calculate normalized distance from center with aspect correction
    vec2 centeredUv = (uv - 0.5) * vec2(uAspect, 1.0);
    float dist = length(centeredUv);
    
    // Envelope for distortion: peaks at progress = 0.5, zero at 0.0 and 1.0
    float envelope = sin(uProgress * PI);
    
    // Wave 1: Concentric ripple expanding outward
    float ripple = sin(dist * 18.0 - uProgress * 12.0) * 0.04 * envelope;
    
    // Wave 2: Directional warp echoing curved metal sheets in Tami's "בנה ביתך"
    float directionalWarp = cos(uv.y * 10.0 + uv.x * 6.0 + uPointer.x * 2.0) * 0.03 * envelope;
    
    // Combined displacement vector
    vec2 displacement = normalize(centeredUv + vec2(0.001)) * ripple + vec2(directionalWarp, -directionalWarp);
    
    // Sample textures with non-linear UV coordinates
    vec2 uvCurrent = uv + displacement * (1.0 - uProgress);
    vec2 uvNext    = uv - displacement * uProgress;
    
    vec4 colCurrent = texture2D(uTexCurrent, uvCurrent);
    vec4 colNext    = texture2D(uTexNext, uvNext);
    
    // Specular silver reflection flash simulating polished metal / mirror glass
    // Clamped distance falloff max(0.0, 1.0 - dist) avoids negative specular color in ultrawide corners
    float specularFlash = smoothstep(0.4, 0.5, uProgress) * (1.0 - smoothstep(0.5, 0.6, uProgress));
    vec3 flashColor = vec3(0.92, 0.94, 0.98) * specularFlash * 0.35 * max(0.0, 1.0 - dist);
    
    // Smoothstep cross-fade
    float blendFactor = smoothstep(0.3, 0.7, uProgress);
    vec4 finalColor = mix(colCurrent, colNext, blendFactor) + vec4(flashColor, 0.0);
    
    gl_FragColor = finalColor;
}
