// prototype/js/app.js
// Main application entry point for Tamar "Tami" Sheaffer 70th Birthday 2.5D Spatial Prototype
// Manages UI interactions, Web Audio ambient soundscape, keyboard shortcuts, and scene state.

import { SpatialSceneManager } from './scene_manager.js';

class SpatialApp {
  constructor() {
    this.canvas = document.getElementById('spatial-canvas');
    this.loadingOverlay = document.getElementById('loading-overlay');
    this.infoDrawer = document.getElementById('info-drawer');
    this.infoToggleBtn = document.getElementById('info-toggle-btn');
    this.closeDrawerBtn = document.getElementById('close-drawer-btn');
    this.fullscreenBtn = document.getElementById('fullscreen-btn');
    this.audioToggleBtn = document.getElementById('audio-toggle');
    this.gyroToggleBtn = document.getElementById('gyro-toggle');
    this.resetPoseBtn = document.getElementById('reset-pose-btn');
    this.sceneBtns = document.querySelectorAll('.scene-btn');
    this.audioContext = null;
    this.isAudioPlaying = false;

    this.initSceneManager();
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
      }
    });

    // Make manager accessible for debugging/testing
    window.__SPATIAL_MANAGER__ = this.manager;
  }

  initUIListeners() {
    // Scene Selector Buttons
    this.sceneBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        const sceneId = btn.getAttribute('data-scene');
        if (sceneId && this.manager) {
          this.manager.transitionTo(sceneId);
        }
      });
    });

    // Curatorial Info Drawer Toggle
    if (this.infoToggleBtn && this.infoDrawer) {
      this.infoToggleBtn.addEventListener('click', () => {
        this.infoDrawer.classList.toggle('open');
      });
    }

    if (this.closeDrawerBtn && this.infoDrawer) {
      this.closeDrawerBtn.addEventListener('click', () => {
        this.infoDrawer.classList.remove('open');
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
          if (active) {
            this.gyroToggleBtn.classList.add('active');
            this.gyroToggleBtn.textContent = '✓ ג׳יירו פעיל';
          } else {
            this.gyroToggleBtn.classList.remove('active');
            this.gyroToggleBtn.textContent = '⟲ מצב הטיה: עכבר / ג׳יירו';
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

    // Dismiss drawer on clicking outside
    document.addEventListener('click', (e) => {
      if (this.infoDrawer && this.infoDrawer.classList.contains('open')) {
        if (!this.infoDrawer.contains(e.target) && !this.infoToggleBtn.contains(e.target)) {
          this.infoDrawer.classList.remove('open');
        }
      }
    });
  }

  updateActiveSceneUI(sceneId) {
    this.sceneBtns.forEach(btn => {
      if (btn.getAttribute('data-scene') === sceneId) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    // Update curatorial drawer highlights based on current scene
    const storyHerzliya = document.getElementById('story-herzliya');
    const storyDenya = document.getElementById('story-denya');
    if (storyHerzliya && storyDenya) {
      if (sceneId === 'herzliya') {
        storyHerzliya.style.borderColor = 'var(--accent-gold)';
        storyDenya.style.borderColor = 'var(--panel-border)';
      } else {
        storyDenya.style.borderColor = 'var(--accent-gold)';
        storyHerzliya.style.borderColor = 'var(--panel-border)';
      }
    }
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
    let b0 = 0, b1 = 0, b2 = 0;
    for (let i = 0; i < bufferSize; i++) {
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
    const freqs = [146.83, 220.00, 329.63]; // D3, A3, E4
    freqs.forEach(f => {
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
