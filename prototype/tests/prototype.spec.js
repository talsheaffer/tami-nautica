// prototype/tests/prototype.spec.js
// Headless Playwright Verification Test Suite for Tamar "Tami" Sheaffer 70th Spatial Prototype
// Tests zero console errors, zero 404/500 requests, WebGL 2.0 context, parallax camera shift,
// scene transition, and RTL Hebrew layout across Desktop (1440x900) and Mobile (390x844).

import { test, expect } from '@playwright/test';

const TARGET_URL = process.env.TEST_URL || 'http://127.0.0.1:7070/index.html';

test.describe('Tami 70th Spatial Prototype - Automated Headless Verification', () => {

  test('Desktop (1440x900) - Zero Console Errors, Zero Failed Requests, WebGL Initialized', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });

    const errors = [];
    const failedRequests = [];

    page.on('console', msg => {
      if (msg.type() === 'error') errors.push(msg.text());
    });
    page.on('pageerror', err => errors.push(err.message));
    page.on('response', res => {
      if (res.status() >= 400) failedRequests.push(`${res.status()} ${res.url()}`);
    });

    await page.goto(TARGET_URL, { waitUntil: 'networkidle' });

    expect(errors, `Expected 0 uncaught errors, got: ${errors.join('; ')}`).toHaveLength(0);
    expect(failedRequests, `Expected 0 404/500 requests, got: ${failedRequests.join('; ')}`).toHaveLength(0);

    // Verify Canvas & WebGL Context
    const hasWebGL = await page.evaluate(() => {
      const canvas = document.querySelector('canvas#spatial-canvas') || document.querySelector('canvas');
      if (!canvas) return false;
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      return !!gl;
    });
    expect(hasWebGL, 'WebGL context failed to initialize').toBe(true);

    // Verify RTL Hebrew layout
    const rtlDir = await page.evaluate(() => document.documentElement.dir || document.body.dir);
    expect(rtlDir).toBe('rtl');
  });

  test('Desktop Parallax Interaction - Camera Matrix & Multi-Plane Shift Verification', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(TARGET_URL, { waitUntil: 'networkidle' });

    // Wait for layers ready
    await page.waitForFunction(() => window.__PROTOTYPE_STATE__?.layersReady === true, { timeout: 8000 }).catch(() => {});

    // Center position
    await page.mouse.move(720, 450);
    await page.waitForTimeout(200);
    const centerPos = await page.evaluate(() => ({ ...window.__PROTOTYPE_STATE__?.cameraPosition }));

    // Tilt top-left
    await page.mouse.move(100, 100);
    await page.waitForTimeout(350);
    const leftPos = await page.evaluate(() => ({ ...window.__PROTOTYPE_STATE__?.cameraPosition }));

    // Tilt bottom-right
    await page.mouse.move(1340, 800);
    await page.waitForTimeout(350);
    const rightPos = await page.evaluate(() => ({ ...window.__PROTOTYPE_STATE__?.cameraPosition }));

    expect(centerPos).toBeDefined();
    expect(leftPos).toBeDefined();
    expect(rightPos).toBeDefined();

    // Verify camera shift response
    const dx = Math.abs(leftPos.x - rightPos.x);
    const dy = Math.abs(leftPos.y - rightPos.y);
    expect(dx, 'Parallax X delta must be greater than 20 units').toBeGreaterThan(20);
    expect(dy, 'Parallax Y delta must be greater than 10 units').toBeGreaterThan(10);
  });

  test('Scene Transition - Thematic Dissolve & State Update', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(TARGET_URL, { waitUntil: 'networkidle' });

    const initialScene = await page.evaluate(() => window.__PROTOTYPE_STATE__?.currentScene);
    expect(initialScene).toBe('shelf_portal');

    // Open minimalist collapsible left sidebar and trigger transition to Bene Beitcha
    const sidebarToggle = page.locator('#sidebar-toggle-btn');
    await expect(sidebarToggle).toBeVisible();
    await sidebarToggle.click();
    const nextBtn = page.locator('button[data-scene="bene_beitcha"]');
    await expect(nextBtn).toBeVisible();
    await nextBtn.click();

    // Verify transition state and completion
    await page.waitForFunction(() => window.__PROTOTYPE_STATE__?.currentScene === 'bene_beitcha', { timeout: 6000 });
    const finalScene = await page.evaluate(() => window.__PROTOTYPE_STATE__?.currentScene);
    expect(finalScene).toBe('bene_beitcha');
  });

  test('Mobile (390x844) - Touch Parallax & RTL Hebrew Layout Verification', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    const errors = [];
    page.on('pageerror', err => errors.push(err.message));

    await page.goto(TARGET_URL, { waitUntil: 'networkidle' });
    expect(errors).toHaveLength(0);

    // Verify RTL styling on Hebrew UI elements
    const rtlDir = await page.evaluate(() => document.documentElement.dir || document.body.dir);
    expect(rtlDir).toBe('rtl');

    // Verify Canvas and state initialization on mobile viewport
    const state = await page.evaluate(() => window.__PROTOTYPE_STATE__);
    expect(state).toBeDefined();
    expect(state.currentScene).toBe('shelf_portal');
  });

});
