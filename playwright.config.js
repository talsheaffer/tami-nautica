// playwright.config.js
// Playwright configuration for Tamar "Tami" Sheaffer 70th Birthday 2.5D Spatial Prototype
// Configures hardware WebGL emulation via SwiftShader/ANGLE for reliable headless execution.

import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './prototype/tests',
  timeout: 30000,
  expect: { timeout: 5000 },
  fullyParallel: false,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: process.env.TEST_URL || 'http://127.0.0.1:7070',
    trace: 'on-first-retry',
    launchOptions: {
      args: [
        '--use-gl=angle',
        '--use-angle=swiftshader',
        '--enable-webgl',
        '--ignore-gpu-blocklist',
        '--no-sandbox',
        '--disable-gpu-sandbox'
      ]
    }
  },
  projects: [
    {
      name: 'Desktop-Chrome',
      use: { 
        viewport: { width: 1440, height: 900 },
        deviceScaleFactor: 1.0
      }
    },
    {
      name: 'Mobile-Safari-iPhone14',
      use: {
        ...devices['iPhone 14'],
        viewport: { width: 390, height: 844 },
        hasTouch: true,
        isMobile: true
      }
    }
  ]
});
