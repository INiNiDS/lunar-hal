import { expect, test } from '@playwright/test';
import { hasExternalStack, openAvailableApp, openDesktop } from './fixtures';

test.describe('WebOS visual regression', () => {
  test.skip(!hasExternalStack, 'Set LUNAR_E2E_BASE_URL to run against an isolated stack.');

  test('compact Models window', async ({ page }) => {
    await openDesktop(page);
    const window = await openAvailableApp(page, 'models');
    await window.evaluate((element) => {
      Object.assign((element as HTMLElement).style, {
        width: '480px', height: '640px', left: '8px', top: '8px',
      });
    });
    await expect(window).toHaveScreenshot('models-480x640.png', {
      animations: 'disabled',
      caret: 'hide',
      maxDiffPixelRatio: 0.02,
    });
  });

  test('Fill-mode Sandbox window', async ({ page }) => {
    await openDesktop(page);
    const window = await openAvailableApp(page, 'sandbox');
    await expect(window).toHaveScreenshot('sandbox-fill.png', {
      animations: 'disabled',
      caret: 'hide',
      maxDiffPixelRatio: 0.02,
    });
  });
});
