import { expect, test } from '@playwright/test';
import { hasExternalStack, openAvailableApp, openDesktop } from './fixtures';

test.describe('WebOS responsive client-area behavior', () => {
  test.skip(!hasExternalStack, 'Set LUNAR_E2E_BASE_URL to run against an isolated stack.');

  test('Models keeps its content inside a compact floating window', async ({ page }) => {
    await openDesktop(page);
    const window = await openAvailableApp(page, 'models');
    await expect(window).toBeVisible();

    await window.evaluate((element) => {
      Object.assign((element as HTMLElement).style, {
        width: '480px', height: '640px', left: '8px', top: '8px',
      });
    });
    const body = window.getByTestId('app-window-body');
    await expect(body).toBeVisible();
    await expect(body).toEvaluate((element) => element.scrollWidth <= element.clientWidth);
  });

  test('Sandbox uses Fill mode without a nested page scrollbar', async ({ page }) => {
    await openDesktop(page);
    const window = await openAvailableApp(page, 'sandbox');
    const body = window.getByTestId('app-window-body');
    await expect(body).toHaveClass(/app-window-body--fill/);
    await expect(window.locator('.sandbox-app')).toBeVisible();
    await expect(body).toEvaluate((element) => getComputedStyle(element).overflow === 'hidden');
  });
});
