import { expect, test } from '@playwright/test';
import { hasExternalStack, openAvailableApp, openDesktop } from './fixtures';

test.describe('Gallery responsive shell', () => {
  test.skip(!hasExternalStack, 'Set LUNAR_E2E_BASE_URL to run against an isolated stack.');

  test('uses the container-query gallery layout inside its window', async ({ page }) => {
    await openDesktop(page);
    const window = await openAvailableApp(page, 'siren_gallery');
    const layout = window.locator('.gallery-layout');
    await expect(layout).toBeVisible();
    await expect(window.getByTestId('app-window-body')).toEvaluate(
      (element) => element.scrollWidth <= element.clientWidth,
    );
  });
});
