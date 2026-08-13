import { expect, test } from '@playwright/test';
import { hasExternalStack, openAvailableApp, openDesktop } from './fixtures';

test.describe('WebOS window lifecycle', () => {
  test.skip(!hasExternalStack, 'Set LUNAR_E2E_BASE_URL to run against an isolated stack.');

  test('minimize preserves one window instance and both launchers restore it', async ({ page }) => {
    await openDesktop(page);
    const appWindow = await openAvailableApp(page, 'models');
    const windows = page.locator('[data-testid="webos-window"][data-app-id="models"]');
    await expect(windows).toHaveCount(1);

    await appWindow.getByTitle('Minimize').click();
    await expect(appWindow).toHaveAttribute('aria-hidden', 'true');
    await page.getByTestId('dock-app-models').click();
    await expect(appWindow).toHaveAttribute('aria-hidden', 'false');

    await appWindow.getByTitle('Minimize').click();
    await expect(appWindow).toHaveAttribute('aria-hidden', 'true');
    await page.getByTestId('desktop-app-models').click();
    await expect(appWindow).toHaveAttribute('aria-hidden', 'false');
    await expect(windows).toHaveCount(1);
  });

  test('losing browser focus clears the drag overlay', async ({ page }) => {
    await openDesktop(page);
    const appWindow = await openAvailableApp(page, 'models');
    const titlebar = appWindow.getByTestId('window-titlebar');
    const box = await titlebar.boundingBox();
    expect(box).not.toBeNull();

    await page.mouse.move(box!.x + box!.width / 2, box!.y + box!.height / 2);
    await page.mouse.down();
    await expect(page.getByTestId('window-drag-overlay')).toBeVisible();

    await page.evaluate(() => globalThis.dispatchEvent(new Event('blur')));
    await expect(page.getByTestId('window-drag-overlay')).toHaveCount(0);
    await page.mouse.up();
  });
});
