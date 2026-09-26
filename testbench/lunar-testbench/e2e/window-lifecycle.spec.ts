import { expect, test } from '@playwright/test';
import { hasExternalStack, openAvailableApp, openDesktop } from './fixtures';

test.describe('Lunar-OS window lifecycle', () => {
  test.skip(!hasExternalStack, 'Set LUNAR_E2E_BASE_URL to run against an isolated stack.');

  test('minimize unmounts window from DOM and dock/desktop launchers restore it from RAM', async ({ page }) => {
    await openDesktop(page);
    const appWindow = await openAvailableApp(page, 'models');
    const windows = page.locator('[data-testid="webos-window"][data-app-id="models"]');
    await expect(windows).toHaveCount(1);

    await appWindow.getByTitle('Minimize').click();
    // Minimized window is completely unmounted from the DOM
    await expect(windows).toHaveCount(0);

    // Dock launcher restores the window from RAM
    await page.getByTestId('dock-app-models').click();
    await expect(windows).toHaveCount(1);
    await expect(windows).toHaveAttribute('aria-hidden', 'false');

    await windows.getByTitle('Minimize').click();
    await expect(windows).toHaveCount(0);

    // Desktop launcher restores the window from RAM
    await page.getByTestId('desktop-app-models').click();
    await expect(windows).toHaveCount(1);
    await expect(windows).toHaveAttribute('aria-hidden', 'false');
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

  test('form values survive restore and closing one app leaves the other window intact', async ({ page }) => {
    await openDesktop(page);
    const modelsWin = await openAvailableApp(page, 'models');
    const xPc = modelsWin.getByRole('spinbutton', { name: 'x_pc' });
    await xPc.fill('42.5');

    const trainingWin = await openAvailableApp(page, 'training');
    const dataPath = trainingWin.getByRole('textbox', { name: 'Data path' });
    await dataPath.fill('ai_data/e2e-training.parquet');
    await expect(modelsWin).toBeVisible();
    await expect(trainingWin).toBeVisible();

    await modelsWin.getByTitle('Minimize').click();
    const modelsLocator = page.locator('[data-testid="webos-window"][data-app-id="models"]');
    await expect(modelsLocator).toHaveCount(0);
    await expect(trainingWin).toBeVisible();

    await page.getByTestId('dock-app-models').click();
    const restoredModels = modelsLocator;
    await expect(restoredModels).toHaveCount(1);
    await expect(restoredModels.getByRole('spinbutton', { name: 'x_pc' })).toHaveValue('42.5');
    await expect(trainingWin.getByRole('textbox', { name: 'Data path' }))
      .toHaveValue('ai_data/e2e-training.parquet');

    await trainingWin.getByTitle('Close').click();
    await expect(page.locator('[data-testid="webos-window"][data-app-id="training"]')).toHaveCount(0);
    await expect(restoredModels).toHaveCount(1);
    await expect(restoredModels.getByRole('spinbutton', { name: 'x_pc' })).toHaveValue('42.5');
  });

});
