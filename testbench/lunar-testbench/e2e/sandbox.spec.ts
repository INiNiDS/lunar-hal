import { expect, test } from '@playwright/test';
import { hasExternalStack, openAvailableApp, openDesktop } from './fixtures';

test.describe('Sandbox iframe composition', () => {
  test.skip(!hasExternalStack, 'Set LUNAR_E2E_BASE_URL to run against an isolated stack.');

  test('embeds the managed web frontend rather than a duplicate renderer', async ({ page }) => {
    await openDesktop(page);
    const window = await openAvailableApp(page, 'sandbox');
    const iframe = window.locator('iframe.sandbox-frame');
    await expect(iframe).toBeVisible();
    await expect(iframe).toHaveAttribute('src', /\/editor\?embedded=sandbox&scene_id=/);
    await expect(window.getByTestId('app-window-body')).toHaveClass(/app-window-body--fill/);
  });

  test('iframe camera and selected scene star survive suspend and restore', async ({ page }) => {
    await openDesktop(page);
    const sandboxWindow = await openAvailableApp(page, 'sandbox');
    const iframe = sandboxWindow.locator('iframe.sandbox-frame');
    await expect(iframe).toBeVisible();
    const frame = page.frameLocator('iframe.sandbox-frame');
    const sceneIdFromSrc = async () => {
      const src = await iframe.getAttribute('src');
      return src ? new URL(src).searchParams.get('scene_id') : null;
    };
    const originalSceneId = await sceneIdFromSrc();
    await sandboxWindow.getByTestId('sandbox-scene-name').fill(`Sandbox E2E ${Date.now()}`);
    await sandboxWindow.getByTestId('sandbox-create-scene').click();
    await expect.poll(sceneIdFromSrc).not.toBe(originalSceneId);

    await sandboxWindow.getByTestId('sandbox-custom-star-name').fill('E2E restore sentinel');
    await sandboxWindow.getByTestId('sandbox-add-custom-star').click();
    const adminStar = sandboxWindow.getByTestId('sandbox-scene-star')
      .filter({ hasText: 'E2E restore sentinel' });
    await expect(adminStar).toBeVisible();
    const starId = await adminStar.getAttribute('data-star-id');
    expect(starId).toBeTruthy();
    const starTestId = `embedded-star-scene-${starId}`;
    const sceneStar = frame.getByTestId(starTestId);
    await expect(sceneStar).toBeVisible();
    await sceneStar.click({ force: true });
    await expect(sceneStar).toHaveAttribute('data-selected', 'true');

    const starMap = frame.getByTestId('stellar-star-map');
    const originalZoom = Number(await starMap.getAttribute('data-camera-zoom'));
    const originalOffsetX = Number(await starMap.getAttribute('data-camera-x'));
    await starMap.getByTestId('camera-zoom-in').click();
    await expect.poll(async () => Number(await starMap.getAttribute('data-camera-zoom')))
      .toBeGreaterThan(originalZoom);

    const mapBox = await starMap.boundingBox();
    expect(mapBox).not.toBeNull();
    await page.mouse.move(mapBox!.x + mapBox!.width / 2, mapBox!.y + mapBox!.height / 2);
    await page.mouse.down();
    await page.mouse.move(mapBox!.x + mapBox!.width / 2 + 35, mapBox!.y + mapBox!.height / 2 + 20);
    await page.mouse.up();
    await expect.poll(async () => Number(await starMap.getAttribute('data-camera-x')))
      .not.toBe(originalOffsetX);
    const expectedZoom = await starMap.getAttribute('data-camera-zoom');
    const expectedOffsetX = await starMap.getAttribute('data-camera-x');

    await sandboxWindow.getByTitle('Minimize').click();
    await expect(page.locator('iframe.sandbox-frame')).toHaveCount(0);

    await page.getByTestId('dock-app-sandbox').click();
    const restoredWindow = page.locator('[data-testid="webos-window"][data-app-id="sandbox"]');
    await expect(restoredWindow).toHaveCount(1);
    const restoredIframe = restoredWindow.locator('iframe.sandbox-frame');
    await expect(restoredIframe).toBeVisible();
    const restoredFrame = page.frameLocator('iframe.sandbox-frame');
    const restoredMap = restoredFrame.getByTestId('stellar-star-map');
    await expect(restoredMap).toHaveAttribute('data-camera-zoom', expectedZoom!);
    await expect(restoredMap).toHaveAttribute('data-camera-x', expectedOffsetX!);
    await expect(restoredFrame.getByTestId(starTestId!)).toHaveAttribute('data-selected', 'true');
  });

  test('fifty consecutive minimize/restore cycles do not leak iframes or DOM resources', async ({ page }) => {
    await openDesktop(page);
    await openAvailableApp(page, 'sandbox');

    for (let i = 0; i < 50; i++) {
      // Minimize
      await page.locator('[data-testid="webos-window"][data-app-id="sandbox"]').getByTitle('Minimize').click();
      await expect(page.locator('iframe.sandbox-frame')).toHaveCount(0);

      // Restore
      await page.getByTestId('dock-app-sandbox').click();
      await expect(page.locator('iframe.sandbox-frame')).toHaveCount(1);
    }

    // Verify after 50 cycles: exactly 1 iframe exists in DOM
    const totalIframes = await page.evaluate(() => document.querySelectorAll('iframe').length);
    expect(totalIframes).toBe(1);

    // Verify active window is visible and responsive
    const finalWindow = page.locator('[data-testid="webos-window"][data-app-id="sandbox"]');
    await expect(finalWindow).toBeVisible();
    await expect(finalWindow).toHaveAttribute('aria-hidden', 'false');
  });
});
