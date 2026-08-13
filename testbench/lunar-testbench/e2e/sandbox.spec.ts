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
});
