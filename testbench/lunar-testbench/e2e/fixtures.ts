import { expect, test, type Page } from '@playwright/test';

export const hasExternalStack = Boolean(process.env.LUNAR_E2E_BASE_URL);

export async function openDesktop(page: Page) {
  await page.goto('/');
  await expect(page.locator('.desktop-surface')).toBeVisible();
}

export async function openAvailableApp(page: Page, appId: string) {
  const launcher = page.getByTestId(`desktop-app-${appId}`);
  await expect(launcher).toBeVisible();
  test.skip(!(await launcher.isEnabled()), `${appId} requires its managed service to be running`);
  await launcher.click();
  const window = page.locator(`[data-testid="webos-window"][data-app-id="${appId}"]`);
  await expect(window).toBeVisible();
  return window;
}
