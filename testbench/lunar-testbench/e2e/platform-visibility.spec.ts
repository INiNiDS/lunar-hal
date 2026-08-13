import { expect, test } from '@playwright/test';
import { hasExternalStack, openDesktop } from './fixtures';

test.describe('platform-aware Sandbox visibility', () => {
  test.skip(!hasExternalStack, 'Set LUNAR_E2E_BASE_URL to run against an isolated stack.');

  test('does not offer Sandbox for a non-web frontend runtime', async ({ page, request }) => {
    await openDesktop(page);
    const startBackend = process.env.LUNAR_E2E_START_BACKEND_URL ?? 'http://127.0.0.1:16181';
    const response = await request.get(`${startBackend}/services`);
    test.skip(!response.ok(), 'Start backend service endpoint is unavailable to this test stack.');
    const services = await response.json() as Array<{ name: string; status: { kind?: string } | string; platform?: string; public_url?: string | null }>;
    const frontend = services.find((service) => service.name === 'frontend');
    const sandbox = page.getByTestId('desktop-app-sandbox');
    const frontendIsEmbeddable = frontend?.platform === 'web' && Boolean(frontend.public_url);

    if (frontendIsEmbeddable) {
      await expect(sandbox).toBeVisible();
    } else {
      await expect(sandbox).toHaveCount(0);
    }
  });
});
