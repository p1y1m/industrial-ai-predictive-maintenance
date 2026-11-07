import {expect, test} from '@playwright/test';

test('offline dashboard is interactive and independent of the API', async ({page}) => {
  await page.route('**/api/**', (route) => route.abort());
  await page.goto('/');

  await expect(page.getByRole('heading', {name: 'Fleet intelligence'})).toBeVisible();
  await expect(page.getByText(/OFFLINE PORTFOLIO DEMO/)).toBeVisible();
  await expect(page.getByText('NASA C-MAPSS · FD001')).toBeVisible();
  await expect(page.getByText('fd001-c9c5dec1-seed42')).toBeVisible();
  await expect(page.getByText('100', {exact: true}).first()).toBeVisible();
  await expect(page.getByText('lstm', {exact: true})).toBeVisible();

  await page.getByRole('button', {name: 'elevated'}).click();
  const queueStates = page.locator('.queue .asset small').filter({hasText: 'elevated'});
  await expect(queueStates.first()).toBeVisible();
  await expect(page.locator('.queue .asset small').filter({hasText: 'critical'})).toHaveCount(0);
  await queueStates.first().locator('..').locator('..').click();
  await expect(page.getByText('ENGINE ANALYSIS', {exact: true})).toBeVisible();
  await expect(page.getByText(/predicted interval/)).toBeVisible();

  await page.screenshot({path: '../reports/screenshots/fleet-command-center.png', fullPage: true});
});

test('offline dashboard remains usable at a mobile viewport', async ({page}) => {
  await page.setViewportSize({width: 390, height: 844});
  await page.route('**/api/**', (route) => route.abort());
  await page.goto('/');

  await expect(page.getByRole('heading', {name: 'Fleet intelligence'})).toBeVisible();
  await expect(page.getByText(/OFFLINE PORTFOLIO DEMO/)).toBeVisible();
  await expect(page.getByRole('heading', {name: 'Action queue'})).toBeVisible();
  await expect(page.getByRole('heading', {name: 'Model evidence'})).toBeVisible();

  const pageWidth = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(pageWidth.scroll).toBeLessThanOrEqual(pageWidth.client);
});
