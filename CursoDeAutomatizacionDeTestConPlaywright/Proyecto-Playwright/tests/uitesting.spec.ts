import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  await page.goto('http://uitestingplayground.com/');
  //await page.getByRole('link', { name: 'Resources' }).click();
  await page.click('a.nav-link:has-text("Resources")'); // año 2024 en junio
  //await page.locator('a.nav-link:has-text("Resources")').click();
  await page.getByRole('link', { name: 'Home' }).click();
  await page.getByRole('link', { name: 'Click' }).click();
  await page.getByRole('button', { name: 'Button That Ignores DOM Click' }).click();
});