/*import { test, expect } from '@playwright/test';

test('has title', async ({ page }) => {
  await page.goto('https://playwright.dev/');

  // Expect a title "to contain" a substring.
  await expect(page).toHaveTitle(/Playwright/);
});

test('get started link', async ({ page }) => {
  await page.goto('https://playwright.dev/');

  // Click the get started link.
  await page.getByRole('link', { name: 'Get started' }).click();

  // Expects page to have a heading with the name of Installation.
  await expect(page.getByRole('heading', { name: 'Installation' })).toBeVisible();
});*/

/*import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  await page.goto('https://www.kia.com.co/');
  await page.getByRole('button', { name: 'Nuestros Modelos' }).click();
  await page.getByRole('link', { name: 'All New K3 Cross' }).click();
  await page.getByLabel('Azul').click();
});*/

import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  await page.goto('https://www.kia.com.co/');
  await page.getByRole('button', { name: 'Nuestros Modelos' }).click();
  await page.getByRole('link', { name: 'All New K3 Cross' }).click();
  await page.getByLabel('Azul').click();
  await page.getByLabel('Menú del vehículo').getByRole('link', { name: 'Especificaciones' }).click();
  await page.getByRole('button', { name: 'Nuestros Modelos' }).click();
  await page.getByRole('link', { name: 'All New Picanto' }).click();
  await page.getByLabel('Negro').click();
  await page.getByLabel('Rojo').click();
  await page.getByLabel('Plata').click();
});
