import { test, expect } from '@playwright/test';
const { chromium } = require('playwright');

test("añadir producto al carrito", async () => {
    const browser = await chromium.launch({ headless: false });
    const page = await browser.newPage();
// ir a la url https://automationexercise.com/products
    await page.goto("https://automationexercise.com/products");
//hover del primer producto que encontremos
    await page.hover("body > section:nth-child(3) > div > div > div.col-sm-9.padding-right > div > div:nth-child(3)");
//click en el primer producto ver mas detalles
    await page.click("body > section:nth-child(3) > div > div > div.col-sm-9.padding-right > div > div:nth-child(3) a:has-text('View Product')");
    await expect(page).toHaveURL("https://automationexercise.com/product_details/1");
//click en el boton + (dos veces)
    await page.locator("#quantity").fill("3");
//seleccionar en el menu dropdown un nuevo tamaño
    //await page.locator('#group_1').selectOption({ index: 1 });
//click en boton añadir al carrito
    await page.click('body > section > div > div > div.col-sm-9.padding-right > div.product-details > div.col-sm-7 > div > span > button')
// verificar (expect) "Added!"
    await expect(page.locator('#cartModal > div > div > div.modal-header > h4')).toBeVisible();
    await expect(page.locator('#cartModal > div > div > div.modal-header > h4')).toContainText("Added!");
//click en boton continue shopping
    await page.click('#cartModal > div > div > div.modal-footer > button')
//el modal debe no ser visible
    await page.click('#header > div > div > div > div.col-sm-8 > div > ul > li:nth-child(1) > a');
    await browser.close();
});

