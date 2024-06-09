import { test, expect } from '@playwright/test';

test("test", async ({ page }) => {
    await page.goto("http://uitestingplayground.com/textinput");

    //verify input is visible
    await expect(page.locator("#newButtonName")).toBeVisible();
    //selelct input and fill the input your text
    await page.locator("#newButtonName").fill("Loco");
    await page.pause();
    //click in button
    await page.locator("#updatingButton").click();
    // verify button text update
    await expect(page.locator("#updatingButton")).toContainText("Loco");
    
});

