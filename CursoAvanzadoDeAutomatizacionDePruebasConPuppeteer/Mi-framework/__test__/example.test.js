//const puppeteer = require("puppeteer");
describe("google", ()=>{

    it("abrir  el navegador", async ()=>{
        await page.goto("https://www.google.com/");
        await new Promise((resolve) => setTimeout(resolve, 5000));
    },8000);
});