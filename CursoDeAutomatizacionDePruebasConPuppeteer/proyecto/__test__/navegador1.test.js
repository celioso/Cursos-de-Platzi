const puppeteer = require("puppeteer")

describe('Mi primer test en puppeteer',()=>{

    it("Debe de abrir y cerrar el navegador", async()=>{
        const browser = await puppeteer.launch({
            headless:false,
            defaultViewport: null, 
        });
        const page = await browser.newPage();
        await page.goto("https://www.github.com");
        //await new Promise((resolve) => setTimeout(resolve, 5000));
        await page.waitForSelector("img");
        //Recargar la pagina
        await page.reload();
        await page.waitForSelector("img");

        //Navegar a otro sitio
        await page.goto("https://www.platzi.com");
        await page.waitForSelector('body > main > header > div > figure > svg > g > path:nth-child(2)');

        await page.goBack();
        await page.goForward();
        //await page.waitForSelector("img");

        // Abrir otra pagina
        const page2 = await browser.newPage()
        await page2.goto("https://www.google.com") 

        await browser.close();
    }, 30000);
})