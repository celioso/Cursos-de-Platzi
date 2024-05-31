const puppeteer = require("puppeteer")

describe("tomeout",()=>{
    jest.setTimeout(10000);
    it("uso de timeout", async()=>{
        
        const browser = await puppeteer.launch({
            headless:false,
            defaultViewport: null, 
            //slowMo: 500
        });

        const page = await browser.newPage();
        page.setDefaultTimeout(10000);
        page.setDefaultNavigationTimeout(10000);
        await page.goto("https://platzi.com", {waitUntil: "networkidle2"});
        
        const images = await page.$$eval("img",(imagenes) => imagenes.length,{
            timeout:30000
        });
        console.log("images", images)

        await browser.close();
    });

})