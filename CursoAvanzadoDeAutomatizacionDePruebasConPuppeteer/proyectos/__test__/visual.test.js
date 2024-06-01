const puppeteer = require("puppeteer");
const {toMatchImageSnapshot} = require("jest-image-snapshot");
expect.extend({toMatchImageSnapshot});
const { KnownDevices } = require('puppeteer');

describe("Visual test",()=>{

        let browser
        let page
    
        beforeAll(async()=>{
            browser = await puppeteer.launch({
                headless:false,
                defaultViewport: null, 
                //slowMo: 500
            });

            page = await browser.newPage();
            await page.goto("https://platzi.com", {waitUntil: "networkidle2"});

        },10000);
    
        afterAll(async ()=>{ 
            await browser.close();

        });


    it("Snapshop de toda la pagina", async()=>{

        await page.waitForSelector("img");

        const screenshot = await page.screenshot();

        expect(screenshot).toMatchImageSnapshot();
        
    }, 35000);

    it("Snapshop de solo un elemento", async()=>{

        const image = await page.waitForSelector("img");

        const screenshot = await image.screenshot();

        expect(screenshot).toMatchImageSnapshot({
            failureThreshold:0.0,
            failureThresholdType: "percent"

        });
        
    }, 35000); 

    it("Snapshop de un celular", async()=>{

        const tablet = KnownDevices['iPad Pro'];
        await page.emulate(tablet);
        
        await page.waitForSelector("img");

        const screenshot = await page.screenshot();

        expect(screenshot).toMatchImageSnapshot({
            failureThreshold:0.05,
            failureThresholdType: "percent"

        });
 
    }, 35000);

    it("Remover imagen antes de crear snapshot", async()=>{

        await page.waitForSelector("img");

        //await page.evaluate(() => (document.querySelectorAll("img") || []).forEach((img) => img.remove()));
        
        const screenshot = await page.screenshot();

        expect(screenshot).toMatchImageSnapshot({
            failureThreshold:0.05,
            failureThresholdType: "percent"

        });
 
    }, 35000); 
})