const puppeteer = require("puppeteer");
const {AxePuppeteer} = require("@axe-core/puppeteer")

describe("Accesibilidad",()=>{

        let browser
        let page
    
        beforeAll(async()=>{
            browser = await puppeteer.launch({
                headless:true,
                defaultViewport: null, 
                //slowMo: 500
            });

            page = await browser.newPage();
            await page.goto("https://platzi.com", {waitUntil: "networkidle2"});

        },10000);
    
        afterAll(async ()=>{ 
            await browser.close();

        });


    it("Probar accesibilidad", async()=>{

        await page.waitForSelector("img");
        const snapshot = await page.accessibility.snapshot();
        console.log(snapshot);

    }, 35000);

    // con la libreria @axe-core/puppeteer
    it("Probar accesibilidad con axe", async()=>{

        await page.setBypassCSP(true)
        await page.waitForSelector("img");
        
        const result = await new AxePuppeteer(page).analyze()
        console.log(result.violations) //console.log(result.violations[0].nodes[0] para espesificar el nodo que deseo ver

    }, 35000);
})