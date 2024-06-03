const puppeteer = require("puppeteer");
const {AxePuppeteer} = require("@axe-core/puppeteer")

describe("first paint y first contentful paint",()=>{

        let browser
        let page
    
        beforeAll(async()=>{
            browser = await puppeteer.launch({
                headless:true,
                defaultViewport: null, 
                //slowMo: 500
            });

            page = await browser.newPage();
            //await page.goto("https://platzi.com", {waitUntil: "networkidle2"});

        },10000);
    
        afterAll(async ()=>{ 
            await browser.close();

        });

    test("Medir el performance del first paint y first contentful paint", async()=>{

        const navigationPromise = page.waitForNavigation();
        await page.goto("https://platzi.com");
        await navigationPromise

        const firstPaint = JSON.parse(
            await page.evaluate(()=>JSON.stringify(performance.getEntriesByName("first-paint")))
        );

        const firstContentfulPaint=JSON.parse(
            await page.evaluate(()=>JSON.stringify(performance.getEntriesByName("first-contentful-paint")))
        );
        console.log('firstPaint ', firstPaint[0].startTime)
        console.log('firstContentfulPaint ', firstContentfulPaint[0].startTime)
        

    }, 15000);

    test("Medir el performance frames por segundos", async()=>{

        const devtoolsProtocolClient = await page.target().createCDPSession();
        await devtoolsProtocolClient.send("Overlay.setShowFPSCounter",{show:true});
        await page.goto("https://platzi.com");

        await page.screenshot({path:"framesPorSegundo.jpg", type:"jpeg"})
        
    }, 15000);

})