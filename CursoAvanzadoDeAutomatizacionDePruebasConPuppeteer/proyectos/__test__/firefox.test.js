const puppeteer = require("puppeteer")

const {getText, getCount} = require("./lib/helpers")

describe("Extrayendo informacion",()=>{

        let browser
        let page
    
        beforeAll(async()=>{
            browser = await puppeteer.launch({
                headless:false,
                product: "firefox",
                defaultViewport: null, 
                protocol: 'webDriverBiDi',
                //slowMo: 500
            });

            page = await browser.newPage();
            await page.goto("https://platzi.com");

        },10000);
    
        afterAll(async ()=>{ 
            await browser.close();

        },10000);

    it("Extraer la información de un elemento", async()=>{
        
        await page.waitForSelector("body > main > header > div > nav > ul > li:nth-child(4) > a");

        const nombreBoton = await getText(page, "body > main > header > div > nav > ul > li:nth-child(4) > a");
        console.log("nombreBoton", nombreBoton)


    }, 35000);  

    it("Contar los elementos de una pagina", async()=>{
        
        const images = await getCount(page, "img");
        console.log("images", images)

    }, 35000);

    it("Extraer el titulo de la pagina y url", async()=>{
        
        const titulo = await page.title();
        const url = await page.url();

        console.log("titulo", titulo);
        console.log("url", url);

        

        //await new Promise((resolve) => setTimeout(resolve, 3000));

    }, 35000);

})