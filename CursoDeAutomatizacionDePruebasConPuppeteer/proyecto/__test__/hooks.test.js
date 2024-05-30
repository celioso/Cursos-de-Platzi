const puppeteer = require("puppeteer")

describe("Extrayendo informacion",()=>{

    let browser
    let page

    beforeAll(async()=>{ //beforeAll lo inicia antes de las pruebas y beforeEach lo abre en cada prueba
        browser = await puppeteer.launch({
            headless:false,
            defaultViewport: null, 
            //slowMo: 500
        });

        page = await browser.newPage();
        await page.goto("https://platzi.com", {waitUntil: "networkidle2"});
    });

    afterAll(async ()=>{ //afterAll lo cierra despuesde las pruebas yafterEach lo ciarra al terminar cada prueba
        await browser.close();
    })

    it("Extraer el titulo de la pagina y url", async()=>{

        const titulo = await page.title();
        const url = await page.url();

        console.log("titulo", titulo);
        console.log("url", url);

        

        //await new Promise((resolve) => setTimeout(resolve, 3000));

    },35000);

    it("Extraer la información d eun elemento", async()=>{

        await page.waitForSelector("body > main > header > div > nav > ul > li:nth-child(4) > a");

        const nombreBoton = await page.$eval("body > main > header > div > nav > ul > li:nth-child(4) > a", (button) => button.textContent);

        const [button] = await page.$x("/html/body/main/section[1]/a/div/div[2]/div[2]/button");
        const propiedad = await button.getProperty("textContent");
        const texto = await propiedad.jsonValue();

        //console.log("texto", texto);

        //Segunda Forma

        const texto2 = await page.evaluate((name)=>name.testContent, button);

        const button3 = await page.waitForXPath("/html/body/main/section[1]/a/div/div[2]/div[2]/button");
        const texto3 = await page.evaluate((name)=>name.testContent, button3);
        console.log("texto3", texto3);

    },35000);  

    it("Contar los elementos d euna pagina", async()=>{
        
        const images = await page.$$eval("img",(imagenes) => imagenes.length);
        console.log("images", images)

    },35000);

},50000);