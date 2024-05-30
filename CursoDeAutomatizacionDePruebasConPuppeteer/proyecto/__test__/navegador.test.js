const puppeteer = require("puppeteer")

describe('Mi primer test en puppeteer',()=>{

    it("Debe de abrir y cerrar el navegador", async()=>{
        const browser = await puppeteer.launch({
            headless:false,
            slowMo: 0, // coloca en camara lenta el proceso
            devtools: true, //Abre las herramientas de desarrollador
            //defaultViewport:{
            //    width:2100,
            //    height:1080
            //}

            //args:["--window-size=1920,1080"], //tamaño d ela pantalla
            defaultViewport: null, // colocaa al tamaño d ela ventana
        });
        const page = await browser.newPage();
        await page.goto("https://www.google.com");
        await new Promise((resolve) => setTimeout(resolve, 5000));
        await browser.close();
    }, 30000);
})