const puppeteer = require("puppeteer")
const { KnownDevices } = require('puppeteer');

describe("Capturas de pantalla",()=>{
    let browser
    let page

    beforeAll(async()=>{
        browser = await puppeteer.launch({
            headless:false,
            defaultViewport: null, 
            //slowMo: 500
        });

        page = await (await browser.createBrowserContext()).newPage();
        await page.goto("https://google.com", {waitUntil: "networkidle2"});

    },10000);

    afterAll(async ()=>{ 
        await browser.close();
    })

    test("Captura de pantalla completa", async()=>{

        await page.screenshot({
            path:"./capturaDePantalla.png",
            fullPage:true
        })


    },35000);

    test("Captura de pantalla completa seleccionando un area", async()=>{

        await page.screenshot({
            path:"./CapturaDePantallaSeleccionandoUnArea.png",
            clip:{
                x:0,
                y:0,
                width:500,
                height:500
            }
        })


    },35000);

    test("Captura de pantalla con con fondo transparente", async()=>{
        await page.evaluate(() => (document.body.style.background = "transparent"))
        await page.screenshot({
            path:"./CapturaDePantallatransparente.png",
            omitBackground: true
        })


    },35000);

    test("Captura de pantalla a un elemento", async()=>{
        const elemento = await page.waitForSelector("body > div.L3eUgb > div.o3j99.LLD4me.yr19Zb.LS8OJ > div > img")
        await elemento.screenshot({
            path:"./CapturaDePantalladeunelemento.png",

        })


    },35000);

});