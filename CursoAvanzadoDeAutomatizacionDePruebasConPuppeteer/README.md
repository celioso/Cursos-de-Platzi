# Curso Avanzado de Automatización de Pruebas con Puppeteer

## Emulación de dispositivos

```javascript
const puppeteer = require("puppeteer")
const { KnownDevices } = require('puppeteer');

describe("Emulando informacion",()=>{
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
    })

    test("Emulando dispositivos de forma manual", async()=>{
        await page.emulate({
            name: "Mi dispositivo",
            viewport: {
                width:375,
                height: 667,
                deviceScalaFactor:2,
                isMobile: true,
                hasTouch: true,
                isLandscape:false

            },
            userAgent: "Mozilla/5.0 (Linux; Android 10; SAMSUNG SM-J600G) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/10.1 Chrome/71.0.3578.99 Mobile Safari/537.36",
        });

        await new Promise((resolve) => setTimeout(resolve, 5000));

    },35000);

    test("Emulando sitio de escritorio", async()=>{

        await page.setViewport({
            width:1280,
            height:800
        })
        await new Promise((resolve) => setTimeout(resolve, 5000));

    },35000);

    test("Emulando sitio en una tablet", async()=>{
        //const { KnownDevices } = require('puppeteer');
        const tablet = KnownDevices['iPad Pro'];
        await page.emulate(tablet);

        await new Promise((resolve) => setTimeout(resolve, 5000));

    },35000);

    test("Emulando sitio en una tablet en modo landscape", async()=>{

        //const { KnownDevices } = require('puppeteer');
        const tablet = KnownDevices["iPad landscape"];
        await page.emulate(tablet);

        await new Promise((resolve) => setTimeout(resolve, 5000));

    },35000);

    test("Emulando sitio en un celular", async()=>{

        //const { KnownDevices } = require('puppeteer');
        const iPhone = KnownDevices['iPhone X']
        await page.emulate(iPhone);
        await new Promise((resolve) => setTimeout(resolve, 5000));

    },35000);

});
```

Si quieren el link de los dispositivos es el siguiente

[KnownDevices variable | Puppeteer (pptr.dev)](https://pptr.dev/api/puppeteer.knowndevices/ "KnownDevices variable | Puppeteer (pptr.dev)")