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

## Modo incógnito del navegador
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

        //page = await browser.newPage();
        //await page.goto("https://platzi.com", {waitUntil: "networkidle2"});

        //para abtrir el navegador en modo incognito
        page = await (await browser.createBrowserContext()).newPage(); //para acortar el codigo
       // const context = await browser.createBrowserContext();
        //page = await context.newPage();
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

## Visual Testing

instalar la librria para snapshot:

`npm i --save-dev jest-image-snapshot --legacy-peer-deps`

```javascript
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


    /*it("Snapshop de toda la pagina", async()=>{

        await page.waitForSelector("img");

        const screenshot = await page.screenshot();

        expect(screenshot).toMatchImageSnapshot();
        
    }, 35000);*/  

    /*it("Snapshop de solo un elemento", async()=>{

        const image = await page.waitForSelector("img");

        const screenshot = await image.screenshot();

        expect(screenshot).toMatchImageSnapshot({
            failureThreshold:0.0,
            failureThresholdType: "percent"

        });
        
    }, 35000); */

    /*it("Snapshop de un celular", async()=>{

        const tablet = KnownDevices['iPad Pro'];
        await page.emulate(tablet);
        
        await page.waitForSelector("img");

        const screenshot = await page.screenshot();

        expect(screenshot).toMatchImageSnapshot({
            failureThreshold:0.05,
            failureThresholdType: "percent"

        });
 
    }, 35000); */

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
```

## Generando PDFs

```javascript
const puppeteer = require("puppeteer");

describe("Craecion de PDF",()=>{
    let browser;
    let page;
    beforeAll(async()=>{
        browser = await puppeteer.launch({
            headless:true,
            defaultViewport: null, 
            //slowMo: 500
        });

        page = await (await browser.createBrowserContext()).newPage();
        await page.goto("https://google.com", {waitUntil: "networkidle2"});

    },35000);

    afterAll(async ()=>{ 
        await browser.close();
        return pdf
    })

    test("PDF de pantalla completa", async()=>{

        let pdfCSS = []
        pdfCSS.push('<style>');
        pdfCSS.push('h1{ font-size:10px; margin-left:30px;}');
        pdfCSS.push("</style>");

        const css = pdfCSS.join(" ");

        await page.pdf({
            path: "google.pdf",
            format: "A4",
            printBackground: true, 
            displayHeaderFooter: true,
            headerTemplate: css + "<h1>" + "Mira mi primer PDF con puppeteer" + "</h1>",
            footerTemplate: css + '<h1> page <span class="pageNumber"></span> of <span class="totalPages"></span></h1>',
            margin: {
                top: "100px",
                bottom: "200px",
                right: "30px",
                left: "30px",
            }

        });
        
    },35000);

    test("PDF de pantalla completa en modo landscape", async()=>{

        let pdfCSS = []
        pdfCSS.push('<style>');
        pdfCSS.push('h1{ font-size:10px; margin-left:30px;}');
        pdfCSS.push("</style>");

        const css = pdfCSS.join(" ");

        await page.pdf({
            path: "googleLandscape.pdf",
            format: "A4",
          
            headerTemplate: css + "<h1>" + "Mira mi primer PDF con puppeteer" + "</h1>",
            footerTemplate: css + '<h1> page <span class="pageNumber"></span> of <span class="totalPages"></span></h1>',
            margin: {
                top: "100px",
                bottom: "200px",
                right: "30px",
                left: "30px",
            },
            landscape: true

        });
        
    },35000);

});
```

## Geolocalización

```javascript
const puppeteer = require("puppeteer");

describe("Geolocalizacion",()=>{

        let browser
        let page
    
        beforeAll(async()=>{
            browser = await puppeteer.launch({
                headless:false,
                defaultViewport: null, 
                //slowMo: 500
            });

            page = await browser.newPage();
            //await page.goto("https://platzi.com", {waitUntil: "networkidle2"});

        },10000);
    
        afterAll(async ()=>{ 
            await browser.close();

        });


    it("Cambio de la geolocalizacion", async()=>{
        const context = browser.defaultBrowserContext();

        await context.overridePermissions("https://chercher.tech/practice/geo-location.html", ['geolocation']);

        await page.setGeolocation({latitude:90, longitude: 20});

        await page.goto("https://chercher.tech/practice/geo-location.html");

        await new Promise((resolve) => setTimeout(resolve, 5000));
        await page.setGeolocation({latitude:90, longitude: 0});
        await page.goto("https://chercher.tech/practice/geo-location.html");
        await new Promise((resolve) => setTimeout(resolve, 5000));

    }, 35000);
})
```

## Probando accesibilidad

[WAI-ARIA Overview | Web Accessibility Initiative (WAI) | W3C](https://www.w3.org/WAI/standards-guidelines/aria/ "WAI-ARIA Overview | Web Accessibility Initiative (WAI) | W3C")

[axe-puppeteer - npm](https://www.npmjs.com/package/axe-puppeteer "axe-puppeteer - npm")

instalar paquete @axe-core:

`npm install @axe-core/puppeteer --legacy-peer-deps`

```javascript
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
```

## Puppeteer con Firefox