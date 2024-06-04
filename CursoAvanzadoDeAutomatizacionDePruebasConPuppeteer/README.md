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

instalar el producto PUPPETEER_PRODUCT:
`PUPPETEER_PRODUCT=firefox npm install` o `PUPPETEER_PRODUCT=firefox npm install puppeteer` o `npx puppeteer browsers install firefox` para 2024.
**Nota**: para volver a chrome se usa el siguiente codigo: `npx puppeteer browsers install chrome`.

Actualización mayo de 2024:

- Para instalar firefox para puppeteer usar: npx puppeteer browsers install firefox

- Colocar en browser lo siguiente:

```javascript
browser = await puppeteer.launch({
            headless: false,
            defaultViewport: null,
            product: 'firefox',
            protocol: 'webDriverBiDi',
        })
```

### Ejercicio de la clase

```javascript
const puppeteer = require("puppeteer")

const {getText, getCount} = require("./lib/helpers")

describe("Extrayendo informacion",()=>{

        let browser
        let page
    
        beforeAll(async()=>{
            browser = await puppeteer.launch({
                headless:true,
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
```

## Medir performance: page load

### ¿Que es y para que sirve  `page load`?

El evento **page load** se refiere al momento en que una página web y todos sus recursos dependientes (como hojas de estilo, scripts, imágenes y marcos secundarios) se han cargado completamente. Este evento es crucial tanto en el desarrollo web como en la automatización de navegadores porque marca el punto en el que el contenido de la página está totalmente disponible y listo para la interacción del usuario o de un script automatizado.

### ¿Qué es el evento `page load`?
- **Definición**: El evento page load se dispara cuando el navegador ha terminado de cargar el documento HTML inicial y todos los recursos dependientes (como imágenes, estilos y scripts).
- **Momento de Activación**: Se activa cuando la propiedad readyState del documento cambia a complete, lo cual indica que todos los recursos de la página han sido completamente cargados.

### ¿Para qué sirve el evento `page load`?
- **Sincronización de Acciones**: En la automatización de navegadores y pruebas web, esperar a que se dispare el evento page load asegura que cualquier interacción con la página (como hacer clic en un botón o extraer información) se realiza solo después de que la página esté completamente cargada.
- **Medición de Rendimiento**: Este evento es útil para medir el rendimiento de carga de una página web. Los desarrolladores pueden medir el tiempo desde que se inicia la navegación hasta que se completa la carga para optimizar el rendimiento.
- **Evitar Errores**: Al asegurarse de que la página está completamente cargada antes de realizar cualquier acción, se reduce el riesgo de errores causados por elementos que aún no están disponibles o completamente renderizados.
- **Experiencia del Usuario**: Para los desarrolladores, este evento es útil para mejorar la experiencia del usuario, garantizando que las interacciones solo ocurran cuando todo el contenido esté disponible y no mientras se está cargando.

### Ejemplo de Uso en JavaScript

Aquí hay un ejemplo simple de cómo puedes usar el evento load en JavaScript para ejecutar código solo después de que la página haya terminado de cargar:


```javascript
window.addEventListener('load', function() {
    console.log('La página y todos los recursos están completamente cargados.');
    // Aquí puedes ejecutar cualquier código que dependa de que la página esté completamente cargada
});
```


### Ejemplo de Uso en Puppeteer

En Puppeteer, un popular marco de automatización de navegadores, puedes esperar a que una página se cargue completamente antes de realizar acciones. Aquí hay un ejemplo:


```javascript
const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({ headless: false });
    const page = await browser.newPage();

    // Navegar a la página y esperar a que se cargue completamente
    await page.goto('https://example.com', { waitUntil: 'load' });

    // Realizar acciones después de que la página esté completamente cargada
    const element = await page.$('selector');
    const text = await page.evaluate(element => element.textContent, element);
    console.log(text);

    await browser.close();
})();
```

En el ejemplo de Puppeteer, el método `goto` con la opción `{ waitUntil: 'load' }` asegura que el script espera hasta que el evento `page load` se dispare antes de proceder con las acciones subsecuentes. Esto garantiza que la página está completamente lista para cualquier interacción automatizada.

### Resumen

El evento page load es un indicador crucial en el ciclo de vida de una página web que marca el momento en que todo el contenido y los recursos de la página están completamente cargados y listos para ser utilizados. Este evento es especialmente importante en el desarrollo web y la automatización de pruebas para asegurar que las interacciones y mediciones se realicen solo cuando la página esté en su estado final y completamente preparada.

codigo:

```javascript
const puppeteer = require("puppeteer");
const {AxePuppeteer} = require("@axe-core/puppeteer")

describe("Performance",()=>{

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


    /*test("Medir el performance de la automatizacion", async()=>{

        await page.waitForSelector("img");
        const metrics = await page.metrics();
        console.log(metrics);
    }, 35000);
    
    test("Medir el performance de la pagina", async()=>{

        await page.waitForSelector("img");
        const metrics2 = await page.evaluate(()=>JSON.stringify(window.performance));
        console.log(metrics2);
    }, 35000);*/

    test("Medir el performance del page load", async()=>{
        await page.tracing.start( {path: "profile.json"});
        await page.goto("https://google.com");
        await page.waitForSelector("img");
        await page.tracing.stop()
    }, 35000);

    test("Medir el performance del page load con screenshorts", async()=>{
        await page.tracing.start( {path: "profile.json", screenshots:true});
        await page.goto("https://platzi.com");
        await page.waitForSelector("img");
        await page.tracing.stop()
    }, 35000);

    test("Medir el performance del page load con screenshorts y extrayendolos", async()=>{
        const fs = require('fs')

        await page.tracing.start( {path: "profile.json", screenshots:true});
        await page.goto("https://platzi.com");
        await page.waitForSelector("img");
        await page.tracing.stop()
        const tracing = JSON.parse(fs.readFileSync("./profile.json", "utf8"))
        //Filtrar el JSON
        const traceScreenShots = tracing.traceEvents.filter(
            (x)=>
            x.cat === 'disabled-by-default-devtools.screenshot' &&
            x.name === 'Screenshot' &&
            typeof x.args !== 'undefined' &&
            typeof x.args.snapshot !== 'undefined'
        );

        //Iterar sobre este arreglo para crear la simagenes
        traceScreenShots.forEach(function(snap, index){
            fs.writeFile(`trace-screenshot-${index}.png`, snap.args.snapshot, 'base64', function(err){
                if (err) {
                    console.log('No pude crear el archivo', err)
                };
            });           
        });

    }, 35000);
})
```

## Medir performance: first contentful paint

```javascript
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
```

[lighthouse/puppeteer.md at master · GoogleChrome/lighthouse · GitHub](https://github.com/GoogleChrome/lighthouse/blob/main/docs/puppeteer.md "lighthouse/puppeteer.md at master · GoogleChrome/lighthouse · GitHub")

## Inicializando nuestro framework

//Pasos para arracar el proyecto del framework de jest con puppeteer

1. npm init -y

2. git init

3. npm i puppeteer jest jest-puppeteer @types/jest babel-jest @babel/core @babel/preset-env

4. Crear .gitignore

5. `npx jest --init` o `node node_modules.bin\jest --init` o `node node_modules.bin\jest --init` //--> Esto inicializa la config de jest (Windows 10) o la que me funciono fue `npx jest --init`

6. Respuestas:
- √ Would you like to use Jest when running "test" script in "package.json"? ... yes
- √ Would you like to use Typescript for the configuration file? ... no
- √ Choose the test environment that will be used for testing » node
- √ Do you want Jest to add coverage reports? ... no
- √ Which provider should be used to instrument code for coverage? » babel
- √ Automatically clear mock calls, instances, contexts and results before every test? ... no   
✏️  Modified C:\Users\celio\OneDrive\Escritorio\programación\platzi\CursoAvanzadoDeAutomatizacionDePruebasConPuppeteer\Mi-framework\package.json

📝  Configuration file created at C:\Users\celio\OneDrive\Escritorio\programación\platzi\CursoAvanzadoDeAutomatizacionDePruebasConPuppeteer\Mi-framework\jest.config.js

7. En jest.config.js, descomentar y colocar {bail: 5, preset: "jest-puppeteer"}

```javascript
/** @type {import('jest').Config} */
const config = {
  // All imported modules in your tests should be mocked automatically
  // automock: false,

  // Stop running tests after `n` failures
  bail: 5,
```

```javascript
  // A preset that is used as a base for Jest's configuration
  preset: "jest-puppeteer",
```

8. Crear archivo jest-puppeteer.config.js y pegarle dentro: 
```javascript
module.exports = {
    launch: {
        headless: false,
        slowMo:100,
    },
    browserContext:"default"
}
```
9. Intalar dependencia de desarrollo:
`npm i -D prettier`

10. se crea el archivo `.prettierrc` y se agrega el siguiente codigo:

```javascript
{
    "printWidth": 100,
    "singleQuote": true,
    "useTabs": true,
    "tabWidth": 2,
    "semi": false,
    "trailingComma": "es5",
    "bracketSameLine": true    
}
```

11. Se instala la dependencia de babel:
`npm i @babel/core @babel/preset-env babel-jest`

12. Crear archivo de babel.config.js y pegarle dentro: 

```javascript
module.exports = { 
    presets: [ 
        [ 
            '@babel/preset-env', 
            { 
                targets: { 
                    node:'current', 
                }, 
            }, 
        ], 
    ], 
};
```

10. Crear carpeta de pruebas **__test__** y una prueba:

11. Crear una prueba para probar que todo funcione correctamente:

```javascript
describe("google", ()=>{

    it("abrir  el navegador", async ()=>{
        await page.goto("https://www.google.com/");
        await new Promise((resolve) => setTimeout(resolve, 5000));
    },8000);
});
```

## Creando la Base Page

Se crea la carpeta page y luego el archivo BasePage.js con el siguiente codigo:

```javascript
export default class BasePage {

    async getTitle(){

        return await page.title();
    }

    async getUrl(){

        return await page.url();
    }

    async getText(selector){

        try {
            await page.waitForSelector(selector);
            return await page.$eval(selector, (el)=el.textContent);
        }
        catch (e){
            throw new Error("Error al obtener el texto del selector ${selector}")

        }
       
    }

    async getAttribute(selector, attribute){

        try {
            await page.waitForSelector(selector);
            return await page.$eval(selector, (el)=el.getAttribute(attribute));
        }
        catch (e){
            throw new Error("Error al obtener el atributo del selector ${selector}");

        }
       
    }

    async getValue(selector){

        try {
            await page.waitForSelector(selector);
            return await page.$eval(selector, (el)=el.value);
        }
        catch (e){
            throw new Error("Error al obtener el valor del selector ${selector}");

        }
       
    }

    async getCount(selector){

        try {
            await page.waitForSelector(selector);
            return await page.$$eval(selector, (el)=el.length);
        }
        catch (e){
            throw new Error("Error al obtener el numero de elementos del selector ${selector}");

        }
       
    }

    async click(selector){

        try {
            await page.waitForSelector(selector);
            await page.click(selector);
        }
        catch (e){
            throw new Error("Error al dar click al selector ${selector}");

        }
       
    }

    async type(selector, text, opts={}){

        try {
            await page.waitForSelector(selector);
            await page.type(selector, text, opts);
        }
        catch (e){
            throw new Error("Error escribir en el selector ${selector}");

        }
       
    }

    async doubleClick(selector){

        try {
            await page.waitForSelector(selector);
            await page.click(selector, {clickCount:2});
        }
        catch (e){
            throw new Error("Error escribir en el selector ${selector}");

        }
       
    }

    async wait(time){

        return await new Promise((resolve) => setTimeout(resolve, time));
    }


}
```

[Using with puppeteer · Jest](https://jestjs.io/docs/puppeteer "Using with puppeteer · Jest")

## Page Object Model


**Lecturas recomendadas**

[Xpath cheatsheet](https://devhints.io/xpath "Xpath cheatsheet")

[Curso de Web Scraping con Python y Xpath - Platzi](https://platzi.com/clases/web-scraping/ "Curso de Web Scraping con Python y Xpath - Platzi")

[Space & Beyond | Testim.io demo](https://demo.testim.io/ "Space & Beyond | Testim.io demo")