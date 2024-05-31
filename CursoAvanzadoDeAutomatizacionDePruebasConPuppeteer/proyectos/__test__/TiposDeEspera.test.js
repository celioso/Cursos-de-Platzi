const puppeteer = require("puppeteer")

describe("Tipos de espera",()=>{

    it("Mostrar todos los diferentes tipos de espera", async()=>{
        const browser = await puppeteer.launch({
            headless:false,
            defaultViewport: null, 
            //slowMo: 500
        });

        const page = await browser.newPage();
        await page.goto("https://platzi.com", {waitUntil: "networkidle2"})

        //Espera explicita

        await new Promise((resolve) => setTimeout(resolve, 3000));

        //Espera por un css selector

        //await page.waitForSelector("body > main > header > div > figure > svg")

        //Espera por un xpath

        //await page.waitForXPath("/html/body/main/header/div/figure/svg/g/path[1]");

        await page.goto("https://demoqa.com/modal-dialogs", {waitUntil:"networkidle2"});

        //await page.waitForSelector("#showSmallModal",{visible: true}); // con hidden es para cuando se oculte
        
        const button = await page.waitForSelector("#showSmallModal",{visible: true});

        // Usar XPath para seleccionar el botón y asegurarse de que es visible
        //const button = await page.waitForXPath('//*[@id="showSmallModal"]',{visible: true});
        await button.click();
        
        //Espera por función

        await page.waitForFunction(()=> document.querySelector("#example-modal-sizes-title-sm").innerText === "Small Modal");
        // Ejemplo para observar el viewport

       // const observaResize = page.waitForFunction("window.innerWidth < 100");
        //await page.setViewport({width:50, height:50});

        //await observaResize;
        
        await page.click("#closeSmallModal");
        await page.waitForFunction(()=> !document.querySelector("#example-modal-sizes-title-sm"));

        await browser.close();
    }, 350000);
})