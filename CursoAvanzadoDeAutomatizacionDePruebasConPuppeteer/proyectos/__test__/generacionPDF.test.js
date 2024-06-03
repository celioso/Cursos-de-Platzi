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
    });

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
            headerTemplate: css + '<h1>' + 'Mira mi primer PDF con puppeteer' + '</h1>',
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
        pdfCSS.push('</style>');

        const css = pdfCSS.join(" ");

        await page.pdf({
            path: 'googleLandscape.pdf',
            format: 'A4',
          
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