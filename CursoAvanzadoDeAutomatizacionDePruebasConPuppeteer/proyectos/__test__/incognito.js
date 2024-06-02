const puppeteer = require('puppeteer');

(async () => {
    try {
        // Lanzar el navegador
        const browser = await puppeteer.launch({ headless: false }); // headless: false para ver el navegador
        // Crear un contexto de navegador en modo incógnito
        const context = await browser.createIncognitoBrowserContext();
        // Crear una nueva página en el contexto incógnito
        const page = await context.newPage();
        // Navegar a la URL especificada
        await page.goto('https://platzi.com', { waitUntil: 'networkidle2' });

        // Realizar alguna operación, por ejemplo, obtener el título de la página
        const title = await page.title();
        console.log('Título de la página:', title);

        // Cerrar el navegador
        await browser.close();
    } catch (error) {
        console.error('Error:', error);
    }
})();
