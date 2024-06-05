export default class BasePage {

    async getTitle() {
        return await page.title();
    }

    async getUrl() {
        return await page.url();
    }

    async getText(selector) {
        try {
            await page.waitForSelector(selector);
            return await page.$eval(selector, el => el.textContent);
        } catch (e) {
            throw new Error(`Error al obtener el texto del selector ${selector}`);
        }
    }

    async getAttribute(selector, attribute) {
        try {
            await page.waitForSelector(selector);
            return await page.$eval(selector, el => el.getAttribute(attribute));
        } catch (e) {
            throw new Error(`Error al obtener el atributo del selector ${selector}`);
        }
    }

    async getValue(selector) {
        try {
            await page.waitForSelector(selector);
            return await page.$eval(selector, el => el.value);
        } catch (e) {
            throw new Error(`Error al obtener el valor del selector ${selector}`);
        }
    }

    async getCount(selector) {
        try {
            await page.waitForSelector(selector);
            return await page.$$eval(selector, el => el.length);
        } catch (e) {
            throw new Error(`Error al obtener el número de elementos del selector ${selector}`);
        }
    }

    async click(selector) {
        try {
            await page.waitForSelector(selector);
            await page.click(selector);
        } catch (e) {
            throw new Error(`Error al hacer clic en el selector ${selector}`);
        }
    }

    async type(selector, text, opts = {}) {
        try {
            await page.waitForSelector(selector);
            await page.type(selector, text, opts);
        } catch (e) {
            throw new Error(`Error al escribir en el selector ${selector}`);
        }
    }

    async doubleClick(selector) {
        try {
            await page.waitForSelector(selector);
            await page.click(selector, { clickCount: 2 });
        } catch (e) {
            throw new Error(`Error al hacer doble clic en el selector ${selector}`);
        }
    }

    async wait(time) {
        return await new Promise(resolve => setTimeout(resolve, time));
    }
}