import BasePage from "./BasePage";

export default class LoginPage extends BasePage {

    constructor(){
        super();
        this.navBar = '#navbarSupportedContent';
        this.inputEmail = "#email";
        this.inputPassword = '#password';
        this.submitButton = '#submitBTN';
        this.loginPageText = "//div//div//div[contains(text(),'juan camilo')]";
        
    }

    async visit() {
        try {
            await page.goto("https://phptravels.net/login");
            await page.waitForSelector(this.navBar);
            const url = await this.getUrl();
            console.log(url);
        } catch (e) {
            throw new Error(`Error al visitar la página: ${e.message}`);
        }
    }

    async login(email, password) {
        try {
            await this.type(this.inputEmail, email);
            await this.type(this.inputPassword, password);
            await this.click(this.submitButton);
        } catch (e) {
            throw new Error(`Error al iniciar sesión: ${e.message}`);
        }
    }

    async validateLogin() {
        try {
            await page.waitForSelector(this.loginPageText);
            await page.waitForSelector(this.navBar);
        } catch (e) {
            throw new Error(`Error al validar el inicio de sesión: ${e.message}`);
        }
    }
}