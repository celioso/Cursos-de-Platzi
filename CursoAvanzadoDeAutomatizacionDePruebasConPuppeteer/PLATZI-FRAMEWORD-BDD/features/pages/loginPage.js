module.exports = new LoginPage();
module.exports.LoginPage = LoginPage();

const { I } = inject();

class LoginPage {
    constructor() {
        this.navBar = '#navbarSupportedContent';
        this.inputEmail = '#email';
        this.inputPassword = '#password';
        this.submitButton = '#submitBTN';
        this.loginPageText = "#\#fadein > div.container-fluid > div > div > div.pt-3 > div > div > div > div.w-100.text-center.mt-3 > span";
    }

    visit() {
        I.amOnPage('login');
        I.waitForElement(this.navBar);
        I.seeInCurrentUrl('login');
    }

    login(email, password) {
        I.waitForElement(this.inputEmail);
        I.fillField(this.inputEmail, email);
        I.fillField(this.inputPassword, password);
        I.click(this.submitButton);
        I.saveScreenshot("algo.png")
    }

    validateLogin() {
        I.waitForElement(this.loginPageText, 4);
        I.see('juan camilo', this.loginPageText);
        I.waitForElement(this.navBar, 4);
    }
}