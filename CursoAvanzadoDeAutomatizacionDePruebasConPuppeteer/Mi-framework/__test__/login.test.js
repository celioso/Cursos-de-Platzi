import LoginPage from "../pages/LoginPage";

let loginPage;

describe("Iniciar sesión en la página", () => {

    beforeAll(async () => {
        loginPage = new LoginPage();
        await loginPage.visit(); 
    }, 10000); 

    it("Debería ir a la página", async () => {
        await loginPage.visit()
    }, 20000);

    it("Debería llenar los campos", async () => {
        await loginPage.login("wajav34577@jahsec.com", "123456789");
    }, 20000);

    /*it("Validar que esté en el dashboard", async () => {
        await loginPage.validateLogin();
    }, 30000);*/

});