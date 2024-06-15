import { loginPage } from "./pageObjects/loginPage";

describe('login con POM', ()=>{
    beforeEach(()=>{
        loginPage.visit();
    });

    it("Login erroneo", ()=>{
        loginPage.validateLoginPage()
        loginPage.login('lalalal', 'kshhdytsd');
        loginPage.validateErrorLogin();
    });

    it("Login exitoso", ()=>{
        loginPage.validateLoginPage()
        loginPage.login('username', 'password');
        loginPage.validateSuccessLogin();
    });
});
