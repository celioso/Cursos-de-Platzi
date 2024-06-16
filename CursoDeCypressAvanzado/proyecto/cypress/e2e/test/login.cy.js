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

    it("Login exitoso con cy.env", ()=>{
        loginPage.validateLoginPage();
        cy.log(Cypress.env());

        loginPage.login(Cypress.env("credentials").user, Cypress.env("credentials").password);
        loginPage.validateSuccessLogin();
    });

    it("Login exitoso con cy.env.json", ()=>{
        loginPage.validateLoginPage();
        cy.log(Cypress.env());

        loginPage.login(
            Cypress.env("credentials").user, 
            Cypress.env("credentials").password
    );
    //loginPage.validateErrorLogin();
    });

    it("Login erroneo desde la termina", ()=>{
        loginPage.validateLoginPage();
        cy.log(Cypress.env());

        loginPage.login(
            Cypress.env("credentials").user, 
            Cypress.env("credentials").password
    );
    });
});

describe.only("Login erroneo con configuracion",{
env:{
    usuarioErroneo:"error1",
    passwordErroneo:"error2",
    },
},
()=>{
    beforeEach(() => {
        loginPage.visit();
    });

    it("login erroneo", ()=>{
        loginPage.validateLoginPage();
        cy.log(Cypress.env());

        loginPage.login(
            Cypress.env("UsuarioErroneo"), 
            Cypress.env("passwordErroneo")
        );
        //loginPage.validateErrorLogin
    });

    it("login erroneo con variables de entorno", () => {
        loginPage.validatePageLogin();
        cy.log(Cypress.env());
        loginPage.login(Cypress.env("variable"), Cypress.env("variable"));
        //loginPage.validateErrorLogin();
      });
});

