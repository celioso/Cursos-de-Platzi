import { loginPage } from "./pageObjects/loginPage";

describe('login con POM', ()=>{
    beforeEach(()=>{
        loginPage.visit();
    });

    it("Login erroneo", ()=>{
        loginPage.validatePageLogin()
        loginPage.login('lalalal', 'kshhdytsd');
        loginPage.validateErrorLogin();
    });

    it("Login exitoso", ()=>{
        loginPage.validatePageLogin()
        loginPage.login('username', 'password');
        loginPage.validateSuccessLogin();
    });

    it("Login exitoso con cy.env", ()=>{
        loginPage.validatePageLogin();
        cy.log(Cypress.env());

        loginPage.login(Cypress.env("credentials").user, 
        Cypress.env("credentials").password
            );
        loginPage.validateSuccessLogin();
    });

    it("login erroneo con cy.env.json", () => {
        loginPage.validatePageLogin();

        cy.log(Cypress.env());
        loginPage.login(
            Cypress.env("credentials").user,
            Cypress.env("credentials").password
        );

        loginPage.validateSuccessLogin();
    });

    it("Login erroneo desde la termina", ()=>{
        loginPage.validatePageLogin();
        cy.log(Cypress.env());

        loginPage.login(
            Cypress.env("credentials").user, 
            Cypress.env("credentials").password
    );
    });
});

describe(
    "login erroneo con configuracion",
    {
        env: {
            usuarioErroneo: "error1",
            passwordErroneo: "error2",
        },
    },
    () => {
        beforeEach(() => {
            loginPage.visit();
        });

        it("login erroneo", () => {
            loginPage.validatePageLogin();
            cy.log(Cypress.env());
            loginPage.login(
                Cypress.env("usuarioErroneo"),
                Cypress.env("passwordErroneo")
            );
            loginPage.validateErrorLogin();
        });

        it("login erroneo con variables de entorno", () => {
            loginPage.validatePageLogin();
            cy.log(Cypress.env());
            loginPage.login(Cypress.env("variable"), Cypress.env("variable")
            );
            loginPage.validateErrorLogin();
        });
    }
);

describe("Login con fixtures",()=>{
    beforeEach(()=>{
        loginPage.visit();
    });

    it("login erroneo 2", ()=>{
        loginPage.validatePageLogin();

        cy.fixture("credentials").then( credentials =>{
            loginPage.login(credentials.email, credentials.password);
        });

        loginPage.validateErrorLogin();
    });

    it("login exitoso", ()=>{
        loginPage.validatePageLogin();

        cy.fixture("usuarios").then( credentials =>{
            loginPage.login(credentials.email, credentials.password);
        });

        loginPage.validateErrorLogin();
    });
});

const credentialsForUsers = [
    {
        nombre: "credentials",
        titulo: "Login con credentials",
    },
    {
        nombre: "usuarios",
        titulo: "Login con users",
    }
]

credentialsForUsers.forEach(credentials=>{
    describe.only(credentials.titulo, () =>{
        beforeEach(()=>{
            loginPage.visit();
        });

        it('login exitoso con fixtures', () => {
            loginPage.validatePageLogin();

            cy.fixture(credentials.nombre).then(credentials => {
                    loginPage.login(credentials.email, credentials.password);
                }
            );
            loginPage.validateErrorLogin();
        });
    });
});