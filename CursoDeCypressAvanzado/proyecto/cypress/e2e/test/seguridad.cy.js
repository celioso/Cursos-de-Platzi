let texto;

describe("Seguridad", ()=>{

    it('Navegar entre diferentes dominios', ()=>{
        cy.visit('/');
        cy.visit("https://todo-cypress-iota.vercel.app");
        cy.get("#title").type("Titulo de prueba");
    });

    it("navego a un dominio", ()=>{
        cy.visit("https://todo-cypress-iota.vercel.app");
        cy.get("h1").invoke("text").as("titulo");
    });

    it("navego a otro dominio", ()=>{
        cy.visit("/");
        cy.log(this.titulo);
        //cy.get("h1").invoke('text').as("titulo");
    });

    it("navego en dos dominios en el mismo test", ()=>{
        cy.visit("/");
        cy.get("h1")
            .first()    
            .invoke("text") 
            .then((text)=>{
            Cypress.env({
                textito:text
            })
        });
        cy.origin("https://todo-cypress-iota.vercel.app", {args:{texto: "Hola"}}, function ({texto}){
            cy.visit('/');
            cy.log(texto);
            cy.log(Cypress.env());
        });
        cy.visit("/");
        cy.get("h1").first().invoke("text").should("be.equal", Cypress.env('textico'));
    });

    it.only("Compartir informacion si usar session", ()=>{
        cy.visit("/");
        cy.get("h1").first().invoke("text").then((text) =>{
            cy.task("guardar", {texto:text});
        });
    });

    it.only("Compartir informacion si usar session 2", ()=>{
        cy.visit("https://todo-cypress-iota.vercel.app");
        cy.task("obtener", "texto").then((valor) =>{
            cy.get("#title").type(valor);
        })
    });
});