describe("Seguridad", ()=>{

    it('Navegar entre diferentes dominios', ()=>{
        cy.visit('/');
        cy.visit("https://todo-cypress-iota.vercel.app");
        cy.get("#title").type("Titulo de prueba");
    });

    it.only("navego a un dominio", ()=>{
        cy.visit("https://todo-cypress-iota.vercel.app");
        cy.get("h1").invoke("text").as("titulo");
    });

    it.only("navego a otro dominio", ()=>{
        cy.visit("/");
        cy.log(this.titulo);
        //cy.get("h1").invoke('text').as("titulo");
    })
});