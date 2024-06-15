describe('Trabajando con xpaths', ()=> {

    it('Obtenerlo con un css selector', ()=>{
        cy.visit('/');
        cy.get('#root > div.container > div:nth-child(1) > div:nth-child(1) > div > center > div.card-header > h1').should("contain", "Bulbasaur");
    });

    it('Obtenerlo con un xpath', ()=>{
        cy.visit('/');
        cy.xpath('//h1[contains(text(), "Bulbasaur")]').should("contain", "Bulbasaur");
    });

    it('Obtenerlo con un xpath más corto', ()=>{
        cy.visit('/');
        cy.contains("Bulbasaur").should("be.visible");
    });
});