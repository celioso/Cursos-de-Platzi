describe('flaky tests', ()=>{

    it.only('Single query command', ()=>{
        cy.visit('/');
        cy.get('#root > div.container > div:nth-child(1) > div:nth-child(1) > div > center > div.card-header > h1').should("contain", "Bulbasaaur");

        cy.get('#root > div.container > div:nth-child(1) > div:nth-child(1) > div > center > div.card-header > h1').should("contain", "Bulbasaaur");
    });

    it('Alternar comando con aserciones', ()=>{
        cy.visit('/');
        cy.get('#root > div.container > div:nth-child(1) > div:nth-child(1) > div > center > div.card-header > h1').should("contain", "Bulbasaur").parent().should("have.class", "card-header");
    });
});