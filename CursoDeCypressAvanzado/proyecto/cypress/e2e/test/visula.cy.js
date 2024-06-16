describe("Visual testing", ()=>{

        it('Mi prueba de regresion', ()=>{
            cy.visit('/');

            cy.wait(4000);
            cy.scrollTo("bottom");
            cy.matchImageSnapshot();
        });

        it("Segunda prueba a un solo elemento", ()=>{
            cy.visit('/');
            cy.contains("Bulbasaur").should("be.visible").matchImageSnapshot();
    
        });
});