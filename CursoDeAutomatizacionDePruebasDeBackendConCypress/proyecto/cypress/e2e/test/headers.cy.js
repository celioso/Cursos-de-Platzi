describe('Probando Headers de la Api',()=>{
    it("Validar el Header y el content type", ()=>{
        cy.request('employees').its('headers').its('content-type').should('include','application/json')
    });
});