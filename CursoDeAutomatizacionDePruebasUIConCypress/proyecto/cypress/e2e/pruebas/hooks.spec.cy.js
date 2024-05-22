Cypress.on('uncaught:exception', (err, runnable) => {
    // Registrar el error en la consola
    console.error('Excepción no capturada', err);
    
    // Devolver false aquí previene que Cypress falle la prueba
    return false;
  });

  // before(() => {
// 	//solo si se quiere correr antes de los dos bloques
// 	cy.visit('/automation-practice-form')
// })

describe("Aserciones", () => {
    beforeEach(()=>{ //con beforeEach lo repite las veces que sea necesario
        cy.visit('/automation-practice-form')
    })

    after(()=>{
        cy.visit("/")
    })

    it('Asercion', () => {
        //cy.visit('/automation-practice-form')
        cy.url().should("include", "demoqa.com")
        cy.get("#firstName").should("be.visible").and("have.attr", "placeholder", "First Name")
    });

    it('Asercion 2', () => {
        //cy.visit('/automation-practice-form')
        cy.url().should("include", "demoqa.com")
        cy.get("#firstName").then((element)=>{
            expect(element).to.be.visible
            expect(element).to.have.attr("placeholder", "First Name")
    })
});

    it('Asercion 3', () => {
        //cy.visit('/automation-practice-form')
        cy.url().should("include", "demoqa.com")
        cy.get("#firstName").then((element)=>{
            assert.equal(element.attr("placeholder"),"First Name")
    })
})

});
