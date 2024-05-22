Cypress.on("uncaught:exception", (err, runnable) => {
    // Registrar el error en la consola
    console.error("Excepción no capturada", err);
    
    // Devolver false aquí previene que Cypress falle la prueba
    return false;
  });

describe("Esperando por elementos", () => {

    beforeEach(()=>{
        cy.visit("https://www.platzi.com")
    })

    it("Esperar por un tiempo definido", () => {
        cy.wait(5000)
        
    });

    it("Esperar por un elemento hace una asercion", () => {
        cy.get(".Button-module_Button-label__hlOeK",{timeout:6000}).should("be.visible")
        
    });

   
});

describe("Esperando por elementos", () => {

    beforeEach(()=>{
        cy.visit("/")
    })

    it.only("Deshabilitar el retry", () => {
        //cy.get(".banner-image444", {timeout:5000})
        //cy.get(".:nth-child(3) > :nth-child(1) > .card-body", {timeout:5000})
        cy.get(".:nth-child(3) > :nth-child(1) > .card-body", {timeout:0})  // para deshabilitar
    });

   
});