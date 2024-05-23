Cypress.on('uncaught:exception', (err, runnable) => {
    // Registrar el error en la consola
    console.error('Excepción no capturada', err);
    
    // Devolver false aquí previene que Cypress falle la prueba
    return false;
});

describe("INteractuar con los elementos",()=>{

    it("Click", ()=>{
        cy.visit("/buttons")
        cy.get("button").eq(3).click()
        cy.get("#dynamicClickMessage").should("be.visible").and("contain", "You have done a dynamic click")

    })

    it("Double Click", ()=>{
        cy.visit("/buttons")
        cy.get("#doubleClickBtn").dblclick()
        cy.get("#doubleClickMessage").should("be.visible").and("contain", "You have done a double click")

    })

    it("Right Click", ()=>{
        cy.visit("/buttons")
        cy.get("#rightClickBtn").dblclick()
        cy.get("#rightClickMessage").should("be.visible").and("contain", "You have done a right click")

    })

    it("Force Click", ()=>{
        cy.visit("/dynamic-properties")
        //cy.get("#enableArter").click({timeout:0})
        cy.get("#enableAfter").click({timeout:0, force: true})
        //cy.get("#rightClickBtn").dblclick()
        //cy.get("#rightClickMessage").should("be.visible").and("contain", "You have done a right click")
    })

    it.only("Click por posicion", ()=>{
        cy.visit("/buttons")
        cy.get("button").eq(3).parent().parent().click("topRight")
        cy.get("button").eq(3).parent().parent().click("bottomLeft")
        cy.get("button").eq(3).parent().parent().click(5, 60)
    })
})