Cypress.on("uncaught:exception", (err, runnable) => {
    // Registrar el error en la consola
    console.error("Excepción no capturada", err);
    
    // Devolver false aquí previene que Cypress falle la prueba
    return false;
  });

describe("Guardando elementos", () => {

    it("evitar repeticion", () => {
        cy.visit("/automation-practice-form")
        //Obteniendo el elemento el padre

        const input = cy.get('input[placeholder="First Name"]').parents("form").then((form)=>{

            const inputs = form.find("input")
            const divs = form.find("div")
            const labels = form.find("label")

            expect(inputs.length).to.equal(15)
            cy.wrap(inputs).should("have.length", 15)
            expect(divs.length).to.equal(70)
            expect(labels.length).to.equal(16)   

            //console.log("soy la longitud", inputs.length)
            //debugger // para ver la cantidad de elementos que tiene la paguna siempre hay que tener abierta la consola.
            cy.log("soy la longitud", inputs.length)

            cy.task("log", inputs.length)
        })

        cy.get("form").find("label")
        //cy.pause() // pausa la ejecución       
        cy.get('input[placeholder="First Name"]').then(console.log)
        cy.get('input[placeholder="First Name"]').debug()
    });
   
});