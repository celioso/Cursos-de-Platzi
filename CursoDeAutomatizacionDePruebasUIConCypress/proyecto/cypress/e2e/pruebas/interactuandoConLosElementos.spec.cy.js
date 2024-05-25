Cypress.on('uncaught:exception', (err, runnable) => {
    // Registrar el error en la consola
    console.error('Excepción no capturada', err);
    
    // Devolver false aquí previene que Cypress falle la prueba
    return false;
});

describe("INteractuar con los elementos",()=>{

    let texto

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

    it("Click por posicion", ()=>{
        cy.visit("/buttons")
        cy.get("button").eq(3).parent().parent().click("topRight")
        cy.get("button").eq(3).parent().parent().click("bottomLeft")
        cy.get("button").eq(3).parent().parent().click(5, 60)
    })

    it("Input type text", ()=>{
        cy.visit("/automation-practice-form")
        cy.get("#firstName").type("Mario")
        cy.get("#lastName").type("Fuentes")

        cy.get("#firstName").type("Mario")

        cy.get("#firstName").type("{selectAll}{backspace}")
        cy.get("#firstName").type("Otro nombre")
        cy.get("#firstName").clear()

    })

    it("Checkboxes y radio botones", ()=>{
        cy.visit("/automation-practice-form")
        //cy.get("#gender-radio-1").click()
        //cy.get("#gender-radio-1").click({force:true})
        //cy.get("#gender-radio-1").check({force:true})
        cy.get('label[for="gender-radio-1"]').click()
  
        //cy.get('#hobbies-checkbox-1').click({force:true})
        //cy.get('#hobbies-checkbox-1').check({force:true})
        //cy.get('#hobbies-checkbox-1').uncheck({force:true})
        cy.get('label[for="hobbies-checkbox-1"]').click()
        cy.get('label[for="hobbies-checkbox-1"]').click()

    })

    it("Extrayendo info", function(){
        cy.visit("/automation-practice-form")

        cy.get("#firstName").as("nombre")
        cy.get("@nombre").type("Javier")

        cy.get("@nombre").then(($nombre)=>{
            texto = $nombre.val()
            expect(texto).to.equal("Javier")
        })
        
        cy.get("@nombre").invoke("val").should("equal", "Javier")
        cy.get("@nombre").invoke("val").as("nombreGlobal")

    })

    it("Compartir info", function(){
        cy.visit("/automation-practice-form")
        cy.get("#lastName").as("nombre2")
        cy.get("@nombre2").type(texto)
        cy.get("#firstName").type(this.nombreGlobal)

       
    })

    it('extract information with function', function() {
        cy.visit('https://demoqa.com/automation-practice-form');
        cy.get('#lastName').type('Vida')
        cy.get('#lastName').invoke('val').as('GlobalVariable')
    })

    it('get information with function', function() {
        cy.visit('https://demoqa.com/automation-practice-form');
        cy.get('#firstName').type(this.GlobalVariable)
    })

    it("Interactuando con los dropdown(select)", function(){
        cy.visit("https://itera-qa.azurewebsites.net/home/automation")
        cy.get(".custom-select").select(10)
        cy.get(".custom-select").select("3").should("have.value", "3")
        cy.get(".custom-select").select("Greece").should("have.value", "4")
    })

    it("Interactuando con los dropdown(select) dinamico", function(){
        cy.visit("https://react-select.com/home")
        cy.get("#react-select-6-input").type(" ")

        cy.get("#react-select-6-listbox").children().each(($el, index, $list)=>{

            if($el.text() === "Red"){
                //$el.on("click")
                $el.click()
            }
        })

        //cy.get("#react-select-6-option-3").click()
    })

    it.only("Interactuando con tablas", function(){
        cy.visit("https://www.w3schools.com/html/html_tables.asp")
        cy.get("#customers").find("th").each(($el)=>{
            cy.log($el.text())
        })

        cy.get("#customers").find("th").first().invoke("text").should("equal", "Company")

        cy.get("#customers").find("th").eq(1).invoke("text").should("equal", "Contact")

        cy.get("#customers").find("th").eq(2).invoke("text").should("equal", "Country")
        
        cy.get("#customers").find("tr").should("have.length",7)

        cy.get("#customers").find("tr").eq(1).find("td").eq(1).invoke("text").should("equal","Maria Anders")

        cy.get("#customers").find("tr").eq(1).find("td").eq(1).then($el=>{
            const texto = $el.text()
            expect(texto).to.equal("Maria Anders")
        })
    })
})