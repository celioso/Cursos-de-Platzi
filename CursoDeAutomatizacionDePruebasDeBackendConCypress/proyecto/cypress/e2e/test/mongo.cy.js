describe('Interacción con MongoDB a través de Cypress', () => {
/*after (()=>{
  cy.task('clearNombres')  //esta parte del cogigo borraa toda la base de datos
});*/

  it("select de mongo", function(){
    cy.task("getListing").then(results=>{
      cy.log(results);
      //expect(results).to.have.length(3);
    });
  });

  it("Create de mongo", function(){
    cy.task("createName",{
        "first_name": "Javier",
        "last_name": "Eschweiler",
        "email": "javier@platzi.com"
    }).then(results=>{
      cy.log(results);
      expect(results.acknowledged).to.eq(true);
      expect(results).to.haveOwnPropertyDescriptor("insertedId");
    });
  });



  // codigo mejorado por ChatGPT

    /*const url = 'mongodb://localhost:27017/';

  
    it.only('select de mongo', function() {
      cy.task('getListing', { url }).then(results => {
        cy.log(results);
        // expect(results).to.have.length(50);
      });
    });
  
    it('create de mongo', function() {
      const prueba = {
        first_name: "Carlos",
        last_name: "Eschweiler",
        email: "javier@platzi.com"
      };
      cy.task('createPrueba', { url, prueba }).then(results => {
        cy.log(results);
        // expect(results).to.have.length(50);
      });
    });*/
  });