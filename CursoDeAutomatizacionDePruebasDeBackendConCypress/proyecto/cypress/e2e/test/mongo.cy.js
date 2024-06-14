describe('Interacción con MongoDB a través de Cypress', () => {
    const url = 'mongodb+srv://celioso1:<password>@cluster0.edssdgd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0';
  
    /*before(() => {
      cy.task('clearPrueba', { url });
    });*/
  
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
    });
  });