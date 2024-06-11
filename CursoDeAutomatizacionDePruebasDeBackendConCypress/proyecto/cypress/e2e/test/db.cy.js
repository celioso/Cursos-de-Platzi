describe('Prueba a la base de datos',function(){
    /*after(() => {
        cy.task("queryDb", "DELETE FROM nombres");
      });*/

    it('Insert', function(){
        cy.task("queryDb","INSERT INTO nombres(nombre, apellidoMaterno, apellidoPaterno) VALUES('Javier', 'Fuentes', 'Mora')").then(results =>{
            cy.log(results);
            expect(results.affectedRows).to.eq(1);
            cy.wrap(results.insertId).as("id");
        });
    });

    /*it('Select', function(){
        cy.task("queryDb","SELECT * FROM nombres").then(results =>{
            cy.log(results)
        });
    });*/

    it('Select para comprobar que este lo de la prueba pasada', function(){
        cy.task("queryDb",`SELECT * FROM nombres WHERE id=${this.id}`).then(results =>{
            cy.log(results);
            expect(results[0].nombre).to.eq("Javier");
            expect(results[0].apellidoMaterno).to.eq("Fuentes");
            expect(results[0].apellidoPaterno).to.eq("Mora");
        });
    });
    
    it('Select para borrar que este lo de la prueba pasada', function(){
        cy.task("queryDb",`DELETE FROM nombres WHERE id=${this.id}`).then(results =>{
            cy.log(results);
            expect(results.affectedRows).to.eq(1);
            expect(results.serverStatus).to.eq(2);
        });
    });
});