describe('Uniendo todo', function(){

    it('Debemos eliminar el registro creado', function () {
        const id = 2;
        cy.request({
            url: `employees/${id}`,
            method: "DELETE"
        }).then(response =>{
            expect(response.status).to.eq(200)
        });
    });

    it('Debemos validar que no esta en la DB', function () {
        cy.task("queryDb", `SELECT * FROM employees WHERE id = ${id}`)
        .then(results=>{
            cy.log(results);
            expect(results.length).to.eq(0)
        });
    });

});