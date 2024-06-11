describe('Probando requests',()=>{

    it('Debe de crear un empleado', function () {
        cy.request({
            url:'employees',
            method:'POST',
            body:{
                first_name: "Maria",
                last_name: "Perez",
                email: "Maria@platzi.com"
            }
        }).then(response=>{
            expect(response.status).to.eq(201);
            expect(response.body).to.have.property('id')

            const id = response .body.id;
            cy.wrap(id).as('id')
        })
    });

    it('Debe de validar que se haya creado en la base de tatos',()=>{

        cy.request('GET', 'employees').then(response=>{
            expect(response.body[response.body.length - 1].first_name).to.eq("Maria");
        });
    })

    it("Debemos de modificar al empleado con un nuevo correo", function () {

        cy.request({
            url: `employees/${this.id}`,
            method: "PUT",
            body: {
                first_name: "Pepito 3",
                last_name: "Desarrollador",
                email: "nuevo@correo.com",
            },
        }).then((response) => {
            cy.log(response);
            expect(response.status).to.eq(200);
            expect(response.body).to.have.property("id");
        });
    });

    it("Debemos de eliminar el registro creado", function () {
        cy.request({
            url: `employees/${this.id}`,
            method: "DELETE",
        }).then((response) => {
            expect(response.status).to.eq(200);
        });
    });
});
