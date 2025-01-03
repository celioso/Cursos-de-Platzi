describe("Oribando errores", ()=>{

    it("Debe validar el status code fallido y el mensaje de error", ()=>{

        cy.request({url: 'https://pokeapi.co/api/v2/aaa', failOnStatusCode: false})
        .then(response =>{
        expect(response.status).to.eq(404);
        expect(response.body).to.be.eq("Not Found");
        });
    });

    it("Debe validar el status code fallido y el mensaje de error de rick and morty", ()=>{

        cy.request({url: 'https://rickandmortyapi.com/api/location/5487', failOnStatusCode: false})
        .then(response =>{
        expect(response.status).to.eq(404);
        expect(response.body).to.have.property('error', "Location not found");
        });
    });
})