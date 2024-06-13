describe('Probando mongo', function(){

    it('select de mongo', function(){
        cy.task('getListing').then(results=>{
            cy.log(results);
            expect(results).to.have.length(50)
        })
    })
});