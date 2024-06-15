describe('Login con custon commands', ()=>{

    it('login error', ()=>{

        cy.login('548845','548844')
    });

    it('login correcto', ()=>{

        cy.login('username','password')
    });
});