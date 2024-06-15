describe('Cookies', function(){
    beforeEach(() => {
        cy.session("Cookies",() => {
            cy.setCookie('nombre', 'Mario');
        });
    });

    it('Obtener las cookies', () => {
        cy.clearAllCookies()
        cy.visit("/")
        cy.getCookies().should('be.empty')
    });

    it('Agregar una cookie', () => {
        cy.setCookie('nombre', 'Mario')
        cy.getCookies().should('have.length', 1)
    });

    it('Obtener cookie especifica', () => {
        cy.getCookie('nombre').should('have.a.property', "value", "Mario");
    });
});