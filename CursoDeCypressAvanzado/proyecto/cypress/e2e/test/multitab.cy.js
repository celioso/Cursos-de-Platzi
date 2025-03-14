describe('Navegando entre diferentes tabs', function () {

    it('visitar links con target blank', function () {
        cy.visit('https://demoqa.com/links');
        cy.get('#simpleLink').click();
    });

    it('abrir la pagina dentro de la misma ventana', function () {
        cy.visit('https://demoqa.com/links');
        cy.get('#simpleLink').invoke('removeAttr', 'target').click();

    });
});