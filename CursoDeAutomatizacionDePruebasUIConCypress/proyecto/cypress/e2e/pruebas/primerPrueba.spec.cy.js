describe("Primer Suite de Pruebas", () => {

    describe("Primer Suite de Pruebas", () => {
        it("primer prueba", () => {
            cy.visit("https://platzi.com");
        });
        it("segunda prueba", () => {
            cy.visit("https://explore.skillbuilder.aws/learn/signin");
        });
        it("tercera prueba", () => {
            cy.visit("https://www.youtube.com/watch?v=ubPsYnkxk68&list=PLkVpKYNT_U9c_tu4YcockdrZg_2lnJINe&ab_channel=OpenBootcamp")
        });

    });
    it("primer prueba", () => {
        cy.visit("https://www.chevrolet.com");
    });
    it("segunda prueba", () => {
        cy.visit("https://www.renault.com");
    });
    it("tercera prueba", () => {
        cy.visit("https://www.w3schools.com/python/python_ml_auc_roc.asp")
    });

});

describe("Segunda Suite de Pruebas", () => {
    it("primer prueba", () => {
        cy.visit("https://www.amazon.com");
    });
    it("segunda prueba", () => {
        cy.visit("https://www.mercadolibre.com");
    });
    it('tercera prueba', () => {
        cy.visit("https://www.suzuki.com")
    });

});