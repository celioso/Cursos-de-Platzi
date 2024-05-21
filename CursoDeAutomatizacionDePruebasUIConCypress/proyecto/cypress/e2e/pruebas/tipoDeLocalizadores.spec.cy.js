Cypress.on('uncaught:exception', (err, runnable) => {
    // Registrar el error en la consola
    console.error('Excepción no capturada', err);
    
    // Devolver false aquí previene que Cypress falle la prueba
    return false;
  });

describe('Probando configuracion', () => {

    it('Obteniendo por un tag', () => {
        cy.visit('/automation-practice-form')
        cy.get("input")
        
    });

    it('Obteniendo por un de un atributo', () => {
        cy.visit('/automation-practice-form')
        cy.get('[placeholder="First Name"]')
        
    });

    it('Obteniendo por un de un atributo y un tag', () => {
        cy.visit('/automation-practice-form')
        cy.get('input[placeholder="First Name"]')
        
    });

    it('Obteniendo por un de un id', () => {
        cy.visit('/automation-practice-form')
        cy.get("#firstName")
        
    });

    it('Obteniendo por un de un class', () => {
        cy.visit('/automation-practice-form')
        cy.get(".mr-sm-2.form-control")
        
    });

    it('usando Contains', () => {
        cy.visit('/automation-practice-form')
        cy.contains("Reading")
        cy.contains(".header-wrapper","Widget")
        
    });

    it('usando parent', () => {
        cy.visit('/automation-practice-form')
        //Obteniendo el elemento el padre
        cy.get('input[placeholder="First Name"]').parent()
        //Obteniendo el elemento el padres
        cy.get('input[placeholder="First Name"]').parents()

        cy.get('input[placeholder="First Name"]').parents().find("label")
        
        cy.get("form").find("label")
    });
});
