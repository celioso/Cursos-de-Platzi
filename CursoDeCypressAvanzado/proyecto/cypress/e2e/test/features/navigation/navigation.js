const {
    Given,
    When,
    Then,
} = require('@badeball/cypress-cucumber-preprocessor');
const {loginPage} = require("../../pageObjects/loginPage");

Given('I am on the login page', () => {
    loginPage.validateSuccessLogin();
});

When('I click on the Account Activity Nav', ()=> {
    cy.contains("Account Activity").click();
});

Then('I should see theAccount Activity content',()=> {
    cy.contains("Show Transactions").should("be.visible");
});