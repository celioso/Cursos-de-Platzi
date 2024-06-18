const {
    Given,
    When,
    Then,
} = require('@badeball/cypress-cucumber-preprocessor');
const {loginPage} = require("../../../e2e/test/pageObjects/loginPage");

Given('I am on the login page', () => {
    loginPage.visit();
    loginPage.validatePageLogin();
});

When(`I fill in my email and password with {string} and {string}`, (username, password) => {
    loginPage.login(username, password);
});

Then('I should validate that I\'m logged in', () => {
    loginPage.validateSuccessLogin();
});