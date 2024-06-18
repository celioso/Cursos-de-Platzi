Feature: Login test

    Scenario: I login with correct credentials
        Given I am on the login page
        When I fill in my email and password with "username" and "password"
        Then I should validate that I'm logged in