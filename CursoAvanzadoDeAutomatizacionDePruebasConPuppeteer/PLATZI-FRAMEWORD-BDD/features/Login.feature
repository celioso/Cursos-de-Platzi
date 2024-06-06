Feature: Logging in

Scenario: log in to the page

    Given Im on the main page
    When I fill in the form with my email: "wajav34577@jahsec.com" and my password: "123456789"
    Then I should see the dashboard page

@probando
Scenario Outline: Scenario Outline for login
    Given Im on the main page
    When I fill in the form with my <Email> and my <password>
    Then I should see the dashboard page

    Examples:
            | Email                  | Password    |
            | wajav34577@jahsec.com  | 123456789  |
            | jesuscuadro@gmail.com  | Jexxus2334  |
            | juanito@gmail.com      | Jexxus2334  |
            | pepito@gmail.com       | Jexxus2334  |
            | camila@gmail.com       | Jexxus2334  |
            | sadsadasr@gmail.com    | Jexxus2334  |

