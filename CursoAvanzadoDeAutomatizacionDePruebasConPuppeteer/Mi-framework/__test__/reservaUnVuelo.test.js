import LoginPage from "../pages/LoginPage";
import FlightsPage from "../pages/FlightsPage";
import Navbar from "../componets/NavBar";

let loginPage;
let flightsPage;
let navBar;

describe("Debemos iniciar sesion en la pagina", () => {

    beforeAll(async () => {
        loginPage = new LoginPage();
        flightsPage = new FlightsPage();
        navBar = new Navbar();

    }, 10000); 

    it("Debemos iniciar sesion", async () => {
        await loginPage.visit();
        await loginPage.login("wajav34577@jahsec.com", "123456789");
    }, 30000);

    /*it("Validar que esté en el dashboard", async () => {
        await loginPage.validateLogin();
    }, 30000);*/

    it("Navegar hacia la pagina de vuelos", async () => {
        await navBar.validateNavBarIsPresent();
        await navBar.selectMenuItem("Flights")
    }, 30000);

    it("Validar que estemos en vuelos y seleccionar vuelos", async () => {
        await flightsPage.validatePage();
        await flightsPage.selectFlight("Mexico", "Paris", "20-11-2024", 5)
    }, 30000);

    it("Validar que hayamos buscado el vuelo", async () => {
        await flightsPage.validateFligths();    
    }, 30000);


});