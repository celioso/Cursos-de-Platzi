import BasePage from "../pages/BasePage";

export default class NavBar extends BasePage{

    constructor(){
        super()
        this.navBar = "#\#fadein > header",
        this.menu={
            home:" #\#fadein > header > div > div.d-flex > a > img",
            hotels:"#navbarSupportedContent > div.nav-item--left.ms-lg-5 > ul > li:nth-child(2) > a",
            flights:"#navbarSupportedContent > div.nav-item--left.ms-lg-5 > ul > li:nth-child(1) > a"
            
        };
    };

    async validateNavBarIsPresent(){
        await page.waitForSelector(this.navBar);
        await page.waitForSelector(this.menu.home);
        await page.waitForSelector(this.menu.hotels);
        await page.waitForSelector(this.menu.flights);
    }

    async selectMenuItem(menuItem){
        await this.click(this.menu[menuItem]);


    }
}