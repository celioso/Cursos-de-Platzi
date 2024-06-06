import BasePage from "./BasePage";

export default class FlightsPage extends BasePage {

    constructor(){
        super();
        this.mainDiv="#tab-group-events"
        this.inputs={
            form:"#onereturn > div.col-lg-3.show.active > div.input-items.from_flights.show.active > div.form-floating > span > span.selection > span",
            to:"#onereturn > div:nth-child(2) > div.input-items.flights_arrival.to_flights > div.form-floating > span > span.selection > span",
            date:"#departure",
            passengers:"#onereturn > div.col-lg-2 > div > div > div > a",
            search:"#flights-search",
            firstOption:"#onereturn > div.col-lg-2 > div > div > div > div",
            moreAdultsPassengers:"#onereturn > div.col-lg-2 > div > div > div > div > div.dropdown-item.adult_qty.show.active > div > div > div.qtyInc > svg",

        }
        
    }

    async validatePage() {

        await page.waitForNavigation({ waitUntil: "networkidle2"})
        await page.waitForSelector(this.mainDiv);
        await page.waitForSelector(this.inputs.form);
        await page.waitForSelector(this.inputs.to);
        await page.waitForSelector(this.inputs.date);
        await page.waitForSelector(this.inputs.passengers);
        await page.waitForSelector(this.inputs.search);

    }

    async selectFlighy(from, to, date, passengers) {
        
        await this.type(this.inputs.from, from);
        await this.click(this.inputs.firstOption)

        await this.type(this.inputs.to, to);
        await this.click(this.inputs.firstOption);

        await this.type(this.inputs.date, date);

        if(passengers !==1){

            await this.click(this.inputs.passengers);
            for(let i = 0; i < passengers - 1; i++){
                await this.click(this.inputs.moreAdultsPassengers)
            }
        }

        await this.click(this.input.search)
    }

    async validateFligths() {
    
            await this.wait(5);
    }
}