const regex = /(\d{4})-(\d{2})-(\d{2})/;
const matchers = regex.exec("2023-12-14");
console.table(matchers);