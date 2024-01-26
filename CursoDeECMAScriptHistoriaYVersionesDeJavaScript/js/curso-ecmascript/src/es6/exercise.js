import getData from "./api.js";

export function solution() {
  //return getData();
  return getData().then(movies => movies);
}

