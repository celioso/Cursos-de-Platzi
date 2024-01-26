const anotherFuncion = () => {
    return new Promise((resolve, rejecy)  => {
        if (true) {  // false: para que se active el rejecy
            resolve("Hey!!");
        }
        else {
            rejecy("Whooooops!");
        }
    })
}

anotherFuncion()
    .then(response => console.log(response))
    .catch(err => console.log(err));