from flask import Flask

app = Flask(__name__)


@app.route("/")
def hello():
    return "¡Hola mundo!"

'''if __name__ == "__main__":
    app.run(debug=True)'''

@app.route("/nota")
def nota():
    return "La aplicación es para que el usuario crea sus notas y luego se puedan cambiar o eliminar."