from flask import Flask, request

from config import Config
from models import db
from notes.routes import notes_bp
from auth.routes import auth_bp

def create_app(config_class= Config):
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)
    app.register_blueprint(notes_bp)
    app.register_blueprint(auth_bp)

    with app.app_context(): # crea la base de datos si no existe
        db.create_all() 


    @app.route("/acerca-de")
    def about():
        return "esto es una app de notas"


    @app.route("/contacto", methods=["GET", "POST"])
    def contact():
        if request.method == "POST":
            return "Formulario enviado correctamente", 201
        return "Pagina de contacto"
    
    return app

"""if __name__ == "__main__":
    app.run(debug=True)"""