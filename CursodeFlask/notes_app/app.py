from flask_sqlalchemy import SQLAlchemy
from config import Config
from models import Note
from flask import (Flask, 
                    redirect, 
                    render_template, 
                    request, 
                    url_for)

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)

@app.route('/')
def home():
    notes = Note.query.all()
    return render_template('home.html', notes=notes)

@app.route("/nota")
def nota():
    return "La aplicación es para que el usuario pueda crea sus notas y luego se puedan cambiar o eliminar."

@app.route("/acerca-de")
def about():
    return "esto es una app de notas"

@app.route("/contacto", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        return "Formulario enviado correctamente", 201
    return "Pagina de contacto"

@app.route("/crear-nota", methods=["GET", "POST"])
def create_note():
    if request.method == "POST":
        title = request.form.get("title", "")
        content = request.form.get("content", "")

        note_db = Note(
            title=title, content=content
        )

        db.session.add(note_db)
        db.session.commit()

        return redirect(
            url_for("home")
        )
    return render_template("note_form.html")

@app.route("/editar-note/<int:id>", methods=["GET", "POST"])
def edit_note(id):
    note = Note.query.get_or_404(id)
    if request.method == "POST":
        title = request.form.get("title","")
        content = request.form.get("content", "")
        note.title = title
        note.content = content
        db.session.commit()
        return redirect(url_for("home"))

    return render_template("edit_note.html", note=note)

@app.route("/borrar-note/<int:id>", methods=["POST"])
def delete_note(id):
    note = Note.query.get_or_404(id)
    db.session.delete(note)
    db.session.commit()
    return redirect(url_for("home"))





'''if __name__ == "__main__":
    app.run(debug=True)'''