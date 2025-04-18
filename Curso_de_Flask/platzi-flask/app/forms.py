from flask_wtf import FlaskForm
from wtforms.fields import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    username = StringField("Nombre de usuario", validators = [DataRequired()])
    password = PasswordField("Password", validators = [DataRequired()])
    submit = SubmitField("Enviar")

class TodoForm(FlaskForm):
    description = StringField("Descripción", validators=[DataRequired()])
    submit = SubmitField("Crear")

class DeleteTodoForm():
    submit = SubmitField("Borrar")


class UpdateTodoForm():
    submit = SubmitField("Actualizar")

    