import unittest
from flask import request, make_response, redirect, render_template, session, url_for, flash
from flask_login import login_required, current_user, delete_todo, update_form


from app import create_app
from app.forms import TodoForm, DeleteTodoForm
from app.firestore_service import update_todo, get_todos, put_todo

app = create_app()

@app.cli.command()
def test():
    tests = unittest.TestLoader().discover("tests")
    unittest.TextTestRunner().run(tests)

@app.errorhandler(404)
def not_found(error):
    return render_template("404.html", error = error)

@app.errorhandler(500)
def Internal_Server_Error(error):
    return render_template("50.html", error=error)

@app.route("/")
def index():
    user_ip = request.remote_addr

    response = make_response(redirect("/hello"))
    session["user_ip"] = user_ip
    #response.set_cookie("user_ip", user_ip) # Se usa para guardar la IP en una cookie

    return response

@app.route("/hello", methods=["GET", "POST"])
@login_required
def hello():
    user_ip = session.get("user_ip")
    # user_ip = request.cookies.get("user_ip")  # para cookies
    username = current_user.id
    todo_form = TodoForm()
    delete_form = DeleteTodoForm()

    
    context ={
        "user_ip" : user_ip, 
        "todos" : get_todos(user_id=username),
        "username" : username,
        "todo_form" : todo_form,
        "delete_form" : delete_form,
        "update_form" : update_form,
    }

    if todo_form.validate_on_submit():
        put_todo(user_id=username, description=todo_form.description.data)

        flash("Tu tarea e creo con exito!")

        return redirect(url_for("hello"))

    return render_template("hello.html", **context)



@app.route("/todos/delete/<todo_id>", method=["POST"])
def delete(todo_id):
    user_id = current_user.id
    delete_todo(user_id=user_id, todo_id=todo_id)
    
    return redirect(url_for("hello"))

@app.route("/todos/update/<todo_id>/<int:done>", method=["POST"])
def update(todo_id, done):
    user_id = current_user.id

    update_todo(user_id=user_id, todo_id=todo_id, done=done)

    return redirect(url_for("hello"))