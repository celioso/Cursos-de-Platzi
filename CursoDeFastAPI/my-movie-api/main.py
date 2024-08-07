from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from utils.jwt_manager import create_token
from config.database import engine, Base
from middlewares.error_handler import ErrorHandler
from routers.movie import movie_router
from routers.user import user_router

app = FastAPI()
app.title = "Mi aplicación con FastAPI"
app.version = "0.0.1"

app.add_middleware(ErrorHandler)

app.include_router(movie_router)
app.include_router(user_router)

Base.metadata.create_all(bind=engine)


movies = [
    {
		"id": 1,
		"title": "Avatar",
		"overview": "En un exuberante planeta llamado Pandora viven los Na'vi, seres que ...",
		"year": "2009",
		"rating": 8.9,
		"category": "Acción"
	},
    {
		"id": 2,
		"title": "Godzilla vs. Kong",
		"overview": "Cinco años después de que Godzilla derrotara a Ghidorah...",
		"year": "2021",
		"rating": 7.8,
		"category": "Acción"
	}
]

@app.get("/", tags = ["home"])
def masage():
    return HTMLResponse("<h1>Hello World!</h1>"
                        "<p>Despliegue de App con FastAPP</p>")