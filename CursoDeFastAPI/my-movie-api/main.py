from fastapi import Depends, FastAPI, Body, HTTPException, Path, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from jwt_manager import create_token, validate_token
from fastapi.security import HTTPBearer

app = FastAPI()
app.title = "Mi aplicación con FastAPI"
app.version = "0.0.1"

class JWTBearer(HTTPBearer):
    async def __call__(self, request: Request):
        auth = await super().__call__(request)
        data = validate_token(auth.credentials)
        if data['email'] != "admin@gmail.com":
            raise HTTPException(status_code=403, detail="Credenciales son invalidas") # .decode('utf-8') si sale algun problema Agrega .decode('utf-8')

class User(BaseModel):
    email:str
    password:str

class Movie(BaseModel):
    id:Optional[int] = None
    title:str = Field(min_length=3, max_length=15)
    overview: str = Field(min_length=15, max_length=150)
    year:int  = Field(le = 2024)
    rating:float = Field(ge=1, le=10)
    category: str = Field(min_length=3, max_length=20)
    class Config:
        json_schema_extra = {
            "example":
                {
                "id":1,
                "title": "Mi película",
                "overview":"Descripción de la película",
                "year": 2022,
                "rating":9.8,
                "category": "Acción"
			}
            
		}
    

movies = [
    {
		"id": 1,
		"title": "Avatar",
		"overview": "En un exuberante planeta llamado Pandora viven los Na'vi, seres que ...",
		"year": "2009",
		"rating": 7.8,
		"category": "Acción"
	},
    {
		"id": 2,
		"title": "Avatar",
		"overview": "En un exuberante planeta llamado Pandora viven los Na'vi, seres que ...",
		"year": "2009",
		"rating": 7.8,
		"category": "Terror"
	}
]

@app.get("/", tags = ["home"])
def masage():
    return HTMLResponse("<h1>Hello World!</h1>")

@app.post("/login", tags=["auth"])
def login(user: User):
    if user.email == "admin@gmail.com" and user.password == "admin":
        token: str = create_token(user.dict()).decode('utf-8')
        return JSONResponse(status_code = 200, content=token)
    return JSONResponse(content=["El correo o el usuario no son correctos"],status_code = 404)

@app.get('/movies', tags=['movies'], response_model=List[Movie], status_code=200, dependencies=[Depends(JWTBearer())])
def get_movies() -> List[Movie]:
    return JSONResponse(status_code=200, content=movies)

@app.get("/movies/{id}", tags = ["movies"], response_model = Movie)
def get_movie(id: int = Path(ge=1, le=2000)) -> Movie:
    for item in movies:
        if item["id"] == id:
            return JSONResponse(content=item)
        
    return JSONResponse(content=["El id no existe"],status_code = 404)


"""@app.get("/movies/", tags = ["movies"])
def get_movies_by_category(category: str):
    movies_by_category = []
    for movie in movies:
        if(movie["category"]== category):
            movies_by_category.append(movie)
    return movies_by_category"""

@app.get('/movies/', tags=['movies'], response_model = List[Movie])
def get_movies_by_category(category: str = Query(min_length=5, max_length=15)) -> List[Movie]:
    data = [ item for item in movies if item['category'] == category ]
    return JSONResponse(content=data)

@app.post("/movies/", tags = ["movies"], response_model = dict, status_code = 201)
def create_movie(movie: Movie) -> dict:
    movies.append(movie.model_dump())
    return JSONResponse(content={"message": "Se ha registrado la película"}, status_code = 201)

@app.put("/movies/{id}", tags = ["movies"], response_model = dict, status_code = 200)
def update_movie(id: int, movie:Movie)  -> dict:
    for item in movies:
        if item["id"] == id:
            item["title"] = movie.title
            item["overview"] = movie.overview
            item["year"] = movie.year
            item["rating"] = movie.rating
            item["category"] = movie.category
            return JSONResponse(content={"message": "Se ha modificado la película"}, status_code = 200)
            

@app.delete("/movies/{id}", tags = ["movies"], response_model = dict, status_code = 200)
def delete_movie(id: int) -> dict:
    for item in movies:
        if item["id"] == id:
            movies.remove(item)
            return JSONResponse(content={"message": "Se ha eliminado la película"}, status_code = 200)