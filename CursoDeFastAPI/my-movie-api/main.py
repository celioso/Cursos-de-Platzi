from fastapi import FastAPI, Body, Path, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

app = FastAPI()
app.title = "Mi aplicación con FastAPI"
app.version = "0.0.1"

class Movie(BaseModel):
    id:Optional[int] = None
    title:str = Field(min_length=5, max_length=15)
    overview: str = Field(min_length=15, max_length=50)
    year:int  = Field(le = 2022)
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


@app.get("/movies", tags = ["movies"], response_model = List[Movie])
def get_movies() -> List[Movie]:
    return JSONResponse(content=movies)

@app.get("/movies/{id}", tags = ["movies"], response_model = Movie)
def get_movie(id: int = Path(ge=1, le=2000)) -> Movie:
    for item in movies:
        if item["id"] == id:
            return JSONResponse(content=item)
        
    return JSONResponse(content=["El id no existe"])


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

@app.post("/movies/", tags = ["movies"], response_model = dict)
def create_movie(movie: Movie) -> dict:
    movies.append(movie.model_dump())
    return JSONResponse(content={"message": "Se ha registrado la película"})

@app.put("/movies/{id}", tags = ["movies"], response_model = dict)
def update_movie(id: int, movie:Movie)  -> dict:
    for item in movies:
        if item["id"] == id:
            item["title"] = movie.title
            item["overview"] = movie.overview
            item["year"] = movie.year
            item["rating"] = movie.rating
            item["category"] = movie.category
            return JSONResponse(content={"message": "Se ha modificado la película"})
            

@app.delete("/movies/{id}", tags = ["movies"], response_model = dict)
def delete_movie(id: int) -> dict:
    for item in movies:
        if item["id"] == id:
            movies.remove(item)
            return JSONResponse(content={"message": "Se ha eliminado la película"})