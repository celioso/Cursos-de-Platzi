from fastapi import APIRouter
from fastapi import Depends, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from config.database import Session
from models.movie import Movie as MovieModel
from fastapi.encoders import jsonable_encoder
from middlewares.jwt_bearer import JWTBearer
from services.movie import MovieService

movie_router = APIRouter()

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

@movie_router.get('/movies', tags=['movies'], response_model=List[Movie], status_code=200, dependencies=[Depends(JWTBearer())])
def get_movies() -> List[Movie]:
    db = Session()
    result = MovieService(db).get_movies()
    return JSONResponse(status_code=200, content=jsonable_encoder(result))

@movie_router.get("/movies/{id}", tags = ["movies"], response_model = Movie)
def get_movie(id: int = Path(ge=1, le=2000)) -> Movie:
    db = Session()
    result = MovieService(db).get_movie(id)
    if not result:
        return JSONResponse(status_code=404, content = {"Message":"No encontrado"})     
    return JSONResponse(content=jsonable_encoder(result),status_code = 200)


"""@movie_router.get("/movies/", tags = ["movies"])
def get_movies_by_category(category: str):
    movies_by_category = []
    for movie in movies:
        if(movie["category"]== category):
            movies_by_category.append(movie)
    return movies_by_category"""

@movie_router.get('/movies/', tags=['movies'], response_model = List[Movie])
def get_movies_by_category(category: str = Query(min_length=1, max_length=15)) -> List[Movie]:
    db = Session()
    result = db.query(MovieModel).filter(MovieModel.category == category).all()
    if not result:
        return JSONResponse(status_code=404, content = {"Message":"La categoría no existe."})  
    return JSONResponse(status_code = 200, content=jsonable_encoder(result))

@movie_router.post("/movies/", tags = ["movies"], response_model = dict, status_code = 201)
def create_movie(movie: Movie) -> dict:
    db = Session()
    new_movie = MovieModel(**movie.dict())
    db.add(new_movie)
    db.commit()
    return JSONResponse(content={"message": "Se ha registrado la película"}, status_code = 201)

@movie_router.put("/movies/{id}", tags = ["movies"], response_model = dict, status_code = 200)
def update_movie(id: int, movie:Movie)  -> dict:
    db = Session()
    result = db.query(MovieModel).filter(MovieModel.id == id).first()
    if not result:
        return JSONResponse(status_code=404, content = {"Message":"El id no existe."}) 
    result.title = movie.title
    result.overview = movie.overview
    result.year = movie.year
    result.rating = movie.rating
    result.category = movie.category
    db.commit()
    return JSONResponse(content={"message": "Se ha modificado la película"}, status_code = 200)
            

@movie_router.delete("/movies/{id}", tags = ["movies"], response_model = dict, status_code = 200)
def delete_movie(id: int) -> dict:
    db = Session()
    result = db.query(MovieModel).filter(MovieModel.id == id).first()
    if not result:
        return JSONResponse(status_code=404, content = {"Message":"El id no existe."}) 
    db.delete(result)
    db.commit()
    return JSONResponse(content={"message": "Se ha eliminado la película"}, status_code = 200)