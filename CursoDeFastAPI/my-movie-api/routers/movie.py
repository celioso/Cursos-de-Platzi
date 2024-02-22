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
from schemas.movie import Movie

movie_router = APIRouter()

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
    result = MovieService(db).get_movies_by_category(category)
    if not result:
        return JSONResponse(status_code=404, content = {"Message":"La categoría no existe."})  
    return JSONResponse(status_code = 200, content=jsonable_encoder(result))

@movie_router.post("/movies/", tags = ["movies"], response_model = dict, status_code = 201)
def create_movie(movie: Movie) -> dict:
    db = Session()
    MovieService(db).create_movie(movie)
    return JSONResponse(content={"message": "Se ha registrado la película"}, status_code = 201)

@movie_router.put("/movies/{id}", tags = ["movies"], response_model = dict, status_code = 200)
def update_movie(id: int, movie:Movie)  -> dict:
    db = Session()
    result = MovieService(db).get_movie(id)
    if not result:
        return JSONResponse(status_code=404, content = {"Message":"El id no existe."}) 
    MovieService(db).update_movie(id, movie)
    return JSONResponse(content={"message": "Se ha modificado la película"}, status_code = 200)
            

@movie_router.delete("/movies/{id}", tags = ["movies"], response_model = dict, status_code = 200)
def delete_movie(id: int) -> dict:
    db = Session()
    result : MovieModel = db.query(MovieModel).filter(MovieModel.id == id).first()
    if not result:
        return JSONResponse(status_code=404, content = {"Message":"El id no existe."}) 
    MovieService(db).delate_movie(id)
    return JSONResponse(content={"message": "Se ha eliminado la película"}, status_code = 200)