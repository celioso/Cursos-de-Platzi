from pydantic import BaseModel, Field
from typing import Optional, List

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

