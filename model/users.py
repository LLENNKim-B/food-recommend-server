from pydantic import BaseModel

class User(BaseModel):
    username: str
    email: str
    password: str
    birthdate: str
    gender: str
    height: float
    weight: float
    diet_type: str
    disease: str = None
    activity_id :int

class UserLogin(BaseModel):
    email: str
    password: str