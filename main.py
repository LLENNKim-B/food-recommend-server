# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import recommendation,users
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include router
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(recommendation.router, prefix="/recommendation", tags=["recommendation"])

@app.get("/")
async def root():
    return {"message": "Hello, Food Recommendation API is running 🚀"}
