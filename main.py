from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from api.routes import router
import os

from generate_data import generate_all

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.on_event("startup")
def startup_event():
    print(" Starting API...")
    data_path = "data/prices/aapl.csv"
    if not os.path.exists(data_path):
        print("Generating data...")
        generate_all()
        print(" Data ready!")
    else:
        print(" Data already exists")

if os.path.exists("frontend/static"):
    app.mount("/static", StaticFiles(directory="frontend/static"), name="static")