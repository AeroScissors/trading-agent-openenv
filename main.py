from fastapi import FastAPI
from api.routes import router
import os

# ✅ correct import
from generate_data import generate_all

app = FastAPI()

app.include_router(router)

@app.on_event("startup")
def startup_event():
    print("🚀 Starting API...")

    data_path = "data/prices/aapl.csv"

    if not os.path.exists(data_path):
        print("📦 Generating data...")
        generate_all()   # ✅ correct function
        print("✅ Data ready!")
    else:
        print("✅ Data already exists")

@app.get("/")
def root():
    return {"status": "running"}