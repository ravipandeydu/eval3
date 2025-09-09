from fastapi import FastAPI
import uvicorn
from routes import router

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


# System Status
@app.get("/health")
def health_check():
    return {"status": "healthy"}


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
