from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SLM(BaseModel):
    slm_name: str
    slm_host: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/add_slm")
async def add_slm(slm: SLM):
    slm_name = slm.slm_name
    slm_host = slm.slm_host
    print(f"Adding SLM {slm_name} with host {slm_host}")
    return {"status": "ok"}
