from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(base_dir / "templates")

app.mount(
    "/static",
    StaticFiles(directory=base_dir / "static"),
    name="static"
)
class InputRequest(BaseModel):
    text: str

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

