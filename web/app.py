import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model import predict_match

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_app(cache):
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
    register_handlers(app, cache)
    return app

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

def register_handlers(app, cache):
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        matches_raw = await cache.get_all_matches()

        matches = []
        for key, value in matches_raw.items():
            print(key)
            p1, p2 = key
            print(1)
            odds_a, odds_b = value
            matches.append({"p1": p1, "p2": p2, "odds_a": odds_a, "odds_b": odds_b})
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"matches": matches}
        )

    @app.post("/predict", response_class=HTMLResponse)
    async def predict(request: Request):
        form = await request.form()
        p1 = form.get("p1")
        p2 = form.get("p2")

        odds_a, odds_b = await cache.get_odds_for_match(p1, p2)
        odds_a = odds_a or 2.0
        odds_b = odds_b or 1.8

        try:
            prob = predict_match(p1, p2, odds_a, odds_b)
            result = {
                "p1": p1,
                "p2": p2,
                "prob": round(prob, 3),
                "odds_a": odds_a,
                "odds_b": odds_b,
            }
        except Exception as e:
            result = {"error": str(e)}

        matches_raw = await cache.get_all_matches()
        matches = []
        for key, value in matches_raw.items():
            p1_match, p2_match = key
            odds_a_match, odds_b_match = value
            matches.append({
                "p1": p1_match,
                "p2": p2_match,
                "odds_a": odds_a_match,
                "odds_b": odds_b_match
            })

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"matches": matches, "result": result}
        )