from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_match

class PredictRequest(BaseModel):
    p1: str
    p2: str

def register_api(app: FastAPI, cache):
    @app.post("/predict")
    async def predict(req: PredictRequest):
        odds_a, odds_b = await cache.get_odds_for_match(req.p1, req.p2)

        odds_a = odds_a if odds_a is not None else 2.0
        odds_b = odds_b if odds_b is not None else 1.8

        prob, decision = predict_match(req.p1, req.p2, odds_a, odds_b)

        return {
            "p1": req.p1,
            "p2": req.p2,
            "prob_a": prob,
            "odds_a": odds_a,
            "odds_b": odds_b,
        }

    @app.get("/matches")
    async def get_matches():
        matches = await cache.get_all_matches()
        return [
            {"p1": p1, "p2": p2, "odds_a": odds_a, "odds_b": odds_b}
            for (p1, p2), (odds_a, odds_b) in matches.items()
        ]