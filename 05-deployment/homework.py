# basically do the following but we need uvicorn (i.e wsl/linux)
# And this needs to be in a script along with the next cell
import pickle
from fastapi import FastAPI
import uvicorn
import requests
from typing import Dict, Any

app = FastAPI(title="customer-churn-prediction")

input_file = "pipeline_v1.bin"

with open(input_file, "rb") as f_in:
    dv, model = pickle.load(f_in)


def predict_single(customer):
    X = dv.transform([customer])
    result = model.predict_proba(X)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prob = predict_single(customer)

    return {"churn_probability": prob, "churn": bool(prob >= 0.5)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

    url = "http://localhost:9696/predict"
    client = {
        "lead_source": "organic_search",
        "number_of_courses_viewed": 4,
        "annual_income": 80304.0,
    }
    print(requests.post(url, json=client).json())
