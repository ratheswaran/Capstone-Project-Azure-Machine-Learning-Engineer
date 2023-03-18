import json
import joblib
import pandas as pd
from azureml.core.model import Model


def init():
    global model
    model_path = Model.get_model_path("titanic_pred_model")
    model = joblib.load(model_path)

def run(data):
    try:
        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        # Run inference
        prediction = model.predict(data)
        return prediction.tolist()

    except Exception as e:
        error = str(e)
        return error