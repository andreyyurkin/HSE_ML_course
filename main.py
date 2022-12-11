import pickle
import warnings
from typing import List
import pandas as pd
import pydantic
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

warnings.filterwarnings("ignore")

with open('inf.pickle', 'rb') as f:
    info = pickle.load(f)


def get_predict(predict_line: list) -> List[float]:
    data = info['pr']
    my_model = info['pipe']
    data.loc[0] = predict_line
    return my_model.predict(data)[0]


app = FastAPI()


class Item(pydantic.BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    max_power: float
    seats: int
    mileage_kmpl: float
    engine_CC: int
    torque_new: float
    max_torque_rpm: float


class Items(pydantic.BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> List[float]:
    values = list(item.dict().values())
    return get_predict(values)


@app.post("/upload_predict_items", response_class=FileResponse)
async def predict_items_csv(file: UploadFile):
    data = pd.read_csv(file.file)
    my_model = info['pipe']
    data = pd.DataFrame(data)
    data = data.iloc[:, 1:]
    data['preds'] = list(my_model.predict(data))
    data.to_csv('export.csv')
    return 'export.csv'
