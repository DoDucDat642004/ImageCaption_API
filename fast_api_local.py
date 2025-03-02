
import uvicorn
import os
import io
import threading
import requests
import time
from PIL import Image
from libary_local import *
from model_local import *
from tokenizers_local import *
from load_model_local import *

# Khởi tạo FastAPI
app = FastAPI()

# Trang chính (upload ảnh)
@app.get("/", response_class=HTMLResponse)
async def homepage():
    with open("form_local.html", "r") as f:
        return f.read()

# API dự đoán caption ảnh
@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: str = Form("en")):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image_transform(image).to(device)
        caption = model.predict_caption(image, tokenizers[lang_id[lang]], lang_id=lang_id[lang], mode='greedy')
        return {"caption": caption, "language": lang}
    except Exception as e:
        return {"error": str(e)}
