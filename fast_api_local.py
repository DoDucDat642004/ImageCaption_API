
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

# ✅ Khởi tạo FastAPI
app = FastAPI()

# ✅ Trang chính (upload ảnh)
@app.get("/", response_class=HTMLResponse)
async def homepage():
    with open("form_local.html", "r") as f:
        return f.read()

# ✅ API dự đoán caption ảnh
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

# ✅ Lấy PORT từ biến môi trường (Render tự động set)
PORT = int(os.environ.get("PORT", 8000))
SERVER_URL = os.environ.get("PUBLIC_URL", f"https://fastapi-app.onrender.com")

# ✅ Hàm gửi request giữ server luôn hoạt động (tránh auto sleep)
def keep_awake():
    while True:
        try:
            response = requests.get(SERVER_URL)
            print(f"🔄 Ping {SERVER_URL}: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Lỗi giữ server online: {e}")
        time.sleep(600)  # Ping mỗi 10 phút

# ✅ Chạy luồng giữ server không bị ngủ
keep_awake_thread = threading.Thread(target=keep_awake)
keep_awake_thread.daemon = True
keep_awake_thread.start()

# ✅ Chạy server trên Render
if __name__ == "__main__":
    print(f"🚀 Server chạy tại {SERVER_URL}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
