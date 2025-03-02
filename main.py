# Import các module cần thiết
from libary_local import *
from model_local import *
from tokenizers_local import *
from load_model_local import *
from fast_api_local import app


# ✅ Lấy PORT từ biến môi trường (Railway tự động set PORT)
PORT = int(os.environ.get("PORT", 8000))

# ✅ Chạy FastAPI (Railway tự động chạy)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
