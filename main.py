# Import các module cần thiết
from libary_local import *
from model_local import *
from tokenizers_local import *
from load_model_local import *
from fast_api_local import *


# ✅ Lấy PORT từ biến môi trường (Railway tự động set PORT)
PORT = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    print(f"🚀 Running on http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
