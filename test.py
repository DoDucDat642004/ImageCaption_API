from libary_local import *

with open("public_url.txt", "r", encoding="utf-8") as file:
    url = file.read()

url = f"{url}/predict"
files = {"file": open("/Users/dodat/Downloads/anh-thien-nhien-dep-3d-22.jpg", "rb")}
data = {"lang": "en"}

response = requests.post(url, files=files, data=data)

print(response.json())  # Kiểm tra kết quả
