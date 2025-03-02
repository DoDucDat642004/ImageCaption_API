import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
    except ModuleNotFoundError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        __import__(package)

# Danh sách các thư viện cần kiểm tra và cài đặt
dependencies = [
    "uvicorn", "fastapi", "numpy", "torch", "torchvision", "transformers", "timm", 
    "PIL", "requests", "pyngrok"
]

# Kiểm tra và cài đặt
for package in dependencies:
    install_and_import(package)

import uvicorn
from fastapi import FastAPI, Form, File, UploadFile
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer
from timm import create_model
from PIL import Image
import math
from fastapi.responses import HTMLResponse
import io
import requests
from pyngrok import ngrok
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
