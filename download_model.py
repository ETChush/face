"""下载 MediaPipe Face Landmarker 模型文件"""

import urllib.request
import os

def download_face_landmarker_model():
    """下载 face_landmarker.task 模型文件"""
    
    # 模型文件 URL（使用 Google Cloud Storage）
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    
    # 目标路径
    model_dir = "models"
    model_path = os.path.join(model_dir, "face_landmarker.task")
    
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"正在下载 face_landmarker.task...")
    print(f"从: {model_url}")
    print(f"到: {model_path}")
    
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"✓ 模型文件下载成功: {model_path}")
        print(f"文件大小: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"✗ 下载失败: {str(e)}")
        print("\n请手动下载模型文件:")
        print("1. 访问: https://github.com/google/mediapipe/releases")
        print("2. 下载最新的 face_landmarker.task 文件")
        print(f"3. 将文件放置到: {model_path}")

if __name__ == "__main__":
    download_face_landmarker_model()