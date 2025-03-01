"""安装配置文件"""

from setuptools import setup, find_packages

setup(
    name="face",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "mediapipe>=0.10.0",
        "click>=8.0.0",
        "tqdm>=4.65.0",
        "ultralytics>=8.0.0",
        "insightface>=0.7.0",
        "onnxruntime>=1.15.0",
    ],
    entry_points={
        "console_scripts": [
            "face=src.main:main",
        ],
    },
    python_requires=">=3.8",
) 