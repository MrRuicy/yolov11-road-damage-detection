import sys
import subprocess

print("Python version:", sys.version)
print("Python executable:", sys.executable)

# 检查 OpenCV 是否安装
try:
    import cv2
    print("OpenCV version:", cv2.__version__)
except ModuleNotFoundError:
    print("cv2 module not found, trying to install...")
    subprocess.run(["pip", "install", "opencv-python-headless"])
