import cv2
import os
import shutil
import numpy as np

def variance_of_laplacian(image):
    """计算图像的拉普拉斯变换方差，值越小表示图像越模糊"""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def filter_clear_images(input_folder, output_folder, threshold=100):
    """筛选清晰的图片，模糊的图片会被丢弃"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_images = 0
    kept_images = 0
    removed_images = 0

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"无法读取文件: {filename}，跳过")
            continue

        total_images += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        laplacian_var = variance_of_laplacian(gray)  # 计算清晰度

        if laplacian_var >= threshold:
            shutil.copy(image_path, os.path.join(output_folder, filename))
            kept_images += 1
        else:
            removed_images += 1

    print(f"总共处理图片: {total_images}")
    print(f"保留图片: {kept_images}")
    print(f"去除图片: {removed_images}")

# 运行
input_folder = "H:/fenjie/spt"  # 你的输入文件夹
output_folder = "H:/fenjie/finspt"  # 你的输出文件夹（保存清晰的图片）

filter_clear_images(input_folder, output_folder, threshold=50)
