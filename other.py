import cv2
import dlib
import numpy as np
import os
import re
from scipy.spatial import distance as dist
from collections import defaultdict

# 目录设置
input_folder = "H:/fenjie/finspt"   # 输入文件夹
output_folder = "H:/fenjie/tp"      # 输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 加载 Dlib 人脸检测器 & 68 关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("E:/ai/fenjie/shape_predictor_68_face_landmarks.dat")

# 设置参数
yaw_threshold = 50  # 左右偏头最大角度
pitch_threshold = 50  # 上下俯仰最大角度
eye_aspect_ratio_threshold = 0.37  # 眼睛闭合阈值

# 统计数据
total_images = 0
kept_images = 0
removed_images = 0
no_face_images = 0  # 统计未检测到人脸的图片

def extract_prefix(filename):
    """ 提取文件名前缀（如 1_frame_00030 -> 1）"""
    match = re.match(r"(\d+)_frame_\d+", filename)
    return match.group(1) if match else None

def eye_aspect_ratio(eye):
    """ 计算眼睛纵横比（EAR），用于检测闭眼 """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def is_face_angle_valid(landmarks):
    """ 判断人脸角度是否符合要求 """
    nose = landmarks[30]     # 鼻尖
    left_eye = landmarks[36]  # 左眼角
    right_eye = landmarks[45]  # 右眼角
    chin = landmarks[8]       # 下巴

    # 计算 Yaw（左右偏头角度）
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    yaw = np.degrees(np.arctan2(delta_y, delta_x))

    # 计算 Pitch（上下低头仰头角度）
    delta_nose = nose[1] - chin[1]
    delta_eyes = np.linalg.norm(right_eye - left_eye)
    pitch = np.degrees(np.arctan2(delta_nose, delta_eyes))

    return abs(yaw) <= yaw_threshold and abs(pitch) <= pitch_threshold

# 处理每张图片
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    total_images += 1

    # **获取前缀**
    prefix = extract_prefix(filename)
    if prefix is None:
        removed_images += 1
        continue  # 没有匹配到前缀的文件跳过

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. 人脸检测
    faces = detector(gray)

    if len(faces) == 0:
        no_face_images += 1
        removed_images += 1
        continue  # 没检测到人脸，直接舍弃

    # 2. 遍历所有检测到的人脸
    face_ok = False
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])

        # 3. 眼睛闭合检测
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)

        if ear_left < eye_aspect_ratio_threshold and ear_right < eye_aspect_ratio_threshold:
            continue  # 眼睛闭合，跳过

        # 4. 头部角度检测
        if not is_face_angle_valid(landmarks):
            continue  # 头部角度不符合要求，跳过

        face_ok = True
        break  # 只要有一个合格的人脸，就保留

    if face_ok:
        # 直接保存当前图片
        cv2.imwrite(os.path.join(output_folder, filename), image)
        kept_images += 1
    else:
        removed_images += 1

# 统计结果
print(f"📷 总共处理图片: {total_images}")
print(f"❌ 未检测到人脸: {no_face_images}")
print(f"✅ 保留图片: {kept_images}")
print(f"❌ 去除图片: {removed_images}")
