import cv2
import dlib
import numpy as np
import os
import re
from hashlib import md5
from scipy.spatial import distance as dist
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict

# **目录设置**
input_folder = "H:/fenjie/finspt"   # 输入文件夹
output_folder = "H:/fenjie/tp"      # 输出文件夹
os.makedirs(output_folder, exist_ok=True)

# **加载 Dlib 人脸检测器 & 68 关键点预测器**
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("E:/ai/fenjie/shape_predictor_68_face_landmarks.dat")

# **参数设置（适当放宽限制，提高检测率）**
blur_threshold = 50  # 降低模糊阈值（Laplacian 变换值越高越清晰）
yaw_threshold = 60  # 左右偏头最大角度（越大越宽松）
pitch_threshold = 60  # 上下俯仰最大角度（越大越宽松）
eye_aspect_ratio_threshold = 0.3  # 眼睛闭合阈值（适当放宽）
similarity_threshold = 0.8  # 图片相似度阈值（0~1，越低越严格）

# **统计数据**
total_images = 0
kept_images = 0
removed_images = 0
removal_reasons = defaultdict(int)  # 记录去除原因

# **用于存储已处理图片的哈希值**
processed_hashes = set()

def extract_prefix(filename):
    """ 提取文件名前缀（如 1_frame_00030 -> 1）"""
    match = re.match(r"(\d+)_frame_\d+", filename)
    return match.group(1) if match else None

def calculate_blur(image):
    """ 计算图像的清晰度（Laplacian 变换）"""
    return cv2.Laplacian(image, cv2.CV_64F).var()

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

def dhash(image, size=8):
    """ 计算图像的 dHash（感知哈希），用于检测相似图片 """
    resized = cv2.resize(image, (size + 1, size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hash_histogram(hist):
    """ 计算直方图的 MD5 哈希 """
    return md5(hist.tobytes()).hexdigest()

def is_similar(image, img_hist):
    """ 判断图片是否与已处理图片相似 """
    img_dhash = dhash(image)
    img_hist_hash = hash_histogram(img_hist)

    if img_dhash in processed_hashes or img_hist_hash in processed_hashes:
        return True  # 说明已经有相似图片
    return False

# **处理每张图片**
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
        removal_reasons["文件名不符合规则"] += 1
        continue  # 没有匹配到前缀的文件跳过

    # **转换为灰度图**
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # **1. 清晰度检测**
    blur_value = calculate_blur(gray)
    if blur_value < blur_threshold:
        removed_images += 1
        removal_reasons["图片过于模糊"] += 1
        continue  # 过于模糊，跳过

    # **2. 人脸检测**
    faces = detector(gray)

    if len(faces) == 0:
        removed_images += 1
        removal_reasons["未检测到人脸"] += 1
        continue  # 没检测到人脸，直接舍弃

    # **3. 遍历所有检测到的人脸**
    face_ok = False
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])

        # **4. 眼睛闭合检测**
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)

        if ear_left < eye_aspect_ratio_threshold and ear_right < eye_aspect_ratio_threshold:
            removal_reasons["眼睛闭合"] += 1
            continue  # 眼睛闭合，跳过

        # **5. 头部角度检测**
        if not is_face_angle_valid(landmarks):
            removal_reasons["头部角度不符合要求"] += 1
            continue  # 头部角度不符合要求，跳过

        face_ok = True
        break  # 只要有一个合格的人脸，就保留

    if face_ok:
        # **6. 相似度检测**
        img_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        img_hist = cv2.normalize(img_hist, img_hist).flatten()

        if is_similar(gray, img_hist):
            removed_images += 1
            removal_reasons["图片过于相似"] += 1
            continue

        # **存入已处理图片的哈希**
        processed_hashes.add(dhash(gray))
        processed_hashes.add(hash_histogram(img_hist))

        # **保存当前图片**
        cv2.imwrite(os.path.join(output_folder, filename), image)
        kept_images += 1
    else:
        removed_images += 1

# **统计结果**
print(f"📷 总共处理图片: {total_images}")
print(f"✅ 保留图片: {kept_images}")
print(f"❌ 去除图片: {removed_images}")
print("\n📊 **去除原因统计**:")
for reason, count in removal_reasons.items():
    print(f"  - {reason}: {count} 张")





# import cv2
# import dlib
# import numpy as np
# import os
# import re
# from scipy.spatial import distance as dist
# from skimage.metrics import structural_similarity as ssim
# from collections import defaultdict
#
# # 目录设置
# input_folder = "H:/fenjie/finspt"   # 输入文件夹
# output_folder = "H:/fenjie/tp"      # 输出文件夹
# os.makedirs(output_folder, exist_ok=True)
#
# # 加载 Dlib 人脸检测器 & 68 关键点预测器
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("E:/ai/fenjie/shape_predictor_68_face_landmarks.dat")
#
# # 参数设置
# blur_threshold = 30  # 清晰度阈值（越高越严格）
# yaw_threshold = 50     # 左右偏头最大角度（改为 30° 以内）
# pitch_threshold = 50   # 上下俯仰最大角度
# eye_aspect_ratio_threshold = 0.25  # 眼睛闭合阈值（越低越严格）
# similarity_threshold = 0.8  # 相似度阈值（越低越严格）
#
# # 记录最佳图片（按前缀分组）
# best_images = defaultdict(lambda: None)
# best_image_ssim = defaultdict(lambda: -1)
#
# # 统计数据
# total_images = 0
# kept_images = 0
# removed_images = 0
# removal_reasons = defaultdict(int)  # 记录去除原因的统计
#
# def extract_prefix(filename):
#     """ 提取文件名前缀（如 1_frame_00030 -> 1）"""
#     match = re.match(r"(\d+)_frame_\d+", filename)
#     return match.group(1) if match else None
#
# def calculate_blur(image):
#     """ 计算图像的清晰度（Laplacian 模糊度）"""
#     return cv2.Laplacian(image, cv2.CV_64F).var()
#
# def eye_aspect_ratio(eye):
#     """ 计算眼睛纵横比（EAR），用于检测闭眼 """
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)
#
# def face_orientation(landmarks):
#     """ 计算面部偏航角（Yaw）和俯仰角（Pitch）"""
#     # 关键点索引：
#     left_eye = landmarks[36:42]
#     right_eye = landmarks[42:48]
#     nose = landmarks[27]  # 鼻梁
#     chin = landmarks[8]   # 下巴
#     left_cheek = landmarks[0]  # 左脸颊
#     right_cheek = landmarks[16]  # 右脸颊
#
#     # 计算偏航角（Yaw）: 左右偏头角度
#     yaw = abs(right_cheek[0] - left_cheek[0]) / abs(nose[0] - chin[0]) * 100
#
#     # 计算俯仰角（Pitch）：上下偏头角度
#     pitch = abs(nose[1] - chin[1]) / abs(left_eye[1] - right_eye[1]) * 100
#
#     return yaw, pitch
#
# def compare_images(image1, image2):
#     """ 比较两张图片的相似度，确保尺寸一致 """
#     gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#
#     # 统一尺寸，避免尺寸不匹配
#     h, w = gray1.shape
#     gray2 = cv2.resize(gray2, (w, h))
#
#     score, _ = ssim(gray1, gray2, full=True)
#     return score
#
# # 处理每张图片
# for filename in os.listdir(input_folder):
#     img_path = os.path.join(input_folder, filename)
#     image = cv2.imread(img_path)
#
#     if image is None:
#         removal_reasons["无法读取图像"] += 1
#         continue
#
#     total_images += 1
#
#     # **获取前缀**
#     prefix = extract_prefix(filename)
#     if prefix is None:
#         removal_reasons["无效的文件名前缀"] += 1
#         removed_images += 1
#         continue  # 没有匹配到前缀的文件跳过
#
#     # 转换为灰度图
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 1. 清晰度检测
#     blur_value = calculate_blur(gray)
#     if blur_value < blur_threshold:
#         removal_reasons["图像过于模糊"] += 1
#         removed_images += 1
#         continue  # 过于模糊，跳过
#
#     # 2. 人脸检测
#     faces = detector(gray)
#     if len(faces) == 0:
#         removal_reasons["未检测到人脸"] += 1
#         removed_images += 1
#         continue  # 没检测到人脸，跳过
#
#     # 3. 遍历所有检测到的人脸
#     face_ok = False
#     for face in faces:
#         landmarks = predictor(gray, face)
#         if landmarks is None:
#             continue  # 关键点检测失败，跳过
#
#         landmarks = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])
#
#         # 4. 眼睛闭合检测
#         left_eye = landmarks[36:42]
#         right_eye = landmarks[42:48]
#         ear_left = eye_aspect_ratio(left_eye)
#         ear_right = eye_aspect_ratio(right_eye)
#
#         if ear_left < eye_aspect_ratio_threshold and ear_right < eye_aspect_ratio_threshold:
#             removal_reasons["眼睛闭合"] += 1
#             continue  # 眼睛闭合，跳过
#
#         # 5. 计算头部偏移角度
#         yaw, pitch = face_orientation(landmarks)
#         if yaw > yaw_threshold or pitch > pitch_threshold:
#             removal_reasons["头部偏移过大"] += 1
#             continue  # 头部角度不合适，跳过
#
#         face_ok = True
#         break  # 只要有一个合格的人脸，就保留
#
#     if face_ok:
#         best_image = best_images[prefix]
#
#         if best_image is not None:
#             similarity = compare_images(best_image, image)
#             if similarity > similarity_threshold:
#                 removal_reasons["重复照片"] += 1
#                 removed_images += 1
#                 continue  # 如果相似度高于阈值，则跳过
#
#         # 保存当前图片
#         cv2.imwrite(os.path.join(output_folder, filename), image)
#         kept_images += 1
#         best_images[prefix] = image
#
# # 统计结果
# print(f"📷 总共处理图片: {total_images}")
# print(f"✅ 保留图片: {kept_images}")
# print(f"❌ 去除图片: {removed_images}")
# print("去除原因统计:", dict(removal_reasons))







# import cv2
# import dlib
# import numpy as np
# import os
# from scipy.spatial import distance as dist
# from scipy.spatial.transform import Rotation as R
# from skimage.metrics import structural_similarity as ssim
#
# # 目录设置
# input_folder = "H:/fenjie/finspt"   # 输入文件夹
# output_folder = "H:/fenjie/tp"  # 输出文件夹
# os.makedirs(output_folder, exist_ok=True)
#
# # 加载 Dlib 人脸检测器 & 68 关键点预测器
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("E:/ai/fenjie/shape_predictor_68_face_landmarks.dat")
#   # 确保这个模型文件存在
#
# # 设置参数
# blur_threshold = 100  # 清晰度阈值（越高越严格）
# yaw_threshold = 100     # 左右偏头最大角度
# pitch_threshold = 100   # 上下俯仰最大角度
# eye_aspect_ratio_threshold = 0.3  # 眼睛闭合阈值，改为 0.25 更严格，改为 0.15 更宽松。
# similarity_threshold = 0.8  # 相似度阈值（0.8以上认为是重复的照片越低越严格，越高越宽松）
#
# # 初始化变量
# total_images = 0
# kept_images = 0
# removed_images = 0
# best_image = None
# best_image_ssim = -1  # 初始最好的相似度为 -1
#
# def calculate_blur(image):
#     """ 计算图像的清晰度（Laplacian 模糊度）"""
#     return cv2.Laplacian(image, cv2.CV_64F).var()
#
# def eye_aspect_ratio(eye):
#     """ 计算眼睛纵横比（EAR），用于检测闭眼 """
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])
#     C = dist.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)
#
# def is_face_angle_valid(landmarks):
#     """ 判断人脸角度是否符合要求 """
#     nose = landmarks[30]    # 鼻尖
#     left_eye = landmarks[36]  # 左眼角
#     right_eye = landmarks[45]  # 右眼角
#     chin = landmarks[8]      # 下巴
#
#     # 计算 Yaw（左右偏头角度）
#     delta_x = right_eye[0] - left_eye[0]
#     delta_y = right_eye[1] - left_eye[1]
#     yaw = np.degrees(np.arctan2(delta_y, delta_x))
#
#     # 计算 Pitch（上下低头仰头角度）
#     delta_nose = nose[1] - chin[1]
#     delta_eyes = np.linalg.norm(right_eye - left_eye)
#     pitch = np.degrees(np.arctan2(delta_nose, delta_eyes))
#
#     return abs(yaw) <= yaw_threshold and abs(pitch) <= pitch_threshold
#
# def compare_images(image1, image2):
#     """ 比较两张图片的相似度，使用 SSIM """
#     gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#     score, _ = ssim(gray1, gray2, full=True)
#     return score
#
# # 处理每张图片
# for filename in os.listdir(input_folder):
#     img_path = os.path.join(input_folder, filename)
#     image = cv2.imread(img_path)
#
#     if image is None:
#         continue
#
#     total_images += 1
#
#     # 转换为灰度图
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 1. 清晰度检测
#     blur_value = calculate_blur(gray)
#     if blur_value < blur_threshold:
#         removed_images += 1
#         continue  # 过于模糊，跳过
#
#     # 2. 人脸检测
#     faces = detector(gray)
#     if len(faces) == 0:
#         removed_images += 1
#         continue  # 没检测到人脸，跳过
#
#     # 3. 遍历所有检测到的人脸
#     face_ok = False
#     for face in faces:
#         landmarks = predictor(gray, face)
#         landmarks = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])
#
#         # 4. 眼睛闭合检测
#         left_eye = landmarks[36:42]
#         right_eye = landmarks[42:48]
#         ear_left = eye_aspect_ratio(left_eye)
#         ear_right = eye_aspect_ratio(right_eye)
#
#         if ear_left < eye_aspect_ratio_threshold and ear_right < eye_aspect_ratio_threshold:
#             continue  # 眼睛闭合，跳过
#
#         # 5. 头部角度检测
#         if not is_face_angle_valid(landmarks):
#             continue  # 头部角度不符合要求，跳过
#
#         face_ok = True
#         break  # 只要有一个合格的人脸，就保留
#
#     if face_ok:
#         # 如果有“最好的图片”，计算相似度
#         if best_image is not None:
#             similarity = compare_images(best_image, image)
#             if similarity > similarity_threshold:
#                 removed_images += 1
#                 continue  # 如果相似度高于阈值，则跳过
#         else:
#             similarity = -1  # 第一次没有相似度值
#
#         # 保存当前图片
#         cv2.imwrite(os.path.join(output_folder, filename), image)
#         kept_images += 1
#
#         # 更新“最好的图片”及相似度
#         best_image = image
#         best_image_ssim = similarity if best_image is not None else -1
#     else:
#         removed_images += 1
#
# # 统计结果
# print(f"📷 总共处理图片: {total_images}")
# print(f"✅ 保留图片: {kept_images}")
# print(f"❌ 去除图片: {removed_images}")
