import cv2
import os
import torch
import time
from facenet_pytorch import MTCNN

def save_frames_with_faces(video_path, output_folder, video_name, frame_interval=15):
    print("OpenCV 版本:", cv2.__version__)
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 可用:", torch.cuda.is_available())
    print("GPU 名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU")

    # 使用 MTCNN 进行人脸检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")  # 打印当前设备
    mtcnn = MTCNN(keep_all=True, device=device)  # keep_all=True 允许检测多个脸

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    frame_count = 0
    saved_frame_count = 0
    success, frame = video.read()

    start_time = time.time()

    while success:
        if frame_count % frame_interval == 0:
            # OpenCV 是 BGR 格式，MTCNN 需要 RGB 格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 用 MTCNN 进行人脸检测
            boxes, _ = mtcnn.detect(rgb_frame)

            if boxes is not None:
                # 将视频文件名作为前缀，避免文件名冲突
                frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1

        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧，保存 {saved_frame_count} 帧包含人脸的图片")

        success, frame = video.read()
        frame_count += 1

    video.release()
    total_time = time.time() - start_time
    print(f"视频处理完成，共处理 {frame_count} 帧，保存 {saved_frame_count} 帧包含人脸的图片")
    print(f"总用时: {total_time:.2f} 秒")

# 遍历文件夹中的所有视频文件并处理
def process_videos_in_folder(input_folder, output_folder, frame_interval=15):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']  # 支持的视频文件扩展名
    video_files = [f for f in os.listdir(input_folder) if any(f.endswith(ext) for ext in video_extensions)]

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  # 获取视频文件名（不带扩展名）
        print(f"正在处理视频: {video_path}")
        save_frames_with_faces(video_path, output_folder, video_name)

# 运行代码
input_folder = r"H:\fenjie\sp"  # 输入视频文件夹路径
output_folder = r"H:\fenjie\spt"  # 输出文件夹路径
process_videos_in_folder(input_folder, output_folder)










#=========================================================================================================================
# import cv2
# import os
# import torch
# import time
# from facenet_pytorch import MTCNN
#
# def save_frames_with_faces(video_path, output_folder, frame_interval=15):
#     print("OpenCV 版本:", cv2.__version__)
#     print("PyTorch 版本:", torch.__version__)
#     print("CUDA 可用:", torch.cuda.is_available())
#     print("GPU 名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU")
#
#     # 使用 MTCNN 进行人脸检测
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"当前计算设备: {device}")  # 打印当前设备
#     mtcnn = MTCNN(keep_all=True, device=device)  # keep_all=True 允许检测多个脸
#
#     # 确保输出文件夹存在
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     video = cv2.VideoCapture(video_path)
#     if not video.isOpened():
#         print(f"无法打开视频文件: {video_path}")
#         return
#
#     frame_count = 0
#     saved_frame_count = 0
#     success, frame = video.read()
#
#     start_time = time.time()
#
#     while success:
#         if frame_count % frame_interval == 0:
#             # OpenCV 是 BGR 格式，MTCNN 需要 RGB 格式
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # 用 MTCNN 进行人脸检测
#             boxes, _ = mtcnn.detect(rgb_frame)
#
#             if boxes is not None:
#                 frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
#                 cv2.imwrite(frame_filename, frame)
#                 saved_frame_count += 1
#
#         if frame_count % 100 == 0:
#             print(f"已处理 {frame_count} 帧，保存 {saved_frame_count} 帧包含人脸的图片")
#
#         success, frame = video.read()
#         frame_count += 1
#
#     video.release()
#     total_time = time.time() - start_time
#     print(f"视频处理完成，共处理 {frame_count} 帧，保存 {saved_frame_count} 帧包含人脸的图片")
#     print(f"总用时: {total_time:.2f} 秒")
#
# # 运行代码
# video_path = r"H:\fenjie\sp\4.mp4"
# output_folder = r"H:\fenjie\spt"
# save_frames_with_faces(video_path, output_folder)
