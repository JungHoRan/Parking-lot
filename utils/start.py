import cv2
import os
from datetime import datetime


def detect_cameras():
    # 创建保存照片的文件夹
    save_folder = "photos"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    num_cameras = 4  # 假设最多连接4个摄像头
    for i in range(num_cameras):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"Camera {i} is not available")
        else:
            ret, frame = cap.read()
            if ret:
                photo_path = os.path.join(save_folder, f"camera_{i}.jpg")
                cv2.imwrite(photo_path, frame)
                print(f"Captured image from Camera {i} and saved to {photo_path}")
            else:
                print(f"Failed to capture image from Camera {i}")
        cap.release()

def capture_image():
    # 设置文件保存路径
    folder_path = 'F:/comprehensive/Chinese_license_plate_detection_recognition-main/data/capture_images_entry'
    # 获取当前时间并格式化为字符串形式
    # 检查文件夹是否存在，如果不存在则创建它
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 使用当前时间作为文件名
    file_name = current_time + '.jpg'

    file_path = os.path.join(folder_path, file_name)
    files = os.listdir(folder_path)
    if len(files) >= 10:
        # 排序文件列表，确保最早的文件排在最前面
        files.sort()
        # 删除第一张图片
        os.remove(os.path.join(folder_path, files[0]))

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 捕获第一帧图像
    ret, frame = cap.read()

    # 保存第一帧图像

    cv2.imwrite(file_path, frame)
    print("照片已保存为:", file_path)

    # 关闭摄像头
    cap.release()
def capture_image1():
    # 设置文件保存路径
    folder_path = 'F:/comprehensive/Chinese_license_plate_detection_recognition-main/data/capture_images_exit'
    # 获取当前时间并格式化为字符串形式
    # 检查文件夹是否存在，如果不存在则创建它
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 使用当前时间作为文件名
    file_name = current_time + '.jpg'

    file_path = os.path.join(folder_path, file_name)
    files = os.listdir(folder_path)
    if len(files) >= 10:
        # 排序文件列表，确保最早的文件排在最前面
        files.sort()
        # 删除第一张图片
        os.remove(os.path.join(folder_path, files[0]))

    # 打开摄像头
    cap = cv2.VideoCapture(2)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 捕获第一帧图像
    ret, frame = cap.read()

    # 保存第一帧图像

    cv2.imwrite(file_path, frame)
    print("照片已保存为:", file_path)

    # 关闭摄像头
    cap.release()



# 主程序
if __name__ == "__main__":
    #capture_image()
    detect_cameras()
