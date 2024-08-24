# -*- coding: UTF-8 -*-
import argparse
from pathlib import Path
import os
import cv2
import torch
import asyncio
import websockets
import json
import base64
import threading


import torch.backends.cudnn as cudnn
from numpy import random
import copy
import numpy as np
# from utils.start import capture_image
from datetime import datetime
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.cv_puttext import cv2ImgAddText
from plate_recognition.plate_rec import get_plate_result, allFilePath, init_model, cv_imread
# from plate_recognition.plate_cls import cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
import serial
import time
from utils.start import capture_image,capture_image1

clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
danger = ['危', '险']


def order_points(pts):  # 四个点按照左上 右上 右下 左下排列
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):  # 透视变换得到车牌小图
    # rect = order_points(pts)
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def load_model(weights, device):  # 加载检测模型
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):  # 返回到原图坐标
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    # coords[:, 8].clamp_(0, img0_shape[1])  # x5
    # coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num, device, plate_rec_model,
                           is_color=False):  # 获取车牌坐标以及四个角点坐标并获取车牌号
    h, w, c = img.shape
    result_dict = {}
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness

    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    height = y2 - y1
    landmarks_np = np.zeros((4, 2))
    rect = [x1, y1, x2, y2]
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i] = np.array([point_x, point_y])

    class_label = int(class_num)  # 车牌的的类型0代表单牌，1代表双层车牌
    roi_img = four_point_transform(img, landmarks_np)  # 透视变换得到车牌小图
    if class_label:  # 判断是否是双层车牌，是双牌的话进行分割后然后拼接
        roi_img = get_split_merge(roi_img)
    if not is_color:
        plate_number, rec_prob = get_plate_result(roi_img, device, plate_rec_model, is_color=is_color)  # 对车牌小图进行识别
    else:
        plate_number, rec_prob, plate_color, color_conf = get_plate_result(roi_img, device, plate_rec_model,
                                                                           is_color=is_color)
        # cv2.imwrite("roi.jpg",roi_img)
    result_dict['rect'] = rect  # 车牌roi区域
    result_dict['detect_conf'] = conf  # 检测区域得分
    result_dict['landmarks'] = landmarks_np.tolist()  # 车牌角点坐标
    result_dict['plate_no'] = plate_number  # 车牌号
    result_dict['rec_conf'] = rec_prob  # 每个字符的概率
    result_dict['roi_height'] = roi_img.shape[0]  # 车牌高度
    result_dict['plate_color'] = ""
    if is_color:
        result_dict['plate_color'] = plate_color  # 车牌颜色
        result_dict['color_conf'] = color_conf  # 颜色得分
    result_dict['plate_type'] = class_label  # 单双层 0单层 1双层

    return result_dict


def detect_Recognition_plate(model, orgimg, device, plate_rec_model, img_size, is_color=False):  # 获取车牌信息
    # Load model
    # img_size = opt_img_size
    conf_thres = 0.3  # 得分阈值
    iou_thres = 0.5  # nms的iou值
    dict_list = []
    # orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found '
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]  # 检测前处理，图片长宽变为32倍数，比如变为640X640
    # img =process_data(img0)
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416  图片的BGR排列转为RGB,然后将图片的H,W,C排列变为C,H,W排列

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    # t1 = time_synchronized()/
    pred = model(img)[0]
    # t2=time_synchronized()
    # print(f"infer time is {(t2-t1)*1000} ms")

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # print('img.shape: ', img.shape)
    # print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:13].view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()
                result_dict = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num, device, plate_rec_model,
                                                     is_color=is_color)
                dict_list.append(result_dict)
    return dict_list
    # cv2.imwrite('result.jpg', orgimg)

def trans_result(result_str):

    # 获取当前时间
    current_time = datetime.now()
    # 将时间格式化为精确到秒的字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    result=result_str.split()
    async def send_message():
        uri = "ws://10.129.147.109:8188"  # 替换为你的服务器地址

        data_entry = {
            "id": "CV",
            "toDo": "entry",
            "licensePlateNum": result[0],
            "color": result[1],
            "entryTime": formatted_time  # 入口发送进入时间
        }

        async with websockets.connect(uri) as websocket:
            # 发送入口摄像头数据
            await websocket.send(json.dumps(data_entry))
   #         print(f"Sent entry camera data: {data_entry}")
            # 接收响应
            response_message = await websocket.recv()
            print(f"Response: {response_message}")

            # 解析JSON响应并返回
            response_data = json.loads(response_message)
            return response_data
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    response_data = new_loop.run_until_complete(send_message())
    new_loop.close()
    return response_data
    # response_data = asyncio.get_event_loop().run_until_complete(send_message())
    # return response_data


async def request_data(uri):
    data_request = {
        "id": "CV",
        "toDo": "parking",
    }

    async with websockets.connect(uri) as websocket:
        # 发送入口摄像头数据
        await websocket.send(json.dumps(data_request))
  #      print(f"Sent: {json.dumps(data_request)}")

        # 接收响应
        response_message = await websocket.recv()
        print(f"Received: {response_message}")

        # 解析JSON响应并返回
        response_data = json.loads(response_message)
        return response_data


def OLED(value):
    if value == 0:
        return 'a'
    elif value == 1:
        return 'b'
    elif value == 2:
        return 'c'
    elif value == 3:
        return 'd'
    else:
        return "Other"

def trans_result1(result_str):

    # 获取当前时间
    current_time = datetime.now()
    # 将时间格式化为精确到秒的字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    result=result_str.split()
    async def send_message():
        uri = "ws://10.129.147.109:8188"  # 替换为你的服务器地址

        data_exit = {
            "id": "CV",
            "toDo": "export",
            "licensePlateNum": result[0],
            "color": result[1],
            "entryTime": formatted_time  # 出口发送进入时间
        }

        async with websockets.connect(uri) as websocket:
            # 发送入口摄像头数据
            await websocket.send(json.dumps(data_exit))
   #         print(f"Sent entry camera data: {data_entry}")
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    new_loop.run_until_complete(send_message())
    new_loop.close()
    # asyncio.get_event_loop().run_until_complete(send_message())


def draw_result(orgimg, dict_list, is_color=False):  # 车牌结果画出来
    result_str = ""
    for result in dict_list:
        rect_area = result['rect']

        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h
        rect_area[0] = max(0, int(x - padding_w))
        rect_area[1] = max(0, int(y - padding_h))
        rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

        height_area = result['roi_height']
        landmarks = result['landmarks']
        result_p = result['plate_no']
        if result['plate_type'] == 0:  # 单层
            result_p += " " + result['plate_color']
        else:  # 双层
            result_p += " " + result['plate_color'] + "双层"
        result_str += result_p + " "
        # 将识别到的信息存到excel中

        for i in range(4):  # 关键点
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)  # 画框

        labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # 获得字体的大小
        if rect_area[0] + labelSize[0][0] > orgimg.shape[1]:  # 防止显示的文字越界
            rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])
        orgimg = cv2.rectangle(orgimg, (rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1]))),
                               (int(rect_area[0] + round(1.2 * labelSize[0][0])), rect_area[1] + labelSize[1]),
                               (255, 255, 255), cv2.FILLED)  # 画文字框,背景白色

        if len(result) >= 1:
            orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1])),
                                   (0, 0, 0), 21)
            # orgimg=cv2ImgAddText(orgimg,result_p,rect_area[0]-height_area,rect_area[1]-height_area-10,(0,255,0),height_area)
    print("result_str类型:", type(result_str))
    print("result_str:", result_str)
    return orgimg,result_str


def get_second(capture):
    if capture.isOpened():
        rate = capture.get(5)  # 帧速率
        FrameNumber = capture.get(7)  # 视频文件的帧数
        duration = FrameNumber / rate  # 帧速率/视频总帧数 是时间，除以60之后单位是分钟
        return int(rate), int(FrameNumber), int(duration)

def receive():
    uri = "ws://10.129.147.109:8188"  # 替换为你的服务器地址
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    response_data = new_loop.run_until_complete(request_data(uri))

    new_loop.close()

    return response_data


def send_data(ser):
    while True:
        response_data = receive()
        # 读取特定字段的值
        # parking_sum = response_data
        parking_Rem = response_data.get("record", {}).get("parkingRem")
        print(f"ParkingRem: {parking_Rem}")
        message = OLED(parking_Rem)

        byte_message = message.encode()
        print('发送的数据：', byte_message)
        ser.write(byte_message)
        # ser.write(b'a')
        # print('发送数据: d')
        time.sleep(3)  # 每隔3秒发送一次数据
def trans_result(result_str):

    # 获取当前时间
    current_time = datetime.now()
    # 将时间格式化为精确到秒的字符串
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    result=result_str.split()
    async def send_message():
        uri = "ws://10.129.147.109:8188"  # 替换为你的服务器地址

        data_entry = {
            "id": "CV",
            "toDo": "entry",
            "licensePlateNum": result[0],
            "color": result[1],
            "entryTime": formatted_time  # 入口发送进入时间
        }

        async with websockets.connect(uri) as websocket:
            # 发送入口摄像头数据
            await websocket.send(json.dumps(data_entry))
   #         print(f"Sent entry camera data: {data_entry}")
            # 接收响应
            response_message = await websocket.recv()
            print(f"Response: {response_message}")

            # 解析JSON响应并返回
            response_data = json.loads(response_message)

            return response_data

    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    response_data = new_loop.run_until_complete(send_message())
    new_loop.close()
    return response_data
    # response_data = asyncio.get_event_loop().run_until_complete(send_message())
    # return response_data


async def request_data(uri):
    data_request = {
        "id": "CV",
        "toDo": "parking",
    }

    async with websockets.connect(uri) as websocket:
        # 发送入口摄像头数据
        await websocket.send(json.dumps(data_request))
  #      print(f"Sent: {json.dumps(data_request)}")

        # 接收响应
        response_message = await websocket.recv()
  #     print(f"Received: {response_message}")

        # 解析JSON响应并返回
        response_data = json.loads(response_message)
        return response_data

# 模拟入口处的检测函数
def monitor_entry(detect_model,plate_rec_model,save_path,image_path_entry,ser):
    # ser = serial.Serial(port='COM10', baudrate=9600, timeout=3)  # port替换成你电脑上CH340的端口号
    # print('先发一个')
    # ser.write(b's')
    # ser.write(b'd')
    # camera_open = 0  # 摄像头状态
    result_old = " "
    data_old = b'\x00'  # 设置初始值
    while True:
        received_data = ser.read(1)  # 读取串口数据
        print(received_data)
        if data_old == b'\x00' and received_data == b'\x01':  # 车来了

            capture_image()
            files = os.listdir(image_path_entry)
            image_path = os.path.join(image_path_entry, files[-1])
            img = cv_imread(image_path)
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # detect_one(model,img_path,device)
            dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                                 is_color=opt.is_color)
            if len(dict_list) != 0:  # 如果识别到是车辆
                print("是一辆车")

                ori_img, result_new = draw_result(img, dict_list)
                entry_data = trans_result(result_new)
                to_do = entry_data.get("toDo")
                print("toDo:", to_do)
                print(f"id: {id}")
                result_old = result_new

                img_name = os.path.basename(image_path)
                save_img_path = os.path.join(save_path, img_name)
                cv2.imwrite(save_img_path, ori_img)
                print("识别到车辆")
                if(to_do=="UP"):
                    print('车辆已注册')
                    ser.write(b'\x02')  # 往串口里写2
                else:
                    print('车辆未注册')
                    print("发送数据s")
                    ser.write(b's')
            else:
                print("入口处不是车")
                pass
        else:  # 车离开or检测到同一张车
            pass
        data_old = received_data
    #       time.sleep(0.1)

    # 关闭串口
    ser.close()


# 模拟出口处的检测函数
def monitor_exit(detect_model,plate_rec_model,save_path,image_path_exit,ser):
    # ser = serial.Serial(port='COM7', baudrate=9600, timeout=3)  # port替换成你电脑上CH340的端口号
    result_old = " "
    data_old = b'\x00'  # 设置初始值
    while True:
        received_data = ser.read(1)  # 读取串口数据
        print(received_data)
        if data_old == b'\x00' and received_data == b'\x01':  # 车来了
            capture_image1()
            files = os.listdir(image_path_exit)
            image_path = os.path.join(image_path_exit, files[-1])
            img = cv_imread(image_path)
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # detect_one(model,img_path,device)
            dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                                 is_color=opt.is_color)
            if len(dict_list) != 0:  # 如果识别到是车辆
                print("是一辆车")

                ori_img, result_new = draw_result(img, dict_list)

                if result_old == result_new:
                    print("请尽快通过")
                else:
                    trans_result1(result_new)
                    result_old = result_new

                img_name = os.path.basename(image_path)
                save_img_path = os.path.join(save_path, img_name)
                cv2.imwrite(save_img_path, ori_img)
                print("识别到车辆")
                ser.write(b'\x02')  # 往串口里写2
            else:
                print("出口处不是车")
                pass
        else:  # 车离开or检测到同一张车
            pass
        data_old = received_data
    #       time.sleep(0.1)

    # 关闭串口
    ser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/plate_detect.pt',
                        help='model.pt path(s)')  # 检测模型
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth',
                        help='model.pt path(s)')  # 车牌识别+颜色识别模型
    parser.add_argument('--is_color', type=bool, default=True, help='plate color')  # 是否识别颜色
    parser.add_argument('--image_path', type=str, default='data/capture_images', help='source')  # 图片路径
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')  # 网络输入图片大小
    parser.add_argument('--output_entry', type=str, default='result_entry', help='source')  # 图片结果保存的位置
    parser.add_argument('--output_exit', type=str, default='result_exit', help='source')  # 图片结果保存的位置
    parser.add_argument('--video', type=str, default='', help='source')  # 视频的路径
    #  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                     #使用gpu还是cpu进行识别
    image_path_entry = 'data/capture_images_entry'
    image_path_exit = 'data/capture_images_exit'
    device = torch.device("cpu")
    opt = parser.parse_args()
    print(opt)
    save_path_entry = opt.output_entry
    save_path_exit = opt.output_exit
    if not os.path.exists(save_path_entry):
        os.mkdir(save_path_entry)
    if not os.path.exists(save_path_exit):
        os.mkdir(save_path_exit)
    detect_model = load_model(opt.detect_model, device)  # 初始化检测模型
    plate_rec_model = init_model(device, opt.rec_model, is_color=opt.is_color)  # 初始化识别模型
   # 接受数据调试
   #  received_message = receive();
   #  print(received_message)
 #   创建入口和出口检测的线程
 #    entry_thread = threading.Thread(target=monitor_entry, args=(detect_model, plate_rec_model, save_path_entry,image_path_entry))
 #    exit_thread = threading.Thread(target=monitor_exit, args=(detect_model, plate_rec_model, save_path_entry,image_path_exit))
 #
 #    # 启动线程
 #    entry_thread.start()
 #    exit_thread.start()
 #
 #    # 等待线程完成（实际上这两个线程会一直运行，因此这里不会退出）
 #    entry_thread.join()
 #    exit_thread.join()

    ser = serial.Serial(port='COM10', baudrate=9600, timeout=3)  # port替换成你电脑上CH340的入口端口号
    ser1 = serial.Serial(port='COM7', baudrate=9600, timeout=3)  # port替换成你电脑上CH340的出口端口号
    entry_thread = threading.Thread(target=monitor_entry, args=(detect_model, plate_rec_model, save_path_entry,image_path_entry,ser))
    exit_thread = threading.Thread(target=monitor_exit, args=(detect_model, plate_rec_model, save_path_entry,image_path_exit,ser1))

    send_thread = threading.Thread(target=send_data, args=(ser,))
    send_thread.daemon = True
    send_thread.start()
    # 启动线程
    entry_thread.start()
    exit_thread.start()

    # 等待线程完成（实际上这两个线程会一直运行，因此这里不会退出）
    entry_thread.join()
    exit_thread.join()

