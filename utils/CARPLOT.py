import serial
import asyncio
import time
import random


#说明：
#硬件端超声距离小于20cm表示有物体接近，串口持续发送b'\x01'
#超声距离大于等于20cm表示没有物体接近，串口持续发送b'\x00'
#视觉端检测到车后向串口发送b'\x02'，随后杆子抬起
#测试超声波时用一个平面的物体靠近，超声模块输出才能比较稳定


ser = serial.Serial(port='COM7', baudrate=9600, timeout=3)  # port替换成你电脑上CH340的端口号
car = 1  #
camera_open = 0  # 摄像头状态

while True:
    asyncio.sleep(2)
    received_data = ser.read(1)  # 读取串口数据
    #time.sleep(3)
    print(received_data)
    # if received_data == b'\x01':  # 车来了
    #
    #     car = round(random.random())
    #     if car:  # 如果识别到是车辆
    #         print("识别到车辆")
    #         ser.write(b'\x02')  # 往串口里写2
    #       #  time.sleep(3)  # 执行车牌识别
    #
    # elif received_data == b'\x00':  # 车离开
    #     pass
    #   #print("等待")
    #   #  time.sleep(5)   #车离开，我keil那边直接用超声波距离大于10判断了，不需要回传信息
    #


# 关闭串口
ser.close()
