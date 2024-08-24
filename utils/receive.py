import asyncio
import json
import websockets

async def request_data(uri):
    data_request = {
        "id": "CV",
        "toDo": "parking"
        # "licensePlateNum": "京A66666",
        # "color": "蓝",
        # "entryTime": "2024-06-05 10:54:09"   # 入口发送进入时间
    }

    async with websockets.connect(uri) as websocket:
        # 发送入口摄像头数据
        await websocket.send(json.dumps(data_request))
        print(f"Sent: {json.dumps(data_request)}")

        # 接收响应
        response_message = await websocket.recv()
        print(f"Received: {response_message}")

        # 解析JSON响应并返回
        response_data = json.loads(response_message)
        return response_data

def receive():
    uri = "ws://10.129.147.109:8188"  # 替换为你的服务器地址
    response_data = asyncio.get_event_loop().run_until_complete(request_data(uri))
    return response_data

if __name__ == "__main__":
    response_data = receive()
    parking_sum = response_data.get("record", {}).get("parkingSum")
    print(f"ParkingSum: {parking_sum}")
    # id = response_data.get("record", {}).get("toDo")
    # print(f"id: {id}")

