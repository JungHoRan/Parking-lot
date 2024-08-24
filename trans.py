import asyncio
import websockets
import json

async def send_message():
    uri = "ws://10.129.147.109:8188"  # 替换为你的服务器地址

    data_entry = {
        "id": "CV",
        "toDo": "entry",
        "licensePlateNum": "ABC123",
        "color": "blue",
        "entryTime": "2024-04-16T12:00:00"  # 入口发送进入时间
    }


    async with websockets.connect(uri) as websocket:
        # 发送入口摄像头数据
        await websocket.send(json.dumps(data_entry))
        print(f"Sent entry camera data: {data_entry}")
asyncio.get_event_loop().run_until_complete(send_message())
