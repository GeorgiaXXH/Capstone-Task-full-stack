#! -*- coding: utf-8 -*-

import asyncio
import websockets
from analyze import answer
import re
import random
number_house=random.randint(0,40000)
websocket_users = set()
async def check_user_permit(websocket):
    print("new websocket_users:", websocket)
    websocket_users.add(websocket)
    print("websocket_users list:", websocket_users)
    while True:
        recv_text = await websocket.recv()
        # from analyze import answer
        # import re
        # import random
        # number_house=random.randint(0,40000)
        list1 = answer(str(recv_text).split(":")[1],house_number=number_house)
        response_text = f"Server return: {recv_text}"
        response_text = list1[0] + "\n"
        response_text += re.sub(r'Name: [0-9], dtype: object', '', list1[1]) + "\n"
        response_text += list1[2] + "\n"
        for item in list1[3]:
            if list1[3][item] == 1:
                response_text += "!!!" + item + "!!!" + "\n"
            else:
                response_text += item + "\n"
        # response_text += list1[3]
        response_text += list1[4] + "\n"
        response_text += list1[5] + "\n"
        # print("response_text:", response_text)
        await websocket.send(response_text)
        return True

async def recv_user_msg(websocket):
    while True:
        print(2)
        recv_text = await websocket.recv()
        # from analyze import answer
        # import re
        # import random
        # number_house=random.randint(0,40000)
        list1 = answer(str(recv_text).split(":")[1],house_number=number_house)
        response_text = f"Server return: {recv_text}"
        response_text = list1[0] +"\n"
        response_text += re.sub(r'Name: [0-9], dtype: object', '', list1[1]) +"\n"
        response_text += list1[2] +"\n"
        for item in list1[3]:
            if list1[3][item] == 1:
                response_text += "!!!" + item + "!!!" +"\n"
            else:
                response_text += item +"\n"
        # response_text += list1[3]
        response_text += list1[4]+"\n"
        response_text += list1[5]+"\n"
        # print("response_text:", response_text)
        await websocket.send(response_text)

async def run(websocket, path):
    while True:
        try:
            await check_user_permit(websocket)
            await recv_user_msg(websocket)
        except websockets.ConnectionClosed:
            print("ConnectionClosed...", path)    
            print("websocket_users old:", websocket_users)
            websocket_users.remove(websocket)
            print("websocket_users new:", websocket_users)
            break
        except websockets.InvalidState:
            print("InvalidState...")   
            break
        except Exception as e:
            print("Exception:", e)


if __name__ == '__main__':
    print("127.0.0.1:8181 websocket...")
    asyncio.get_event_loop().run_until_complete(websockets.serve(run, "127.0.0.1", 8181))
    asyncio.get_event_loop().run_forever()
