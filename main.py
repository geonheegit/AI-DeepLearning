from ultralytics import YOLO
import mss
import cv2
import cvzone
import math
import numpy as np

# 마인크래프트 조작용
import pynput
from pynput.mouse import Button

mouse = pynput.mouse.Controller()
keyboard_button = pynput.keyboard.Controller()
keyboard_key = pynput.keyboard.Key

debug = "True"

# ===============================================

model = YOLO("model/MCPVPAI_Nano.pt")

classNames = ['Bee', 'Cave Spider', 'Chest', 'Cow', 'Creeper', 'Dolphin', 'Enderman', 'Goat', 'Iron Golem', 'Llama',
               'Panda', 'Pig', 'Piglin', 'Player', 'Polar Bear', 'Sheep', 'Spider', 'Trader Lama', 'Villager House',
                 'Villager', 'Wolf', 'Zombified Piglin']

target_CenterPosX = 0
target_CenterPosY = 0
target_width = 0
mobClass = 0

width = 850
height = 500
diffX = 0
diffY = 0

with mss.mss() as sct:
    monitor = {"top": 200, "left": 1000, "width": width, "height": height}

    while True:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = model(img)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                target_width = w

                # 마인크래프트 적 중심 위치
                target_CenterPosX, target_CenterPosY = int(x1 + w/2), int(y1 + h/2)

                cvzone.cornerRect(img, (x1, y1, w, h))

                conf = math.ceil((box.conf[0] * 100)) / 100

                cls = int(box.cls[0])
                mobClass = cls

                cvzone.putTextRect(img, f'{classNames[cls]} {conf * 100}%', (max(0, x1), max(35, y1)), scale = 1, thickness = 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # debug
        img = cv2.rectangle(img, (target_CenterPosX, target_CenterPosY), (target_CenterPosX + 1, target_CenterPosY + 1), (0, 0, 255), 2)
        img = cv2.rectangle(img, (int(width / 2), int(height / 2)), (int(width / 2) + 1, int(height / 2) + 1), (0, 255, 0), 2)

        cv2.imshow("Image", img)

        # 마인크래프트 조작
        mobName = "Player"
        mouse_speed = 1000

        if(target_CenterPosX != 0 and target_CenterPosY != 0):
            if(diffX < 0.45):
                diffX = abs((target_CenterPosX - (width / 2)) / target_CenterPosX)

            if(diffY < 0.45):
                diffY = abs((target_CenterPosY - (height / 2)) / target_CenterPosY)

        if(classNames[mobClass] == mobName and debug == "False"):

            keyboard_button.press("w")

            if(width / 2 > target_CenterPosX + 20):
                mouse.move(-mouse_speed * diffX, 0)
                keyboard_button.press("d")
            elif(width / 2 < target_CenterPosX - 20):
                mouse.move(mouse_speed * diffX, 0)
                keyboard_button.press("a")
            else:
                mouse.click(Button.left)

            if(height / 2 > target_CenterPosY + 20):
                mouse.move(0, -mouse_speed * diffY)
            elif(height / 2 < target_CenterPosY - 20):
                mouse.move(0, mouse_speed * diffY)
            else:
                mouse.click(Button.left)


        else:
            keyboard_button.release("w")
            keyboard_button.release("a")
            keyboard_button.release("s")
            keyboard_button.release("d")
            target_CenterPosX = 0
            target_CenterPosY = 0

        if cv2.waitKey(1) == ord('s'):
            break



