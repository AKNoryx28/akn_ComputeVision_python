import modules.HandTracking as HandTrack
import cv2
import time
import keyboard
import os
from PIL import Image

################################################################
wCam, hCam = 850, 480
prevTime = 0
################################################################
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_LIME = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_SILVER = (192, 192, 192)
COLOR_GRAY = (128, 128, 128)
COLOR_MAROON = (128, 0, 0)
COLOR_OLIVE = (128, 128, 0)
COLOR_GREEN = (0, 128, 0)
COLOR_PURPLE = (128, 0, 128)
COLOR_TEA = (0, 128, 128)
COLOR_NAVY = (0, 0, 128)
################################################################

detector = HandTrack.HandDetector()
folderImgFingers = "Fingers/"
fingerList = os.listdir(folderImgFingers)
overlayList = []
tipIds = [4, 8, 12, 16, 20]
totalFingers = 0

for imPath in fingerList:
    imgFinger = cv2.imread(f'{folderImgFingers}{imPath}')
    overlayList.append(imgFinger)
    # print(f'{folderImgFingers}/{imPath}')


cam = cv2.VideoCapture('TestVideos/hc.mp4')
cam.set(3, wCam)
cam.set(4, hCam)


while cam.isOpened:
    ret, img = cam.read()
    if ret:
        img = cv2.resize(img, (wCam, hCam))
        img = detector.findHands(img)
        lmListHand = detector.findPosition(img, draw=False)
        if len(lmListHand) != 0:
            fingers = []

            # Thumb
            if lmListHand[tipIds[0]][1] < lmListHand[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if lmListHand[tipIds[id]][2] < lmListHand[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            totalFingers = fingers.count(1)

            overlayList[totalFingers-1] = cv2.resize(overlayList[totalFingers-1], (150, 200))
            h, w, c = overlayList[totalFingers-1].shape
            img[0:h, 0:w] = overlayList[totalFingers-1]

        # Calculate and show FPS
        curTime = time.time()
        _fps = 1 / (curTime - prevTime)
        prevTime = curTime
        cv2.putText(img, f"fps: {int(_fps)}", (10, 250),
                    cv2.FONT_HERSHEY_PLAIN, 2, COLOR_RED, 2)

        cv2.putText(img, f"Fingers: {totalFingers}", (10, 300), cv2.FONT_HERSHEY_PLAIN, 2, COLOR_LIME, 2)
        cv2.imshow("Window", img)
    else:
        print("No Hands")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break
    elif keyboard.is_pressed('q'):
        break


cam.release()
cv2.destroyAllWindows()
