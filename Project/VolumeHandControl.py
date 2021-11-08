import modules.HandTracking as HandTrack
import cv2
import time
import numpy as np
import keyboard
import math

# Volume controller library
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


################################################################
wCam, hCam = 850, 480
prevTime = 0
################################################################
COLOR_WHITE             = (255, 255, 255)
COLOR_BLACK             = (0, 0, 0)
COLOR_BLUE              = (255, 0, 0)
COLOR_LIME              = (0, 255, 0)
COLOR_RED               = (0, 0, 255)
COLOR_YELLOW            = (255, 255, 0)
COLOR_CYAN              = (0, 255, 255)
COLOR_MAGENTA           = (255, 0, 255)
COLOR_SILVER            = (192, 192, 192)
COLOR_GRAY              = (128, 128, 128)
COLOR_MAROON            = (128, 0, 0)
COLOR_OLIVE             = (128, 128, 0)
COLOR_GREEN             = (0, 128, 0)
COLOR_PURPLE            = (128, 0, 128)
COLOR_TEA               = (0, 128, 128)
COLOR_NAVY              = (0, 0, 128)
################################################################

cam = cv2.VideoCapture(1)
cam.set(3, wCam)
cam.set(4, hCam)
detector = HandTrack.HandDetector(detection_con=0.7)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while cam.isOpened():
    ret, img = cam.read()
    if ret:
        img = cv2.resize(img, (wCam, hCam))

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            centerX, centerY = (x1+x2) // 2, (y1+y2) // 2

            cv2.circle(img, (x1, y1), 10, COLOR_OLIVE, cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, COLOR_OLIVE, cv2.FILLED)
            cv2.circle(img, (centerX, centerY), 10, COLOR_OLIVE, cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), COLOR_BLUE, 3)

            length = math.hypot(x2 - x1, y2 - y1)
            # print(length)

            # Hand range 30 - 300
            # Volume range -64 - 0

            vol = np.interp(length, [30, 300], [minVol, maxVol])
            volBar = np.interp(length, [30, 300], [400, 100])
            volPer = np.interp(length, [30, 300], [0, 100])
            # print(int(length), vol)
            volume.SetMasterVolumeLevel(vol, None)
            
            if length < 40:
                 cv2.circle(img, (centerX, centerY), 10, COLOR_LIME, cv2.FILLED)


        else:
            # print("No hands")
            pass

        
        cv2.rectangle(img, (50, 100), (85, 400), COLOR_YELLOW, 2)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), COLOR_GRAY, cv2.FILLED)

        curTime = time.time()
        _fps = 1 / (curTime - prevTime)
        prevTime = curTime
        cv2.putText(img, f"fps: {int(_fps)}", (10, 40),
                    cv2.FONT_HERSHEY_PLAIN, 2, COLOR_RED, 2)
        cv2.putText(img, str(int(volPer))+" %", (55, 429),
                    cv2.FONT_HERSHEY_PLAIN, 2, COLOR_MAROON, 2)


        cv2.imshow("Window", img)
    else:
        print("Video not found")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break
    elif keyboard.is_pressed('q'):
        break

cam.release()
cv2.destroyAllWindows()
