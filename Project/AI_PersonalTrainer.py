import cv2
import keyboard
import time
import numpy as np
import modules.PoseTracking as PoseTracking

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

cam = cv2.VideoCapture('TestVideos/little_jump.mp4')
cam.set(3, wCam)
cam.set(4, hCam)
detector = PoseTracking.PoseDetection()

def showFps(image, pos_w, pos_h, prevTime=0):
    curTime = time.time()
    _fps = 1 / (curTime - prevTime)
    prevTime = curTime
    cv2.putText(img, f"fps: {int(_fps)}", (pos_w, pos_h),
                cv2.FONT_HERSHEY_PLAIN, 2, COLOR_RED, 2)
    return prevTime

while cam.isOpened:
    ret, img = cam.read()
    if ret:
        img = cv2.resize(img, (hCam, wCam))
        img = detector.findPose(img)

        prevTime = showFps(img, 10, 40, prevTime)
        cv2.imshow("Window", img)
    else:
        print("error")
        break

    if cv2.waitKey(1) & 0xFF == 27:
            break
    elif keyboard.is_pressed('q'):
            break

cam.release()
cv2.destroyAllWindows()
