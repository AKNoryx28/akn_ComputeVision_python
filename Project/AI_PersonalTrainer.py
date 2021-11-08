import cv2
import keyboard
import time
import numpy as np
import modules.PoseTracking as PoseTracking

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

cam = cv2.VideoCapture('TestVideos/little_jump.mp4')

detector = PoseTracking.PoseDetection()
count1, count2 = 0, 0
dir1, dir2 = 0, 0


def showFps(image, pos_w, pos_h, prevTime=0):
    curTime = time.time()
    _fps = 1 / (curTime - prevTime)
    prevTime = curTime
    cv2.putText(img, f"fps: {int(_fps)}", (pos_w, pos_h),
                cv2.FONT_HERSHEY_PLAIN, 2, COLOR_RED, 2)
    return prevTime


while cam.isOpened:
    ret, img = cam.read()
    # img = cv2.imread('src/Project/AiTrainer/pic.jpg')
    if ret:
        img = cv2.resize(img, (hCam, wCam))
        img = detector.findPose(img, False)
        lmList = detector.findPosition(img, False)
        if len(lmList) != 0:
            angle_l = detector.findAngle(img, 23, 25, 27)
            angle_r = detector.findAngle(img, 24, 26, 28)

            perLeft = np.interp(angle_l, (75, 150), (100, 0))
            perRight = np.interp(angle_r, (75, 150), (100, 0))
            perLBar = np.interp(angle_l, (75, 150), (530, 840))
            perRBar = np.interp(angle_r, (75, 150), (530, 840))
            perLPer = np.interp(angle_l, (75, 150), (100, 0))
            perRPer = np.interp(angle_r, (75, 150), (100, 0))

            # print(int(perLeft), "  ", int(perRight))
            if perLeft == 100:
                color1 = COLOR_SILVER
                if dir1 == 0:
                    count1 += 0.5
                    dir1 = 1
            elif perLeft == 0:
                color1 = COLOR_CYAN
                if dir1 == 1:
                    count1 += 0.5
                    dir1 = 0

            if perRight == 100:
                color2 = COLOR_SILVER
                if dir2 == 0:
                    count2 += 0.5
                    dir2 = 1
            elif perRight == 0:
                color2 = COLOR_CYAN
                if dir2 == 1:
                    count2 += 0.5
                    dir2 = 0

            cv2.putText(img, "L: "+str(int(count1)), (6, 520),
                        cv2.FONT_HERSHEY_PLAIN, 1, COLOR_WHITE, 2)
            cv2.putText(img, "R: "+str(int(count2)), (48, 520),
                        cv2.FONT_HERSHEY_PLAIN, 1, COLOR_WHITE, 2)
            cv2.rectangle(img, (5, int(perLBar)), (40, 840), color1, cv2.FILLED)
            cv2.rectangle(img, (5, 530), (40, 840), COLOR_GREEN, 2)
            cv2.rectangle(img, (44, int(perRBar)), (79, 840), color2, cv2.FILLED)
            cv2.rectangle(img, (44, 530), (79, 840), COLOR_GREEN, 2)
            cv2.putText(img, str(int(perLPer)) +"    "+ str(int(perRPer)), (9, 830),
                        cv2.FONT_HERSHEY_PLAIN, 1, COLOR_BLUE, 2)

        prevTime = showFps(img, 10, 40, prevTime)
        cv2.imshow("Window", img)
    else:
        print("error")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break
    elif keyboard.is_pressed('q'):
        # cv2.imwrite('src/Project/AiTrainer/pic.jpg',img)
        break


cam.release()
cv2.destroyAllWindows()
