from os import system
import cv2
import numpy as np
import time
import keyboard
import modules.HandTracking as HandTrack
import os

################################################################
brushThickness = 15
eraserThickness = 50
################################################################

# Set up camera and hand detector
cam = cv2.VideoCapture('TestVideos/hc.mp4')
detector = HandTrack.HandDetector(max_hands=1, detection_con=0.85)
prevTime = 0

# Get the header image and put it into List
vPaintDir = "src/Project/res/vpaint"
vPaintListDir = os.listdir(vPaintDir)
print(vPaintListDir)
vPaintList = []

for imPath in vPaintListDir:
    imgFinger = cv2.imread(f'{vPaintDir}/{imPath}')
    vPaintList.append(imgFinger)
header = vPaintList[1]
drawColor = (0, 0, 0)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


# Function to Draw FPS
def showFps(image, pos_w, pos_h, prevTime=0):
    curTime = time.time()
    _fps = 1 / (curTime - prevTime)
    prevTime = curTime
    cv2.putText(image, f"fps: {int(_fps)}", (pos_w, pos_h),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    return prevTime


# Loop while camera is running
while cam.isOpened:
    ret, img = cam.read()

    # Do the processing if success
    if ret:
        # resize image and flip
        img = cv2.resize(img, (1280, 720))
        img = cv2.flip(img, 1)

        # Find hand landmarks
        img = detector.findHands(img)
        lmList = detector.findPosition(img, hand_num=0, draw=False)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]                  # tip of index tip finger
            # tip of index middle finger
            x2, y2 = lmList[12][1:]
            fingers = detector.fingersUp()          # Check witch finger are up

            # If selection Mode - Two finger are up
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                # print("Selection mode")
                if y1 < 105:
                    if 300 < x1 < 465:
                        drawColor = (90, 221, 254)
                        header = vPaintList[4]
                    elif 465 < x1 < 635:
                        drawColor = (255, 255, 255)
                        header = vPaintList[3]
                    elif 635 < x1 < 772:
                        drawColor = (85, 215, 126)
                        header = vPaintList[1]
                    elif 808 < x1 < 978:
                        drawColor = (23, 23, 255)
                        header = vPaintList[2]
                    elif 1080 < x1 < 1184:
                        drawColor = (0, 0, 0)
                        header = vPaintList[0]
                # Draw indicator selected color
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),
                              drawColor, cv2.FILLED)

            # Else if selection Mode - One finger are up
            elif fingers[1] and fingers[2] == False:
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == (0, 0, 0):
                    cv2.circle(img, (x1, y1), eraserThickness, drawColor, cv2.FILLED)
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.circle(img, (x1, y1), brushThickness, drawColor, cv2.FILLED)
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

                xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50,255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)


        img[0:105, 0:1280] = header
        # Call fps function
        prevTime = showFps(img, 20, 150, prevTime)
        # Show image after processing
        # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
        cv2.imshow('image', img)
        # cv2.imshow('image canvas', imgCanvas)
    else:
        print("No image")
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break
    elif keyboard.is_pressed('q'):
        break

cam.release()
cv2.destroyAllWindows()
