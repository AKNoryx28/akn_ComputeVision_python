import cv2
import numpy as np
import keyboard
import modules.HandTracking as HandTrack
import modules.myenv as myenv
import autopy

#####################################
wCam, hCam = 640, 480
wScr, hScr = autopy.screen.size()
frameR = 230                           # Frame Reduction
smoothening = 100
#####################################

pTime = 0
pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

cam = cv2.VideoCapture(1)
detector = HandTrack.HandDetector(max_hands=1, detection_con=0.5)
env = myenv.MyEnvironment()


while cam.isOpened:
    ret, img = cam.read()
    if ret:
        img = cv2.resize(img, (wCam, hCam))

        # 1. Find hand landmark
        img = detector.findHands(img)
        lmList = detector.findPosition(img, 0, False)

        if len(lmList) != 0:
            # 2. Get the tip of the index and middle finger
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # 3. Check if the fingers up
            fingers = detector.fingersUp()
            cv2.rectangle(img, (frameR, frameR), (wCam -
                          frameR, hCam - frameR), env.COLOR_MAROON, 2)

            # 4. Only index finger : Moving mode
            if fingers[1] == 1 and fingers[2] == 0:
                # 5. Convert coordinates
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # 6. Smoothen values
                cLocX = pLocX + (x3 - pLocX) / smoothening
                cLocY = pLocY + (y3 - pLocY) / smoothening

                # 7. Move mouse
                autopy.mouse.move(wScr - cLocX, cLocY)
                cv2.circle(img, (x1, y1), 15, env.COLOR_BLUE, cv2.FILLED)
                pLocX, pLocY = cLocY, cLocY

            # 8. Both index and middle finger are up : Click Mode
            elif fingers[1] == 1 and fingers[2] == 1:
                # 9. Find distance between fingers
                length, img, lineInfo = detector.findDistance(8, 12, img)
                # 10. Click mouse if distance short
                if length < 40:
                    cv2.circle(
                        img, (lineInfo[4], lineInfo[5]), 15, env.COLOR_WHITE, cv2.FILLED)
                    autopy.mouse.click()

        # 11. Frame rate
        pTime = env.DisplayFps(img, (20, 40), 2, env.COLOR_LIME, 2, pTime)
        # 12. Display
        cv2.imshow('image', img)
    else:
        print('No Video')
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break
    elif keyboard.is_pressed('q'):
        break

cam.release()
cv2.destroyAllWindows()
