from cv2 import FILLED, VideoCapture, imshow, waitKey, putText, FONT_HERSHEY_PLAIN, destroyAllWindows
from time import time
from keyboard import is_pressed
# import modules.HandTracking as ht
# import modules.PoseTracking as pt
import modules.FaceTracking as ft

cap = VideoCapture('TestVideos/1.mp4')
# handDetector = ht.HandDetector()
# poseDetector = pt.PoseDetection()
faceDetector = ft.FaceDetector(0.7, 1)
currentTime = 0
pTime = 0

while True:
    ret, img = cap.read()
    if ret:
        # img = handDetector.findHands(img, True)
        # img = poseDetector.findPose(img, True)
        img, bbox, = faceDetector.findFace(img, True)
        # lmsListHand = handDetector.findPosition(img, draw=True)
        # if len(lmsListHand) != 0:
        #     print(lmsListHand[0])
        # lmsListPose = poseDetector.findPosition(img, True)
        # if lmsListPose != None:
        #     circle(img, (lmsListPose[0][1], lmsListPose[0][2]), 5, (255, 255, 0), FILLED)

        currentTime = time()
        _FPS = 1 / (currentTime - pTime)
        pTime = currentTime

        putText(img, "fps: " + str(int(_FPS)), (15, 70), FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        imshow("HandTracking", img)
        waitKey(1)

    if is_pressed('q'):
        break

cap.release()
destroyAllWindows()
