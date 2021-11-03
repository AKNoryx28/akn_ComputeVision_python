from cv2 import FILLED, VideoCapture, circle, resize, imshow, waitKey, putText, FONT_HERSHEY_PLAIN, destroyAllWindows, imwrite
from time import time
from keyboard import is_pressed
import modules.HandTracking as ht
import modules.PoseTracking as pt
import modules.FaceTracking as ft
import modules.FaceMesh as fm

cap = VideoCapture('TestVideos/1.mp4')
handDetector = ht.HandDetector()
poseDetector = pt.PoseDetection()
faceDetector = ft.FaceDetector(0.7, 1)
faceMesh = fm.FaceMesh(num_face=2, min_detection_confidence=0.7)
currentTime = 0
pTime = 0

while cap.isOpened():
    ret, img = cap.read()
    if ret:
        img = resize(img, (1300, 720))
        # img = handDetector.findHands(img, True)
        # img = poseDetector.findPose(img, True)
        # img, bbox, = faceDetector.findFace(img, True)
        img, faces = faceMesh.findFaceMesh(img, draw_connections=True, draw_id=False)

        # lmsListHand = handDetector.findPosition(img, draw=True)
        # if len(lmsListHand) != 0:
        #     print(lmsListHand[0])

        # lmsListPose = poseDetector.findPosition(img, True)
        # if lmsListPose != None:
        #     circle(img, (lmsListPose[0][1], lmsListPose[0][2]), 5, (255, 255, 0), FILLED)

        # if len(faces) != 0:
        #     print(faces[0])

        currentTime = time()
        _FPS = 1 / (currentTime - pTime)
        pTime = currentTime

        putText(img, "fps: " + str(int(_FPS)), (15, 70), FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        imshow("HandTracking", img)
    else:
        print("Video or webcam not found")
        break

    if waitKey(1) & 0xFF == 27:
        break
    if is_pressed('q'):
        imwrite("result_face_mesh_connections.png", img)
        break

cap.release()
destroyAllWindows()
