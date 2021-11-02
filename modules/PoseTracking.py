import cv2
import mediapipe as mp
# import time


class PoseDetection:
    def __init__(self, static_mode=False, model_complexity=1, smooth=True, segmentation=False, sooth_segmentation=True,
                 min_detec_con=0.5, min_track_con=0.5):
        self.static_mode = static_mode
        self.model_complexity = model_complexity
        self.smooth = smooth
        self.segmentation = segmentation
        self.sooth_segmentation = sooth_segmentation
        self.min_detec_con = min_detec_con
        self.min_track_con = min_track_con

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_mode,
                                     self.model_complexity,
                                     self.smooth,
                                     self.segmentation,
                                     self.sooth_segmentation,
                                     self.min_detec_con,
                                     self.min_track_con)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.result = self.pose.process(imageRGB)
        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(image, self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return image

    def findPosition(self, image, draw=False):
        if self.result.pose_landmarks:
            lmList = []
            for ids, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([ids, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
            return lmList


# def main():
#     pTime = 0
#     cam = cv2.VideoCapture('videos/tmp.mp4')
#     detector = PoseDetection()
#     while True:
#         success, image = cam.read()
#         image = detector.findPose(image, draw=True)
#         landmarkList = detector.findPosition(image, draw=False)
#         print(landmarkList, "\n")
#         cv2.circle(image, (landmarkList[14][1], landmarkList[14][2]), 10, (0, 255, 0), cv2.FILLED)
#         cv2.circle(image, (landmarkList[13][1], landmarkList[13][2]), 10, (0, 255, 0), cv2.FILLED)
#
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime
#         cv2.putText(image, "fps: " + str(int(fps)), (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
#
#         cv2.imshow("Video", image)
#         cv2.waitKey(1)
#
#
# if __name__ == '__main__':
#     main()
