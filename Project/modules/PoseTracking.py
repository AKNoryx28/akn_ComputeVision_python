from re import A
import cv2
import mediapipe as mp
import math


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
                self.mpDraw.draw_landmarks(
                    image, self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return image

    def findPosition(self, image, draw=False):
        if self.result.pose_landmarks:
            self.lmList = []
            for ids, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([ids, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
            return self.lmList

    def findAngle(self, image, p1, p2, p3, draw=True):
        # Get the landmark
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) -
                             math.atan2(y1-y2, x1-x2))
        if angle < 0:
            angle += 360

        # Drawing point locations
        if draw:
            # line
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.line(image, (x3, y3), (x2, y2), (255, 0, 255), 2)
            # circle
            cv2.circle(image, (x1, y1), 7, (255, 255, 0), cv2.FILLED)
            cv2.circle(image, (x1, y1), 13, (255, 255, 0), 1)
            cv2.circle(image, (x2, y2), 7, (255, 255, 0), cv2.FILLED)
            cv2.circle(image, (x2, y2), 13, (255, 255, 0), 1)
            cv2.circle(image, (x3, y3), 7, (255, 255, 0), cv2.FILLED)
            cv2.circle(image, (x3, y3), 13, (255, 255, 0), 1)
            # Indicator
            cv2.putText(image, str(int(angle)), (x2 - 40, y2 - 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

        return angle


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
