import cv2
import mediapipe as mp
# import time


class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_comp=1, detection_con=0.5, tracking_con=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.modelComplexity = model_comp
        self.detectionCon = detection_con
        self.trackingCon = tracking_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        self.maxHands,
                                        self.modelComplexity,
                                        self.detectionCon,
                                        self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, image, draw=True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for handLms in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)

        return image

    def findPosition(self, image, hand_num=0, draw=True):
        self.lmList = []

        if self.result.multi_hand_landmarks:
            currentHands = self.result.multi_hand_landmarks[hand_num]
            for ids, lm in enumerate(currentHands.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([ids, cx, cy])

                if draw:
                    cv2.circle(image, (cx, cy), 8, (255, 255, 0), cv2.FILLED)

        return self.lmList
    
    def fingersUp(self, draw=True):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


# def main():
#     pTime = 0
#     cTime = 0
#     cap = cv2.VideoCapture(1)
#     detector = HandDetector()
#
#     while True:
#         success, img = cap.read()
#         img = detector.findHands(img)
#         landmarkList = detector.findPosition(img)
#         if len(landmarkList) != 0:
#             print(landmarkList[4])
#
#         cTime = time.time()
#         fps = 1/(cTime - pTime)
#         pTime = cTime
#
#         cv2.putText(img, str(int(fps)), (15, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#
#         cv2.imshow("Image", img)
#         cv2.waitKey(1)
#
#
# if __name__ == "__main__":
#     main()
