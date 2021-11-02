import cv2
import mediapipe as mp
import time
from keyboard import is_pressed


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Args:
        min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
            detection to be considered successful. See details in
            https://solutions.mediapipe.dev/face_detection#min_detection_confidence.
        model_selection: 0 or 1. 0 to select a short-range model that works
            best for faces within 2 meters from the camera, and 1 for a full-range
            model best for faces within 5 meters. See details in
            https://solutions.mediapipe.dev/face_detection#model_selection.
        """
        # initialize parameters
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence

        # Variables
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            self.min_detection_confidence, self.model_selection)

    def findFace(self, img, draw=True):
        img = cv2.resize(img, (1300, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.faceDetection.process(imgRGB)
        bboxs = []

        if self.result.detections:
            for id, detections in enumerate(self.result.detections):
                # mpDraw.draw_detection(img, detections)
                bboxC = detections.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detections.score])

                if draw:
                    self.fancyDraw(img, bbox)
                    cv2.putText(img, f"{int(detections.score[0] * 100)} %",
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=8, rt=1):
        x, y, w, h = bbox
        x1, y1, = x + w, y + h

        # Rectangle box thin
        cv2.rectangle(img, bbox, (255, 0, 255), rt)

        # Top left corner x, y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        # Top right corner x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        # Bottom left corner x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        # Bottom right corner x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)


# def main():
#     pTime = 0
#     cap = cv2.VideoCapture('TestVideos/1.mp4')
#     detector = FaceDetector(model_selection=1)
#     while True:
#         ret, img = cap.read()
#         if ret:
#             img, bbox = detector.findFace(img, True)

#             # Calculate FPS
#             cTime = time.time()
#             _FPS = 1 / (cTime - pTime)
#             pTime = cTime
#             cv2.imshow("Video", img)
#             cv2.putText(img, f"fps: {int(_FPS)}", (20, 40),
#                         cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

#             cv2.waitKey(1)

#         if is_pressed('q'):
#             break


# if __name__ == '__main__':
#     main()
