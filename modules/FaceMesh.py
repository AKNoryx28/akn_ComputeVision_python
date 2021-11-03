import cv2
import mediapipe as mp
import time
import keyboard


class FaceMesh:
    def __init__(self, static_mode=False, num_face=1, refine_landmarks=False,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_mode = static_mode
        self.num_face = num_face
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            self.static_mode, self.num_face, self.refine_landmarks,
            self.min_detection_confidence, self.min_tracking_confidence)
        self.drawSpec = self.mpDraw.DrawingSpec(
            color=[255, 255, 0], thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw_connections=True, draw_id=False):
        # img = cv2.resize(img, (1300, 720))
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        face = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw_connections:
                    self.mpDraw.draw_landmarks(
                        image=img,
                        landmark_list=faceLms,
                        connections=self.mpFaceMesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawSpec,
                        connection_drawing_spec=self.drawSpec)
                for lmId, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y, z = int(lm.x * iw), int(lm.y * ih), lm.z
                    if draw_id:
                        cv2.putText(img, str(lmId), (x, y),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)

        return img, faces


def main():
    pTime = 0
    cap = cv2.VideoCapture('TestVideos/3.mp4')
    detectorFaceMesh = FaceMesh(num_face=2, min_detection_confidence=0.7)
    while cap.isOpened:
        ret, img = cap.read()
        if ret:
            img, faces = detectorFaceMesh.findFaceMesh(img, False, True)
            # if len(faces) != 0:
                # print(faces[0])

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f"FPS: {int(fps)}", (15, 70),
                        cv2.FONT_HERSHEY_PLAIN, 2, (100, 100, 200), 2)
            cv2.imshow("Face Mesh", img)
        else:
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break
        if keyboard.is_pressed('q'):
            cv2.imwrite("result_face_mesh_id.png", img)
            break

    cap.release()
    cv2.destroyAllWindows


if __name__ == '__main__':
    main()
