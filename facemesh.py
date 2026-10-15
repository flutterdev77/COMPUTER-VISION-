import cv2 as cv
import mediapipe as mp
import os
import sys
import time

VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videoplayback.mp4")


class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, refineLandmarks=True, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLandmarks = refineLandmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            refine_landmarks=self.refineLandmarks,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFaceMesh(self, frame):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(
                    frame, faceLms,
                    self.mpFaceMesh.FACEMESH_TESSELATION,
                    self.drawSpec, self.drawSpec
                )

                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = frame.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    if id == 1:
                        cv.circle(frame, (x, y), 5, (255, 0, 255), cv.FILLED)
        return frame


def main():
    useCamera = "--camera" in sys.argv
    cap = cv.VideoCapture(0 if useCamera else VIDEO_PATH)

    ptime = 0
    detector = FaceMeshDetector()

    while True:
        success, frame = cap.read()
        if not success:
            if useCamera:
                print("Could not read from the camera, so check camera permissions!")
            else:
                print("Could not read the video, so check the path!")
            break
        if useCamera:
            frame = cv.flip(frame, 1)

        frame = detector.findFaceMesh(frame)

        ctime = time.time()
        fps = 1 / (ctime - ptime) if ctime != ptime else 0
        ptime = ctime

        cv.putText(frame, f'FPS: {int(fps)}', (20, 70),
                   cv.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv.imshow("Face Mesh", frame)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
