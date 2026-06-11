import cv2 as cv
import mediapipe as mp
import time

class posedetection:
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, frame):
            RGBimg = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = self.pose.process(RGBimg)
            if result.pose_landmarks:
                self.mpDraw.draw_landmarks(frame, result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                for id, lm in enumerate(result.pose_landmarks.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv.circle(frame, (cx, cy), 5, (255, 0, 0), cv.FILLED)
            return frame

if __name__ == "__main__":
        cap = cv.VideoCapture('/home/goge/Desktop/cv/photos&videos/videoplayback.mp4')
        ctime = 0
        ptime = 0
        detector = posedetection()
        while True:
            success, frame = cap.read()
            frame = detector.findPose(frame)
            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime
            cv.putText(frame, f"FPS: {str(int(fps))}", (15, 30), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            cv.imshow('image',frame)
            if cv.waitKey(20) & 0xFF==ord('d'):
                break
