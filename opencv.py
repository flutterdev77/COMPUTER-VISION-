import cv2 as cv
import mediapipe as mp
import numpy as np
import os
import sys
import time
from functools import lru_cache

VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videoplayback.mp4")

MESH_COLOR = (224, 224, 224)
P = mp.solutions.holistic.PoseLandmark
# Each limb gets a grid of mesh points: (start joint, end joint, points along, points across)
LIMBS = [
    (P.LEFT_SHOULDER, P.LEFT_ELBOW, 20, 10), (P.LEFT_ELBOW, P.LEFT_WRIST, 20, 10),
    (P.RIGHT_SHOULDER, P.RIGHT_ELBOW, 20, 10), (P.RIGHT_ELBOW, P.RIGHT_WRIST, 20, 10),
    (P.LEFT_HIP, P.LEFT_KNEE, 24, 10), (P.LEFT_KNEE, P.LEFT_ANKLE, 24, 10),
    (P.RIGHT_HIP, P.RIGHT_KNEE, 24, 10), (P.RIGHT_KNEE, P.RIGHT_ANKLE, 24, 10),
    (P.LEFT_ANKLE, P.LEFT_FOOT_INDEX, 8, 6), (P.RIGHT_ANKLE, P.RIGHT_FOOT_INDEX, 8, 6),
]
TORSO_ROWS, TORSO_COLS = 26, 22
NECK_ROWS, NECK_COLS = 10, 12
RING_BOW = 0.3      # minimum ring curvature, so the wrap still reads as 3D when a part lies flat in the image
TORSO_DEPTH = 0.6   # torso depth relative to its width


@lru_cache(maxsize=None)
def gridEdges(rows, cols):
    # flat point indices for every mesh edge: across, along, and one diagonal per cell
    idx = np.arange(rows * cols).reshape(rows, cols)
    starts = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel(), idx[:-1, :-1].ravel()])
    ends = np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel(), idx[1:, 1:].ravel()])
    return starts, ends


def towardCamera(a, b, za, zb):
    # sine of how far a body part points toward or away from the camera; 0 means it lies flat in the image
    d = np.array([b[0] - a[0], b[1] - a[1], zb - za])
    return min(abs(d[2]) / max(np.linalg.norm(d), 1), 0.9)


def wrapRows(centers, leftHalf, rightHalf, across, axis, bow, cols):
    # lay each row over the front half of a tube: points bunch toward the edges like a cylinder
    # seen from the front, and each ring bows along the body axis
    theta = np.linspace(-np.pi / 2, np.pi / 2, cols)[None, :, None]
    half = np.where(theta < 0, leftHalf[:, None, None], rightHalf[:, None, None])
    return (centers[:, None, :] + np.sin(theta) * half * across[:, None, :]
            + np.cos(theta) * bow[:, None, None] * axis)

class posedetection:
    def __init__(self, modelComplexity=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic(
            model_complexity=modelComplexity,
            smooth_landmarks=True,
            enable_segmentation=True,
            min_detection_confidence=minDetectionCon,
            min_tracking_confidence=minTrackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(color=MESH_COLOR, thickness=1, circle_radius=1)
        self.tiltHistory = {}

    def smoothTilt(self, key, tilt):
        # pose depth is noisier than x/y, so ease each part's tilt over a few frames to stop the rings flickering
        prev = self.tiltHistory.get(key, tilt)
        self.tiltHistory[key] = 0.7 * prev + 0.3 * tilt
        return self.tiltHistory[key]

    def torsoGrid(self, mask, pts, depth):
        shoulders = (pts[P.LEFT_SHOULDER] + pts[P.RIGHT_SHOULDER]) / 2
        hips = (pts[P.LEFT_HIP] + pts[P.RIGHT_HIP]) / 2
        length = max(np.linalg.norm(hips - shoulders), 1)
        shoulderWidth = np.linalg.norm(pts[P.LEFT_SHOULDER] - pts[P.RIGHT_SHOULDER])
        # the joints sit inside the body outline, so run from a little above the shoulders to a little below the hips
        top = shoulders - 0.08 * (hips - shoulders)
        bottom = hips + 0.08 * (hips - shoulders)
        tilt = self.smoothTilt("torso", towardCamera(
            shoulders, hips, (depth[P.LEFT_SHOULDER] + depth[P.RIGHT_SHOULDER]) / 2,
            (depth[P.LEFT_HIP] + depth[P.RIGHT_HIP]) / 2))
        # width comes from the body outline and is limited by torso length, not just shoulder width,
        # so it still fits from the side when the shoulders overlap
        return self.limbGrid(mask, top, bottom, tilt, TORSO_ROWS, TORSO_COLS,
                             minHalf=0.15 * length, maxHalf=max(0.65 * shoulderWidth, 0.35 * length),
                             depthRatio=TORSO_DEPTH)

    def neckGrid(self, mask, pts, depth):
        shoulders = (pts[P.LEFT_SHOULDER] + pts[P.RIGHT_SHOULDER]) / 2
        hips = (pts[P.LEFT_HIP] + pts[P.RIGHT_HIP]) / 2
        length = max(np.linalg.norm(hips - shoulders), 1)
        base = shoulders - 0.08 * (hips - shoulders)   # top row of the torso
        # pose has no chin point; on the front-facing test video the chin sits about 0.39 of the way
        # from the ears to the torso top. Ears are tracked from the front, side and back, unlike the mouth
        ears = (pts[P.LEFT_EAR] + pts[P.RIGHT_EAR]) / 2
        chin = ears + 0.39 * (base - ears)
        zChin = (depth[P.LEFT_EAR] + depth[P.RIGHT_EAR]) / 2
        zBase = (depth[P.LEFT_SHOULDER] + depth[P.RIGHT_SHOULDER]) / 2
        tilt = self.smoothTilt("neck", towardCamera(chin, base, zChin, zBase))
        # run chin -> base so the rings bow downward, the same way as the torso
        return self.limbGrid(mask, chin, base, tilt, NECK_ROWS, NECK_COLS,
                             minHalf=0.08 * length, maxHalf=0.12 * length)

    def limbGrid(self, mask, a, b, tilt, rows, cols, minHalf=None, maxHalf=None, depthRatio=1.0):
        length = max(np.linalg.norm(b - a), 1)
        minHalf = 0.08 * length if minHalf is None else minHalf
        maxHalf = 0.3 * length if maxHalf is None else maxHalf
        n = np.array([a[1] - b[1], b[0] - a[0]]) / length
        centers = a + np.linspace(0, 1, rows)[:, None] * (b - a)
        halfWidths = []
        for side in (-1, 1):
            if mask is None:
                halfWidths.append(np.full(rows, (minHalf + maxHalf) / 2))
                continue
            # walk outward from the bone until the segmentation mask stops saying "body"
            steps = np.arange(1, max(int(maxHalf), 1) + 1)
            probe = centers[:, None, :] + side * steps[None, :, None] * n
            x = np.clip(probe[..., 0].astype(int), 0, mask.shape[1] - 1)
            y = np.clip(probe[..., 1].astype(int), 0, mask.shape[0] - 1)
            inside = mask[y, x] > 0.5
            edge = np.where(inside.all(axis=1), len(steps), inside.argmin(axis=1))
            hw = np.clip(edge, minHalf, maxHalf)
            halfWidths.append(np.convolve(np.pad(hw, 1, mode="edge"), [0.25, 0.5, 0.25], "valid"))
        bow = max(tilt, RING_BOW) * depthRatio * (halfWidths[0] + halfWidths[1]) / 2
        return wrapRows(centers, halfWidths[0], halfWidths[1], np.tile(n, (rows, 1)),
                        (b - a) / length, bow, cols)

    def drawGrid(self, frame, grid, mask):
        rows, cols = grid.shape[:2]
        g = np.round(grid).reshape(-1, 2).astype(np.int32)
        h, w = frame.shape[:2]
        x, y = g[:, 0], g[:, 1]
        # only keep points inside the frame and on the body outline, like draw_landmarks skips off-frame points
        keep = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        if mask is not None:
            keep &= mask[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)] > 0.5
        starts, ends = gridEdges(rows, cols)
        both = keep[starts] & keep[ends]
        lines = np.stack([g[starts[both]], g[ends[both]]], axis=1)
        if len(lines):
            cv.polylines(frame, list(lines), False, MESH_COLOR, 1, cv.LINE_AA)
        for px, py in g[keep]:
            cv.circle(frame, (int(px), int(py)), 2, MESH_COLOR, 1)

    def findPose(self, frame):
            RGBimg = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = self.holistic.process(RGBimg)
            if result.pose_landmarks:
                h, w, c = frame.shape
                lms = result.pose_landmarks.landmark
                pts = np.array([(lm.x * w, lm.y * h) for lm in lms])
                # pose z is relative depth on roughly the same scale as x
                depth = np.array([lm.z * w for lm in lms])
                visible = [lm.visibility >= 0.5 for lm in lms]
                # parts whose joints are hidden are skipped, so they are not drawn at guessed positions
                mask = result.segmentation_mask
                # side-on, the far shoulder is hidden but its position is still estimated, so one is enough
                if visible[P.LEFT_SHOULDER] or visible[P.RIGHT_SHOULDER]:
                    self.drawGrid(frame, self.torsoGrid(mask, pts, depth), mask)
                    self.drawGrid(frame, self.neckGrid(mask, pts, depth), mask)
                for a, b, rows, cols in LIMBS:
                    if visible[a] and visible[b]:
                        tilt = self.smoothTilt((a, b), towardCamera(pts[a], pts[b], depth[a], depth[b]))
                        self.drawGrid(frame, self.limbGrid(mask, pts[a], pts[b], tilt, rows, cols), mask)
            for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
                if hand:
                    self.mpDraw.draw_landmarks(frame, hand, self.mpHolistic.HAND_CONNECTIONS,
                                               self.drawSpec, self.drawSpec)
            return frame

if __name__ == "__main__":
        useCamera = "--camera" in sys.argv
        cap = cv.VideoCapture(0 if useCamera else VIDEO_PATH)
        ctime = 0
        ptime = 0
        detector = posedetection()
        while True:
            success, frame = cap.read()
            if not success:
                if useCamera:
                    print("Could not read from the camera, so check camera permissions!")
                break
            if useCamera:
                frame = cv.flip(frame, 1)
            frame = detector.findPose(frame)
            ctime = time.time()
            fps = 1 / (ctime - ptime) if ctime != ptime else 0
            ptime = ctime
            cv.putText(frame, f"FPS: {str(int(fps))}", (15, 30), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            cv.imshow('image',frame)
            if cv.waitKey(20) & 0xFF==ord('d'):
                break
        cap.release()
        cv.destroyAllWindows()
