import mediapipe as mp
from utils.math_utils import euclidean_distance

class FaceDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def detect(self, rgb_frame):
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark

        mouth_open = euclidean_distance(
            (lm[13].x, lm[13].y),
            (lm[14].x, lm[14].y)
        )

        left_eye = euclidean_distance(
            (lm[159].x, lm[159].y),
            (lm[145].x, lm[145].y)
        )
        right_eye = euclidean_distance(
            (lm[386].x, lm[386].y),
            (lm[374].x, lm[374].y)
        )

        if mouth_open > 0.05:
            return "surprise"
        elif mouth_open > 0.025:
            return "happy"
        elif (left_eye < 0.008 and right_eye > 0.012) or \
             (right_eye < 0.008 and left_eye > 0.012):
            return "wink"

        return None
