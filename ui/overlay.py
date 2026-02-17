import cv2
from config import IMAGE_PATHS

class OverlayUI:
    @staticmethod
    def draw(frame, action):
        cv2.putText(
            frame,
            f"Detected: {action}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        if action and action in IMAGE_PATHS:
            img = cv2.imread(IMAGE_PATHS[action])
            if img is not None:
                img = cv2.resize(img, (400, 400))
                frame[20:420, 20:420] = img
