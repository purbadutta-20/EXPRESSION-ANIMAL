import cv2
import time

from config import CAMERA_INDEX, HOLD_TIME
from detectors.hand_detector import HandDetector
from detectors.face_detector import FaceDetector
from ui.overlay import OverlayUI


class ExpressionAnimalApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.hand_detector = HandDetector()
        self.face_detector = FaceDetector()
        self.locked_action = None
        self.action_start_time = None

    def update_action(self, detected, now):
        if detected:
            if detected != self.locked_action:
                self.locked_action = detected
                self.action_start_time = now
            elif now - self.action_start_time >= HOLD_TIME:
                return self.locked_action
        else:
            self.locked_action = None

        return self.locked_action

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time()

            detected = self.hand_detector.detect(rgb)
            if not detected:
                detected = self.face_detector.detect(rgb)

            final_action = self.update_action(detected, now)
            OverlayUI.draw(frame, final_action)

            cv2.imshow("Expression → animals (PRO)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cleanup()

    def cleanup(self):
        self.cap.release()
        self.hand_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ExpressionAnimalApp().run()
