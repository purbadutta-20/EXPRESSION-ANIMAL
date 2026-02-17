import mediapipe as mp

class HandDetector:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def detect(self, rgb_frame):
        results = self.hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None

        hand = results.multi_hand_landmarks[0].landmark

        thumb_tip = hand[4]
        thumb_ip = hand[3]
        index_mcp = hand[5]

        if thumb_tip.y < thumb_ip.y < index_mcp.y:
            return "thumbs_up"

        return None

    def close(self):
        self.hands.close()
