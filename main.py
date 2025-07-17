import cv2
import os
import sys
import time
import math
import mediapipe as mp

# Suppress logs
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Define gesture mappings
vector_to_play = {
    (0, 0, 0, 0, 0): "Inside Zone",             # Closed fist
    (1, 1, 1, 1, 1): "Slants",                  # Open hand
    (1, 0, 0, 0, 0): "Read Option",             # Thumb up
    (0, 1, 1, 0, 0): "Hail Mary",               # Peace sign
    (0, 1, 0, 0, 0): "Bench",                   # Point up
    (1, 0, 0, 0, 1): "Blitz Play",              # Shaka
    (0, 1, 1, 1, 0): "Curls n' go",             # Three center fingers
    (0, 1, 0, 0, 1): "Jet Sweep Left",          # Devil Horns
    (1, 1, 0, 0, 1): "Jet Sweep Right",         # RockStar
    (1, 0, 1, 0, 0): "Screen Toss",             # Kristers Special
    (1, 1, 0, 0, 0): "Outside Zone"             # L
}

# --- Finger Detection Helpers ---
def vector_angle(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a ** 2 for a in v1))
    mag2 = math.sqrt(sum(b ** 2 for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

def is_finger_extended_by_angle(mcp, pip, tip, threshold=0.75):
    v1 = [pip.x - mcp.x, pip.y - mcp.y, pip.z - mcp.z]
    v2 = [tip.x - pip.x, tip.y - pip.y, tip.z - pip.z]
    cosine = vector_angle(v1, v2)
    return cosine > threshold

def is_thumb_extended(thumb_tip, thumb_ip, wrist):
    return abs(thumb_tip.x - wrist.x) > 0.1 and thumb_tip.y < wrist.y

def finger_states(hand_landmarks):
    lm = hand_landmarks.landmark
    thumb = 1 if is_thumb_extended(lm[mp_hands.HandLandmark.THUMB_TIP],
                                   lm[mp_hands.HandLandmark.THUMB_IP],
                                   lm[mp_hands.HandLandmark.WRIST]) else 0
    index = 1 if is_finger_extended_by_angle(lm[mp_hands.HandLandmark.INDEX_FINGER_MCP],
                                             lm[mp_hands.HandLandmark.INDEX_FINGER_PIP],
                                             lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]) else 0
    middle = 1 if is_finger_extended_by_angle(lm[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
                                              lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                                              lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) else 0
    ring = 1 if is_finger_extended_by_angle(lm[mp_hands.HandLandmark.RING_FINGER_MCP],
                                            lm[mp_hands.HandLandmark.RING_FINGER_PIP],
                                            lm[mp_hands.HandLandmark.RING_FINGER_TIP]) else 0
    pinky = 1 if is_finger_extended_by_angle(lm[mp_hands.HandLandmark.PINKY_MCP],
                                             lm[mp_hands.HandLandmark.PINKY_PIP],
                                             lm[mp_hands.HandLandmark.PINKY_TIP]) else 0
    return (thumb, index, middle, ring, pinky)

# --- Main Detection Loop ---
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    print("Scanning for hand gestures. Press Ctrl+C to exit.")

    previous_message = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            output_message = "No hand detected."

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    vector = finger_states(hand_landmarks)
                    play = vector_to_play.get(vector)

                    if play:
                        output_message = f"Gesture Detected → Football Play: {play}"
                    else:
                        output_message = f"Gesture Vector {vector} not recognized."

            # --- Clean, One-Line Output ---
            if output_message != previous_message:
                sys.stdout.write('\r' + ' ' * len(previous_message))   # Clear line
                sys.stdout.write('\r' + output_message)                # New message
                sys.stdout.flush()
                previous_message = output_message

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting program.")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
