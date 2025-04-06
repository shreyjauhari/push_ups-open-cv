import cv2
import mediapipe as mp
import numpy as np
from math import degrees, acos
from tkinter import *
from PIL import Image, ImageTk
import time

# ----------- Push-Up Counter Class -----------
class PushUpCounter:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.count = 0
        self.position = None
        self.landmarks = []
        self.width = None
        self.height = None

    def process(self, frame):
        self.height, self.width, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            self.landmarks = results.pose_landmarks.landmark
            self._update_count()

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )

        return frame

    def _get_point(self, idx):
        lm = self.landmarks[idx]
        return np.array([int(lm.x * self.width), int(lm.y * self.height)])

    def _calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = acos(np.clip(np.dot((a - b), (c - b)) / 
                              (np.linalg.norm(a - b) * np.linalg.norm(c - b)), -1.0, 1.0))
        return degrees(radians)

    def _update_count(self):
        if len(self.landmarks) > 16:
            r_shoulder = self._get_point(12)
            r_elbow = self._get_point(14)
            r_wrist = self._get_point(16)

            l_shoulder = self._get_point(11)
            l_elbow = self._get_point(13)
            l_wrist = self._get_point(15)

            right_angle = self._calculate_angle(r_shoulder, r_elbow, r_wrist)
            left_angle = self._calculate_angle(l_shoulder, l_elbow, l_wrist)

            if right_angle > 160 and left_angle > 160:
                self.position = "up"
            if right_angle < 90 and left_angle < 90 and self.position == "up":
                self.count += 1
                self.position = "down"
                print(f"✅ Push-up Count: {self.count}")

    def get_count(self):
        return self.count

# ----------- ESC Button Area -----------
exit_button = (20, 20, 120, 60)

# ----------- GUI + Camera Integration -----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

counter = PushUpCounter()
cap = cv2.VideoCapture(0)

root = Tk()
root.title("Push-Up Tracker")

video_label = Label(root)
video_label.grid(row=0, column=0, padx=10, pady=10)

count_label = Label(root, text="Push-Ups: 0", font=("Helvetica", 20))
count_label.grid(row=1, column=0)

def exit_app():
    cap.release()
    root.destroy()

def key_event(event):
    if event.keysym == 'Escape':
        print("🔑 ESC key pressed! Exiting...")
        exit_app()

root.bind('<Escape>', key_event)

# ----------- Main Frame Update Loop -----------
def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame = counter.process(frame)

    # ESC Button UI
    x, y, w_btn, h_btn = exit_button
    cv2.rectangle(frame, (x - 2, y - 2), (x + w_btn + 2, y + h_btn + 2), (0, 0, 0), 4)
    cv2.rectangle(frame, (x, y), (x + w_btn, y + h_btn), (0, 0, 255), cv2.FILLED)
    cv2.putText(frame, "ESC", (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # ESC Gesture via Hand
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(rgb)
    if hands_results.multi_hand_landmarks:
        for handLms in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            index_tip = handLms.landmark[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            if x < cx < x + w_btn and y < cy < y + h_btn:
                print("👋 Exit via hand gesture")
                exit_app()

    # Update Tkinter image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Update counter label
    count_label.configure(text=f"Push-Ups: {counter.get_count()}")

    root.after(10, update_frame)

# ----------- Start Loop -----------
update_frame()
root.mainloop()
