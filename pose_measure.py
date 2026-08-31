"""
pose_measure.py
----------------
mediapipe 1.0+ ভার্সনে পুরনো `mp.solutions.pose` API সম্পূর্ণ সরিয়ে ফেলা হয়েছে।
তাই এখানে নতুন Tasks API (`PoseLandmarker`) ব্যবহার করা হয়েছে।

এই API চালাতে একটা মডেল ফাইল (.task) লাগে, যেটা pip প্যাকেজের সাথে বান্ডিল করা থাকে না —
প্রথমবার অ্যাপ চালু হলে সেটা অটোমেটিক্যালি ডাউনলোড হয়ে /tmp তে ক্যাশ হয়ে থাকে।

গুরুত্বপূর্ণ নোট:
- সাধারণ ওয়েবক্যাম থেকে depth information পাওয়া যায় না, তাই chest circumference
  (বুকের ঘের) সরাসরি মাপা সম্ভব না — সেটা shoulder width থেকে একটা প্রচলিত
  ellipse-approximation ফর্মুলা দিয়ে অনুমান করা হয়।
- shoulder width, length, sleeve length ভালো ক্যালিব্রেশনে মোটামুটি নির্ভুল হয়।
"""

import os
import time
import tempfile
import urllib.request

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

# ছোট, দ্রুত মডেল — রিয়েল-টাইম ওয়েবক্যাম প্রসেসিং এর জন্য যথেষ্ট
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(tempfile.gettempdir(), "pose_landmarker_lite.task")

# Pose landmark ইনডেক্স (mediapipe pose landmarker, ৩৩ পয়েন্ট মডেল)
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_WRIST, R_WRIST = 15, 16
L_ANKLE, R_ANKLE = 27, 28
L_EAR, R_EAR = 7, 8

# স্কেলিটন লাইন আঁকার জন্য কানেকশন (মূল connection সেট, সরাসরি হার্ডকোড করা—
# কারণ mp.solutions.pose.POSE_CONNECTIONS নতুন ভার্সনে আর নেই)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 29), (29, 31), (28, 30), (30, 32),
]


def _ensure_model() -> str:
    """মডেল ফাইল লোকালি না থাকলে একবার ডাউনলোড করে ক্যাশ করে রাখে।"""
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class PoseMeasurer:
    def __init__(self, height_cm: float = 170.0):
        self.height_cm = height_cm
        model_path = _ensure_model()

        base_options = BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._last_ts_ms = 0

    def _next_timestamp_ms(self) -> int:
        # VIDEO mode-এ টাইমস্ট্যাম্প strictly বাড়তে হয়, তাই wall-clock ব্যবহার করা হচ্ছে
        ts = int(time.time() * 1000)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts
        return ts

    def process_frame(self, img_bgr: np.ndarray):
        """একটা ফ্রেম থেকে pose landmark বের করে dict আকারে রিটার্ন করে (pixel coords + visibility)."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

        if result.pose_landmarks:
            h, w, _ = img_bgr.shape
            landmarks = result.pose_landmarks[0]  # প্রথম ডিটেক্ট হওয়া মানুষ
            pts = {}
            for idx, lm in enumerate(landmarks):
                vis = getattr(lm, "visibility", 1.0) or 1.0
                pts[idx] = (lm.x * w, lm.y * h, vis)
            return pts
        return None

    def draw(self, img_bgr: np.ndarray, pts=None) -> np.ndarray:
        """স্ক্রিনে স্কেলিটন ওভারলে করে দেখানোর জন্য (ফিডব্যাক হিসেবে)।
        pts আগে থেকে দিলে আবার নতুন করে detect করবে না (ডাবল কাজ এড়াতে)।"""
        if pts is None:
            pts = self.process_frame(img_bgr)
        annotated = img_bgr.copy()
        if pts:
            for a, b in POSE_CONNECTIONS:
                if a in pts and b in pts:
                    pa = (int(pts[a][0]), int(pts[a][1]))
                    pb = (int(pts[b][0]), int(pts[b][1]))
                    cv2.line(annotated, pa, pb, (0, 255, 0), 2)
            for idx, (x, y, vis) in pts.items():
                if vis > 0.5:
                    cv2.circle(annotated, (int(x), int(y)), 3, (0, 140, 255), -1)
        return annotated

    @staticmethod
    def _avg_point(frames_list, idx, min_visibility=0.5):
        vals = [f[idx][:2] for f in frames_list if idx in f and f[idx][2] > min_visibility]
        if not vals:
            return None
        return np.array(vals).mean(axis=0)

    @staticmethod
    def _dist(p1, p2):
        return float(np.linalg.norm(np.array(p1) - np.array(p2)))

    def compute_measurements(self, frames_list):
        """
        ২০ সেকেন্ডে সংগ্রহ করা সব ফ্রেমের landmark গুলোর গড় নিয়ে stable মাপ বের করে।
        frames_list: প্রতিটা এলিমেন্ট হলো process_frame() থেকে পাওয়া pts dict।
        """
        if not frames_list:
            return None

        shoulder_l = self._avg_point(frames_list, L_SHOULDER)
        shoulder_r = self._avg_point(frames_list, R_SHOULDER)
        hip_l = self._avg_point(frames_list, L_HIP)
        hip_r = self._avg_point(frames_list, R_HIP)
        wrist_l = self._avg_point(frames_list, L_WRIST)
        wrist_r = self._avg_point(frames_list, R_WRIST)
        ankle_l = self._avg_point(frames_list, L_ANKLE)
        ankle_r = self._avg_point(frames_list, R_ANKLE)
        ear_l = self._avg_point(frames_list, L_EAR)
        ear_r = self._avg_point(frames_list, R_EAR)

        if shoulder_l is None or shoulder_r is None:
            return None  # কাঁধ ডিটেক্ট না হলে কিছুই বের করা যাবে না

        if ear_l is None and ear_r is None:
            return None
        if ankle_l is None and ankle_r is None:
            return None

        top_y = min([p[1] for p in [ear_l, ear_r] if p is not None]) - 25  # matha-r চূড়া আনুমানিক
        bottom_y = max([p[1] for p in [ankle_l, ankle_r] if p is not None])

        pixel_height = bottom_y - top_y
        if pixel_height <= 0:
            return None

        scale = self.height_cm / pixel_height  # ১ পিক্সেল = কত cm

        # --- কাঁধের চওড়া ---
        shoulder_width_cm = self._dist(shoulder_l, shoulder_r) * scale

        # --- বুকের ঘের (আনুমানিক, ellipse approximation - Ramanujan's formula) ---
        chest_width_cm = shoulder_width_cm * 1.05
        chest_depth_cm = chest_width_cm * 0.55
        a, b = chest_width_cm / 2, chest_depth_cm / 2
        chest_circumference_cm = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

        # --- শার্টের দৈর্ঘ্য: কাঁধ-মধ্যবিন্দু থেকে কোমর/হিপ পর্যন্ত ---
        length_cm = None
        if hip_l is not None and hip_r is not None:
            shoulder_mid = (shoulder_l + shoulder_r) / 2
            hip_mid = (hip_l + hip_r) / 2
            length_cm = self._dist(shoulder_mid, hip_mid) * scale * 1.15

        # --- হাতার দৈর্ঘ্য: কাঁধ থেকে কব্জি পর্যন্ত ---
        sleeve_cm = None
        if wrist_l is not None and wrist_r is not None:
            sleeve_l = self._dist(shoulder_l, wrist_l)
            sleeve_r = self._dist(shoulder_r, wrist_r)
            sleeve_cm = ((sleeve_l + sleeve_r) / 2) * scale

        return {
            "shoulder_width": round(shoulder_width_cm, 1),
            "chest_circumference": round(chest_circumference_cm, 1),
            "length": round(length_cm, 1) if length_cm else None,
            "sleeve_length": round(sleeve_cm, 1) if sleeve_cm else None,
            "frames_used": len(frames_list),
        }
