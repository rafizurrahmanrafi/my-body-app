"""
pose_measure.py
----------------
MediaPipe Pose (classic `mp.solutions.pose` API) ব্যবহার করে শরীরের ল্যান্ডমার্ক
ডিটেক্ট করা এবং সেখান থেকে শার্ট/টি-শার্টের মাপ বের করা হয়।

নোট: mediapipe 1.0+ ভার্সনে নতুন একটা "Tasks API" (PoseLandmarker) এসেছে, কিন্তু
সেটা এখনো নতুন এবং Streamlit Cloud-এর লিনাক্স কন্টেইনারে C-bindings লোড করতে গিয়ে
সমস্যা করছে (দ্রুত পরিবর্তনশীল, এখনো পুরোপুরি স্টেবল না)। তাই এখানে ইচ্ছাকৃতভাবে
mediapipe==0.10.14 (requirements.txt এ পিন করা) এবং তার পুরনো, বহু বছর ধরে
প্রমাণিত `mp.solutions.pose` API ব্যবহার করা হয়েছে।

গুরুত্বপূর্ণ নোট:
- সাধারণ ওয়েবক্যাম থেকে depth information পাওয়া যায় না, তাই chest circumference
  (বুকের ঘের) সরাসরি মাপা সম্ভব না — সেটা shoulder width থেকে একটা প্রচলিত
  ellipse-approximation ফর্মুলা দিয়ে অনুমান করা হয়।
- shoulder width, length, sleeve length ভালো ক্যালিব্রেশনে মোটামুটি নির্ভুল হয়।
"""

import mediapipe as mp
import numpy as np
import cv2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# MediaPipe Pose ল্যান্ডমার্ক ইনডেক্স
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_WRIST, R_WRIST = 15, 16
L_ANKLE, R_ANKLE = 27, 28
L_EAR, R_EAR = 7, 8


class PoseMeasurer:
    def __init__(self, height_cm: float = 170.0):
        self.height_cm = height_cm
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, img_bgr: np.ndarray):
        """একটা ফ্রেম থেকে pose landmark বের করে dict আকারে রিটার্ন করে (pixel coords + visibility)।
        দ্বিতীয় রিটার্ন ভ্যালু raw mediapipe landmark অবজেক্ট, যেটা draw() এ স্কেলিটন আঁকতে লাগে।"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(img_rgb)
        if result.pose_landmarks:
            h, w, _ = img_bgr.shape
            pts = {}
            for idx, lm in enumerate(result.pose_landmarks.landmark):
                pts[idx] = (lm.x * w, lm.y * h, lm.visibility)
            return pts, result.pose_landmarks
        return None, None

    def draw(self, img_bgr: np.ndarray, raw_landmarks=None) -> np.ndarray:
        """স্ক্রিনে স্কেলিটন ওভারলে করে দেখানোর জন্য (ফিডব্যাক হিসেবে)।
        raw_landmarks আগে থেকে দিলে আবার নতুন করে detect করবে না (ডাবল কাজ এড়াতে)।"""
        annotated = img_bgr.copy()
        if raw_landmarks is None:
            _, raw_landmarks = self.process_frame(img_bgr)
        if raw_landmarks is not None:
            mp_drawing.draw_landmarks(annotated, raw_landmarks, mp_pose.POSE_CONNECTIONS)
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
