import streamlit as st
import cv2
import numpy as np
import math
import time
import av

# MediaPipe স্ট্যান্ডার্ড ইমপোর্ট (লাল দাগ দূর করার জন্য)
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# পেজ কনফিগারেশন
st.set_page_config(page_title="AI Body Measurement", layout="centered", page_icon="📏")

st.title("📏 AI Body Measurement & Size Predictor")
st.write("ক্যামেরার সামনে সোজা হয়ে দাঁড়ান যাতে মাথা থেকে পা পর্যন্ত দেখা যায়। ২০ সেকেন্ড মেজারমেন্ট নেওয়ার পর নিচে ফাইনাল সাইজ দেখতে পাবেন।")

# Pose Detector ইনিশিয়ালাইজেশন
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def calculate_distance(p1, p2, w, h):
    return math.hypot((p2.x - p1.x) * w, (p2.y - p1.y) * h)

# সেশন স্টেট সেটআপ
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = {'shoulder': '--', 'length': '--', 'sleeve': '--', 'size': '--'}

class PoseTransformer(VideoProcessorBase):
    def __init__(self):
        self.start_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        img_h, img_w, _ = img.shape
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, int(20 - elapsed_time))

        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

            ankle_y = (l_ankle.y + r_ankle.y) / 2
            body_height_px = abs(ankle_y - nose.y) * img_h

            if body_height_px > 100:
                pixels_per_inch = body_height_px / 66.14  # Average height scaling (5ft 6in)

                shoulder_width_in = calculate_distance(l_shoulder, r_shoulder, img_w, img_h) / pixels_per_inch
                mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
                mid_hip_y = (l_hip.y + r_hip.y) / 2
                shirt_length_in = (abs(mid_hip_y - mid_shoulder_y) * img_h) / pixels_per_inch
                sleeve_length_in = calculate_distance(l_shoulder, l_wrist, img_w, img_h) / pixels_per_inch

                approx_chest_in = shoulder_width_in * 2.15
                size = "S" if approx_chest_in < 38 else "M" if approx_chest_in < 40 else "L" if approx_chest_in < 42 else "XL"

                # ডাটা আপডেট
                st.session_state['metrics'] = {
                    'shoulder': f"{shoulder_width_in:.1f}",
                    'length': f"{shirt_length_in:.1f}",
                    'sleeve': f"{sleeve_length_in:.1f}",
                    'size': size
                }

                # ভিজ্যুয়াল টেক্সট
                cv2.putText(img, f"Timer: {remaining_time}s", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(img, f"Shoulder: {shoulder_width_in:.1f} in", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Length: {shirt_length_in:.1f} in", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Size: {size}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(img, "Step back to show full body", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC সার্ভার কনফিগারেশন
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ক্যামেরা স্ট্রিমিং ইউআই
webrtc_streamer(
    key="body-measurement-stream",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=PoseTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.divider()

# মেজারমেন্ট রেজাল্ট ড্যাশবোর্ড
st.subheader("🎯 আপনার চূড়ান্ত মেজারমেন্ট ও সাইজ:")

m = st.session_state['metrics']
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Shoulder (কাঁধ)", value=f"{m['shoulder']} in")
    st.metric(label="Sleeve Length (হাতা)", value=f"{m['sleeve']} in")

with col2:
    st.metric(label="Shirt Length (লম্বা)", value=f"{m['length']} in")
    st.metric(label="Recommended Size", value=m['size'])