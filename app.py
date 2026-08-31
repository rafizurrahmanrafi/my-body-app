"""
app.py
------
Streamlit ওয়েব অ্যাপ: ক্যামেরার সামনে ২০ সেকেন্ড দাঁড়ালে T-shirt/Shirt এর মাপ বের করে দেয়।
চালানোর জন্য: streamlit run app.py
"""

import time

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from pose_measure import PoseMeasurer

st.set_page_config(page_title="Shirt Measurement App", page_icon="👕", layout="centered")

# মোবাইল/ভিন্ন নেটওয়ার্ক থেকেও ক্যামেরা কানেকশন স্টেবল রাখতে STUN সার্ভার ব্যবহার করা হচ্ছে।
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.title("👕 T-Shirt / Shirt Measurement App")
st.write(
    "ক্যামেরার সামনে সোজা হয়ে দাঁড়ান (পুরো শরীর যেন ফ্রেমে দেখা যায়), "
    "তারপর নিচের বাটনে ক্লিক করে ২০ সেকেন্ড ঠায় দাঁড়িয়ে থাকুন — আমরা আপনার শার্টের মাপ বের করে দেব।"
)



with st.expander("📌 ভালো ফলাফলের জন্য কিছু টিপস (ক্লিক করে দেখুন)"):
    st.markdown(
        """
        - ক্যামেরা থেকে **প্রায় ২-২.৫ মিটার** দূরে দাঁড়ান, যেন মাথা থেকে পা পর্যন্ত পুরো শরীর দেখা যায়।
        - **আঁটসাঁট (fitted) পোশাক** পরুন — ঢিলেঢালা পোশাকে মাপ ভুল আসবে।
        - দেয়াল/ব্যাকগ্রাউন্ড যতটা সম্ভব **সাদামাটা** রাখুন।
        - **আলো** পর্যাপ্ত থাকতে হবে, ক্যামেরার দিকে মুখ করে সোজা দাঁড়ান, হাত দুপাশে সোজা ঝুলিয়ে রাখুন (T-pose না করলেও চলবে)।
        - **সঠিক উচ্চতা** নিচে লিখুন — এটাই ক্যালিব্রেশনের একমাত্র রেফারেন্স, ভুল হলে সব মাপ ভুল আসবে।
        """
    )

st.warning(
    "⚠️ সাধারণ ওয়েবক্যামে কোনো depth sensor থাকে না, তাই **বুকের ঘের (chest circumference)** "
    "সরাসরি মাপা সম্ভব না — সেটা কাঁধের চওড়া থেকে একটা প্রচলিত approximation ফর্মুলা দিয়ে অনুমান করা হয়। "
    "কাঁধের চওড়া, দৈর্ঘ্য ও হাতার মাপ তুলনামূলক বেশি নির্ভুল। চূড়ান্ত/দর্জির কাছে অর্ডারের আগে টেপ দিয়ে যাচাই করে নিন।"
)

height_cm = st.number_input(
    "আপনার উচ্চতা (সেন্টিমিটারে) লিখুন — ক্যালিব্রেশনের জন্য জরুরি",
    min_value=100.0,
    max_value=220.0,
    value=170.0,
    step=0.5,
)

duration = st.slider("ক্যাপচার সময় (সেকেন্ড)", min_value=10, max_value=30, value=20)

camera_choice = st.radio(
    "কোন ক্যামেরা ব্যবহার করবেন?",
    ["পিছনের ক্যামেরা (Back) — মোবাইলে বেস্ট কোয়ালিটি", "সামনের ক্যামেরা (Front / Selfie)"],
    index=0,
)
facing_mode = "environment" if camera_choice.startswith("পিছনের") else "user"

if facing_mode == "environment":
    st.caption(
        "📱 মোবাইলে এই সেটিংয়ে ব্রাউজার নিজে থেকেই পিছনের ক্যামেরা খোলার চেষ্টা করবে। "
        "ফোন অন্য কাউকে ধরিয়ে ভিডিও ধারণ করুন, বা ট্রাইপড/দেয়ালে হেলান দিয়ে রাখুন যেন পুরো শরীর ফ্রেমে আসে।"
    )

# ---------------- Session state ----------------
if "measuring" not in st.session_state:
    st.session_state.measuring = False
if "results" not in st.session_state:
    st.session_state.results = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None


class VideoProcessor:
    """webrtc থেকে আসা প্রতিটা ফ্রেম প্রসেস করে, measuring চলাকালীন landmark জমা রাখে।"""

    def __init__(self):
        self.measurer = PoseMeasurer(height_cm=height_cm)
        self.frames_collected = []
        self.measuring = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        pts = self.measurer.process_frame(img)

        if self.measuring and pts is not None:
            self.frames_collected.append(pts)

        annotated = self.measurer.draw(img, pts=pts)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


ctx = webrtc_streamer(
    key="shirt-measure",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "facingMode": {"ideal": facing_mode},
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
        },
        "audio": False,
    },
    async_processing=True,
)

col1, col2 = st.columns(2)
start_btn = col1.button("▶️ মাপ নেওয়া শুরু করুন", disabled=st.session_state.measuring)
reset_btn = col2.button("🔄 রিসেট করুন")

if reset_btn:
    st.session_state.measuring = False
    st.session_state.results = None
    st.session_state.start_time = None
    if ctx.video_processor:
        ctx.video_processor.frames_collected = []
    st.rerun()

if start_btn and ctx.video_processor is not None:
    ctx.video_processor.frames_collected = []
    ctx.video_processor.measuring = True
    st.session_state.measuring = True
    st.session_state.results = None
    st.session_state.start_time = time.time()
    st.rerun()

progress_placeholder = st.empty()
status_placeholder = st.empty()

if st.session_state.measuring:
    elapsed = time.time() - st.session_state.start_time
    pct = min(elapsed / duration, 1.0)
    progress_placeholder.progress(pct)
    remaining = max(0, int(duration - elapsed))
    status_placeholder.info(f"⏱️ মাপা হচ্ছে... আর {remaining} সেকেন্ড বাকি। নড়াচড়া না করে সোজা দাঁড়িয়ে থাকুন।")

    if elapsed >= duration:
        st.session_state.measuring = False
        if ctx.video_processor:
            ctx.video_processor.measuring = False
            frames = ctx.video_processor.frames_collected
            measurer = PoseMeasurer(height_cm=height_cm)
            results = measurer.compute_measurements(frames)
            st.session_state.results = results
        time.sleep(0.3)
        st.rerun()
    else:
        time.sleep(0.3)
        st.rerun()

if st.session_state.results:
    r = st.session_state.results
    st.success(f"✅ মাপ সম্পন্ন হয়েছে ({r['frames_used']} টি ফ্রেম ব্যবহার করে)")

    st.subheader("👕 শার্ট / টি-শার্টের মাপ")

    rows = [
        ("কাঁধের চওড়া (Shoulder Width)", r["shoulder_width"]),
        ("বুকের ঘের - আনুমানিক (Chest Circumference)", r["chest_circumference"]),
        ("শার্টের দৈর্ঘ্য (Length)", r["length"]),
        ("হাতার দৈর্ঘ্য (Sleeve Length)", r["sleeve_length"]),
    ]

    st.table(
        {
            "মাপের ধরন": [name for name, _ in rows],
            "সেন্টিমিটার (cm)": [v if v is not None else "N/A" for _, v in rows],
            "ইঞ্চি (inch)": [round(v / 2.54, 1) if v is not None else "N/A" for _, v in rows],
        }
    )

    st.caption(
        "নোট: এই মাপগুলো ওয়েবক্যাম ভিত্তিক pose-estimation থেকে পাওয়া আনুমানিক মাপ। "
        "চূড়ান্ত অর্ডারের আগে অন্তত একবার টেপ দিয়ে সরাসরি মিলিয়ে দেখে নেওয়া ভালো।"
    )
elif st.session_state.results is None and not st.session_state.measuring and start_btn is False:
    st.info("ক্যামেরা চালু হলে উপরে আপনার ভিডিও দেখা যাবে। প্রস্তুত হলে 'মাপ নেওয়া শুরু করুন' চাপুন।")
