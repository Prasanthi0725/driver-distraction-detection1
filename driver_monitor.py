
import time
import threading
import queue

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import pythoncom
import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode


# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Driver Distraction Monitor",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 Driver Distraction Monitor")
st.caption("Live eye-closure and head-direction monitoring")


# ---------------- LANGUAGE MESSAGES ----------------

MESSAGES = {
    "English": {
        "eyes": "{name}, please open your eyes and keep them open. Stay alert.",
        "head": "{name}, please look straight at the road and focus.",
    },
    "Telugu": {
        "eyes": "{name}, దయచేసి మీ కళ్ళు తెరిచి ఉంచండి. అప్రమత్తంగా ఉండండి.",
        "head": "{name}, దయచేసి రోడ్డు వైపు నేరుగా చూడండి. డ్రైవింగ్‌పై దృష్టి పెట్టండి.",
    },
    "Hindi": {
        "eyes": "{name}, कृपया अपनी आँखें खोलें और खुली रखें। सतर्क रहें।",
        "head": "{name}, कृपया सड़क की ओर सीधे देखें और ध्यान दें।",
    },
}


# ---------------- VOICE ALERTS ----------------

class VoiceAlerts:
    def __init__(self):
        self.messages = queue.Queue(maxsize=3)
        self.last_alert_time = {}
        self.lock = threading.Lock()

        threading.Thread(
            target=self.speak_worker,
            daemon=True,
        ).start()

    def alert(self, alert_type, item):
        now = time.time()

        with self.lock:
            if now - self.last_alert_time.get(alert_type, 0) < 4:
                return
            self.last_alert_time[alert_type] = now

        try:
            self.messages.put_nowait(item)
        except queue.Full:
            try:
                self.messages.get_nowait()
                self.messages.task_done()
            except queue.Empty:
                pass

            try:
                self.messages.put_nowait(item)
            except queue.Full:
                pass

    def reset(self):
        with self.lock:
            self.last_alert_time.clear()

    @staticmethod
    def select_voice(engine, language):
        codes = {
            "English": ("en-", "0409", "0809"),
            "Telugu": ("te-", "044a"),
            "Hindi": ("hi-", "0449"),
        }

        try:
            voices = engine.getProperty("voices")

            for voice in voices:
                info = (
                    str(getattr(voice, "id", ""))
                    + " "
                    + str(getattr(voice, "name", ""))
                    + " "
                    + str(getattr(voice, "languages", ""))
                ).lower()

                if any(code.lower() in info for code in codes[language]):
                    engine.setProperty("voice", voice.id)
                    return

            print(
                f"No matching {language} voice installed. "
                "Using the default Windows voice."
            )

        except Exception as error:
            print("Voice selection error:", error)

    def speak_worker(self):
        pythoncom.CoInitialize()
        engine = None
        current_language = None

        try:
            while True:
                message, language = self.messages.get()

                try:
                    if engine is None:
                        engine = pyttsx3.init("sapi5")
                        engine.setProperty("rate", 145)
                        engine.setProperty("volume", 1.0)
                        current_language = None

                    if current_language != language:
                        self.select_voice(engine, language)
                        current_language = language

                    engine.say(message)
                    engine.runAndWait()

                except Exception as error:
                    print("Voice alert error:", error)

                    try:
                        if engine is not None:
                            engine.stop()
                    except Exception:
                        pass

                    engine = None
                    current_language = None

                finally:
                    self.messages.task_done()

        finally:
            pythoncom.CoUninitialize()


# ---------------- EYE DETECTION ----------------

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def distance(a, b):
    return np.linalg.norm(
        np.array(a, dtype=float) - np.array(b, dtype=float)
    )


def get_eye_ratio(landmarks, indexes, width, height):
    points = []

    for index in indexes:
        p = landmarks[index]
        points.append((p.x * width, p.y * height))

    vertical1 = distance(points[1], points[5])
    vertical2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal < 1:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


# ---------------- HEAD DIRECTION ----------------

def get_head_direction(landmarks):
    left_x = landmarks[234].x
    right_x = landmarks[454].x
    nose_x = landmarks[1].x

    face_width = right_x - left_x

    if face_width <= 0:
        return "Unknown"

    position = (nose_x - left_x) / face_width

    if position < 0.35:
        return "Left"
    if position > 0.65:
        return "Right"

    return "Forward"


# ---------------- CAMERA PROCESSOR ----------------

class DriverMonitor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.name = "Driver"
        self.language = "English"
        self.status = "Waiting for camera"
        self.eye_status = "Unknown"
        self.head_direction = "Unknown"

        self.closed_frames = 0
        self.eye_close_threshold = 0.19
        self.required_closed_frames = 15

        self.state_lock = threading.Lock()
        self.voice = VoiceAlerts()

    def set_driver(self, name, language):
        changed = False

        with self.state_lock:
            if self.name != name or self.language != language:
                changed = True

            self.name = name
            self.language = language

        if changed:
            self.voice.reset()

    def get_status(self):
        with self.state_lock:
            return {
                "status": self.status,
                "eyes": self.eye_status,
                "head": self.head_direction,
            }

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        height, width = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        with self.state_lock:
            name = self.name
            language = self.language

        warning = None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ratio = get_eye_ratio(
                landmarks, LEFT_EYE, width, height
            )
            right_ratio = get_eye_ratio(
                landmarks, RIGHT_EYE, width, height
            )
            average_ratio = (left_ratio + right_ratio) / 2.0

            if average_ratio < self.eye_close_threshold:
                self.closed_frames += 1
            else:
                self.closed_frames = 0

            if self.closed_frames >= self.required_closed_frames:
                eyes = "Closed"
                warning = "eyes"
            else:
                eyes = "Open"

            head = get_head_direction(landmarks)

            if warning is None and head in ("Left", "Right"):
                warning = "head"

            status = "WARNING" if warning else "NORMAL"

            for index in LEFT_EYE + RIGHT_EYE + [1, 234, 454]:
                p = landmarks[index]
                cv2.circle(
                    image,
                    (int(p.x * width), int(p.y * height)),
                    2,
                    (0, 255, 255),
                    -1,
                )

            cv2.putText(
                image,
                f"Eye ratio: {average_ratio:.3f}",
                (15, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        else:
            eyes = "Unknown"
            head = "Unknown"
            status = "FACE NOT DETECTED"
            self.closed_frames = 0

        with self.state_lock:
            self.eye_status = eyes
            self.head_direction = head
            self.status = status

        if warning:
            message = MESSAGES[language][warning].format(name=name)
            self.voice.alert(warning, (message, language))

        color = (0, 200, 0) if status == "NORMAL" else (0, 0, 255)

        for text, y in [
            (f"Driver: {name}", 30),
            (f"Status: {status}", 65),
            (f"Eyes: {eyes}", 100),
            (f"Head: {head}", 135),
        ]:
            cv2.putText(
                image,
                text,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color if text.startswith("Status:") else (255, 255, 255),
                2,
            )

        return frame.from_ndarray(image, format="bgr24")


# ---------------- SESSION ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "driver_name" not in st.session_state:
    st.session_state.driver_name = ""

if "language" not in st.session_state:
    st.session_state.language = "English"


# ---------------- LOGIN ----------------

if not st.session_state.logged_in:
    st.subheader("Driver Login")

    with st.form("login_form"):
        name = st.text_input("Driver name")

        language = st.selectbox(
            "Alert language",
            ["English", "Telugu", "Hindi"],
        )

        start = st.form_submit_button(
            "Start Monitoring",
            use_container_width=True,
        )

    if start:
        if not name.strip():
            st.error("Please enter the driver's name.")
        else:
            st.session_state.driver_name = name.strip()
            st.session_state.language = language
            st.session_state.logged_in = True
            st.rerun()


# ---------------- MONITORING ----------------

else:
    st.subheader(f"Welcome, {st.session_state.driver_name}")
    st.write(f"Selected language: **{st.session_state.language}**")

    st.info(
        "Allow camera access and press START. "
        "Keep your face clearly visible in the camera."
    )

    context = webrtc_streamer(
        key="driver-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=DriverMonitor,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    if context.video_processor is not None:
        context.video_processor.set_driver(
            st.session_state.driver_name,
            st.session_state.language,
        )

        current = context.video_processor.get_status()

        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Status", current["status"])
        col2.metric("Eye State", current["eyes"])
        col3.metric("Head Direction", current["head"])

        if not context.state.playing:
            st.warning("Press START in the camera panel.")

    st.divider()

    if st.button("Next Driver", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.driver_name = ""
        st.session_state.language = "English"
        st.rerun()


st.caption(
    "Prototype only. Face and eye estimates may be inaccurate. "
    "Do not rely on this application as the sole basis for driving safety."
)
