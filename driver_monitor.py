
import time
import threading


import cv2
import json
import mediapipe as mp
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

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
    """Publishes voice-alert events for the browser to speak."""

    def __init__(self):
        self.last_alert_time = {}
        self.cooldown = 4
        self.event_id = 0
        self.latest_event = None
        self.lock = threading.Lock()

    def alert(self, alert_type, item):
        message, language = item
        current_time = time.time()

        with self.lock:
            last_time = self.last_alert_time.get(alert_type, 0)

            if current_time - last_time < self.cooldown:
                return

            self.last_alert_time[alert_type] = current_time
            self.event_id += 1

            self.latest_event = {
                "id": self.event_id,
                "type": alert_type,
                "message": message,
                "language": language,
                "timestamp": current_time,
            }

    def get_latest_event(self):
        with self.lock:
            return self.latest_event

    def reset(self):
        with self.lock:
            self.last_alert_time.clear()
            self.latest_event = None


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
                "alert": self.voice.get_latest_event(),
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
    st_autorefresh(
        interval=1000,
        key="driver_monitor_refresh",
    )
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
        alert_event = current.get("alert") or {}

        alert_data = json.dumps(alert_event)

        components.html(
            f"""
            <div style="font-family: Arial; padding: 10px;">
                <button id="voiceButton"
                    style="padding: 10px 18px; cursor: pointer;">
                    Enable Voice Alerts
                </button>
                <span id="voiceStatus">
                    Voice alerts are disabled
                </span>
            </div>

            <script>
                const button = document.getElementById("voiceButton");
                const status = document.getElementById("voiceStatus");
                const eventData = {alert_data};

                const enabledKey = "drivesafe_voice_enabled";
                const spokenKey = "drivesafe_last_spoken";

                let enabled = localStorage.getItem(enabledKey) === "true";
                let lastSpoken = Number(
                    localStorage.getItem(spokenKey) || "0"
                );

                function updateButton() {{
                    button.textContent = enabled
                        ? "Voice Alerts Enabled"
                        : "Enable Voice Alerts";

                    status.textContent = enabled
                        ? "Listening for driver warnings"
                        : "Voice alerts are disabled";
                }}

                button.onclick = function() {{
                    enabled = !enabled;
                    localStorage.setItem(enabledKey, enabled);
                    updateButton();

                    if (enabled) {{
                        window.speechSynthesis.cancel();
                    }}
                }};

                updateButton();

                if (enabled && eventData.id > lastSpoken) {{
                    const speech = new SpeechSynthesisUtterance(
                        eventData.message
                    );

                    const languages = {{
                        "English": "en-IN",
                        "Telugu": "te-IN",
                        "Hindi": "hi-IN"
                    }};

                    speech.lang = languages[eventData.language] || "en-IN";

                    window.speechSynthesis.speak(speech);

                    lastSpoken = eventData.id;
                    localStorage.setItem(spokenKey, lastSpoken);
                }}
            </script>
            """,
            height=75,
        )

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
