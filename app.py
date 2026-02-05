import argparse
import collections
import sys
import time
from dataclasses import dataclass
from typing import Deque, Tuple

import cv2
import dlib
import numpy as np
import requests

try:
    from imutils import face_utils
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'imutils'. Install requirements.txt before running."
    ) from exc


@dataclass
class DetectionConfig:
    ear_threshold: float = 0.23
    ear_consec_frames: int = 20
    mar_threshold: float = 0.7
    mar_consec_frames: int = 15
    alert_window_seconds: int = 600
    alert_repeat_count: int = 3
    low_light_clahe: bool = True
    gps_endpoint: str = "https://ipinfo.io/json"


def eye_aspect_ratio(eye: np.ndarray) -> float:
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth: np.ndarray) -> float:
    vertical_1 = np.linalg.norm(mouth[2] - mouth[10])
    vertical_2 = np.linalg.norm(mouth[4] - mouth[8])
    horizontal = np.linalg.norm(mouth[0] - mouth[6])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_gps_location(endpoint: str) -> str:
    try:
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        data = response.json()
        location = data.get("loc", "unknown")
        return f"{data.get('city', 'unknown')}, {data.get('region', '')} ({location})"
    except requests.RequestException:
        return "Location unavailable"


def send_gps_alert(event_count: int, endpoint: str) -> None:
    location = get_gps_location(endpoint)
    message = (
        "[ALERT] Drowsiness detected multiple times. "
        f"Count: {event_count}. GPS: {location}"
    )
    print(message)


class DrowsinessDetector:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.ear_counter = 0
        self.mar_counter = 0
        self.drowsy_events: Deque[float] = collections.deque()
        self.alarm_active = False
        self.last_alarm_time: float | None = None

    def _prune_events(self, current_time: float) -> None:
        while self.drowsy_events and current_time - self.drowsy_events[0] > self.config.alert_window_seconds:
            self.drowsy_events.popleft()

    def register_drowsiness(self, current_time: float) -> None:
        self.drowsy_events.append(current_time)
        self._prune_events(current_time)
        if len(self.drowsy_events) >= self.config.alert_repeat_count:
            send_gps_alert(len(self.drowsy_events), self.config.gps_endpoint)

    def update_alarm(self, ear: float, mar: float, current_time: float) -> Tuple[bool, bool]:
        yawning = False
        if ear < self.config.ear_threshold:
            self.ear_counter += 1
        else:
            self.ear_counter = 0
            self.alarm_active = False

        if mar > self.config.mar_threshold:
            self.mar_counter += 1
        else:
            self.mar_counter = 0

        if self.mar_counter >= self.config.mar_consec_frames:
            yawning = True

        if self.ear_counter >= self.config.ear_consec_frames:
            if not self.alarm_active:
                self.alarm_active = True
                self.last_alarm_time = current_time
                self.register_drowsiness(current_time)

        return self.alarm_active, yawning


def preprocess_frame(frame: np.ndarray, use_clahe: bool) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Road Guardian: Drowsiness detection demo")
    parser.add_argument(
        "--shape-predictor",
        required=True,
        help="Path to dlib's 68-face-landmarks predictor file",
    )
    parser.add_argument("--source", type=int, default=0, help="Camera source index")
    parser.add_argument("--no-clahe", action="store_true", help="Disable low-light CLAHE")
    parser.add_argument("--ear", type=float, default=0.23, help="EAR threshold")
    parser.add_argument("--mar", type=float, default=0.7, help="MAR threshold")
    parser.add_argument(
        "--gps-endpoint",
        default="https://ipinfo.io/json",
        help="Endpoint for GPS/IP lookup",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = DetectionConfig(
        ear_threshold=args.ear,
        mar_threshold=args.mar,
        low_light_clahe=not args.no_clahe,
        gps_endpoint=args.gps_endpoint,
    )

    detector = DrowsinessDetector(config)

    detector_model = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (m_start, m_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    video_stream = cv2.VideoCapture(args.source)
    if not video_stream.isOpened():
        print("Unable to open camera source", file=sys.stderr)
        return 1

    try:
        while True:
            grabbed, frame = video_stream.read()
            if not grabbed:
                break

            frame = cv2.resize(frame, (640, 480))
            gray = preprocess_frame(frame, config.low_light_clahe)

            rects = detector_model(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[l_start:l_end]
                right_eye = shape[r_start:r_end]
                mouth = shape[m_start:m_end]

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                mar = mouth_aspect_ratio(mouth)

                current_time = time.time()
                alarm, yawning = detector.update_alarm(ear, mar, current_time)

                if alarm:
                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                if yawning:
                    cv2.putText(
                        frame,
                        "YAWNING DETECTED",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )

                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (480, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    f"MAR: {mar:.2f}",
                    (480, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            cv2.imshow("Road Guardian", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video_stream.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
