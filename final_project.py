"""
HonkSense: Real-Time Indian Horn & Driver Emotion Decoder
final_project.py
"""

import time
import threading
import queue
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import sounddevice as sd
import cv2
import librosa


# ----------------------------------------------------------------------
# Configuration & label space
# ----------------------------------------------------------------------

# Audio (horn) labels
HORN_LABELS = [
    "polite_overtake_request",
    "impatience_in_traffic",
    "strong_warning",
    "panic_emergency",
    "ambient_noise/not_a_honk",
]

# Driver emotion labels
EMOTION_LABELS = [
    "calm",
    "focused",
    "stressed",
    "angry",
    "fearful",
    "drowsy/low_energy",
]

# Audio settings
AUDIO_SAMPLE_RATE = 16000          # Hz
AUDIO_WINDOW_SEC = 1.0             # analysis window
AUDIO_HOP_SEC = 0.5                # 50% overlap
HORN_MIN_SEC = 0.25                # minimum horn duration
HORN_RMS_THRESH = 0.03             # energy threshold (tune for your mic)

# Video settings
VIDEO_FPS_TARGET = 10              # approximate camera fps (informal)


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-8)


def now_timestamp() -> float:
    """Return current time in seconds for alignment."""
    return time.time()


def to_iso_time(t: float) -> str:
    """Convert unix timestamp to human-readable time string."""
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"


# ----------------------------------------------------------------------
# Audio pipeline: continuous capture + horn detection
# ----------------------------------------------------------------------

class AudioStream:
    """
    Continuous microphone capture into fixed-size chunks using sounddevice.
    Implements overlapping analysis windows via an internal ring buffer.
    """

    def __init__(self, sample_rate: int, window_sec: float, hop_sec: float):
        self.sample_rate = sample_rate
        self.window_len = int(sample_rate * window_sec)
        self.hop_len = int(sample_rate * hop_sec)

        self.buffer = queue.Queue(maxsize=100)
        self._stream = None
        self._stop_event = threading.Event()

        # Internal ring buffer to implement overlapping windows
        self._ring = np.zeros(self.window_len, dtype=np.float32)
        self._ring_filled = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        data = indata.copy().reshape(-1)

        # Slide ring buffer
        n = len(data)
        if n >= self.window_len:
            # If data is longer than window, just keep last window_len samples
            self._ring = data[-self.window_len :]
            self._ring_filled = True
        else:
            # Shift left and append new samples
            self._ring = np.roll(self._ring, -n)
            self._ring[-n:] = data
            self._ring_filled = True

        if self._ring_filled:
            ts = now_timestamp()
            try:
                self.buffer.put_nowait((ts, self._ring.copy()))
            except queue.Full:
                pass  # drop if consumer is too slow

    def start(self, device: Optional[int] = None):
        self._stop_event.clear()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=self.hop_len,
            device=device,
        )
        self._stream.start()

    def read(self, timeout: float = 1.0) -> Optional[Tuple[float, np.ndarray]]:
        try:
            return self.buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class HornDetector:
    """
    Simple horn detector using RMS energy and duration.
    In a real system, replace with a trained horn-event detector.
    """

    def __init__(self, sample_rate: int, rms_thresh: float, min_sec: float):
        self.sample_rate = sample_rate
        self.rms_thresh = rms_thresh
        self.min_samples = int(sample_rate * min_sec)
        self._current_segment = []
        self._current_start_ts: Optional[float] = None

    def _is_horn_window(self, audio: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(audio**2) + 1e-9))
        return rms > self.rms_thresh

    def process(
        self, ts: float, audio_window: np.ndarray
    ) -> Optional[Tuple[float, np.ndarray]]:
        """
        Update with a new audio window.
        Returns (event_timestamp, segment) when a horn event ends, else None.
        """
        if self._is_horn_window(audio_window):
            if not self._current_segment:
                self._current_start_ts = ts
            self._current_segment.append(audio_window)
            return None
        else:
            if self._current_segment:
                segment = np.concatenate(self._current_segment)
                event_ts = self._current_start_ts or ts
                self._current_segment = []
                self._current_start_ts = None

                if len(segment) >= self.min_samples:
                    return (event_ts, segment)
        return None


def extract_logmel(audio: np.ndarray, sr: int, n_mels: int = 64) -> np.ndarray:
    """
    Convert raw audio into log-mel spectrogram (for HornIntentNet).
    """
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=256,
    )
    logS = librosa.power_to_db(S + 1e-9)
    return logS.astype(np.float32)


# ----------------------------------------------------------------------
# Video pipeline: face capture for emotion
# ----------------------------------------------------------------------

class VideoStream:
    """
    Simple OpenCV camera capture.
    """

    def __init__(self, cam_index: int = 0):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            # Gracefully degrade if no camera is available
            print("Warning: Could not open camera; video will be disabled.")
            self.cap = None

    def read(self) -> Optional[Tuple[float, np.ndarray]]:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        ts = now_timestamp()
        return ts, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


class FaceExtractor:
    """
    Basic face detector using OpenCV Haar cascade.
    Replace with MediaPipe or a better detector if you like.
    """

    def __init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def extract_face(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = frame_bgr[y : y + h, x : x + w]
        return face


# ----------------------------------------------------------------------
# Stub models: HornIntentNet and DriverEmotionNet
# Replace these with real trained models later.
# ----------------------------------------------------------------------

class HornIntentNet:
    """
    Stub implementation of horn intent classifier.
    Input: log-mel spectrogram (2D array).
    Output: probability distribution over HORN_LABELS.
    """

    def __init__(self):
        # TODO: load real model weights here (e.g., PyTorch)
        pass

    def predict_proba(self, logmel: np.ndarray) -> np.ndarray:
        """
        Return softmax probabilities over HORN_LABELS.
        This is a simple heuristic stub; replace with real model inference.
        """
        duration = logmel.shape[1] * 256 / AUDIO_SAMPLE_RATE  # rough seconds
        energy = float(np.mean(logmel))

        scores = np.zeros(len(HORN_LABELS), dtype=np.float32)

        # Heuristic mapping just to simulate behavior
        if duration < 0.4:
            scores[0] = 2.5  # polite_overtake_request
            scores[2] = 1.0  # strong_warning
        elif duration < 1.2:
            scores[1] = 2.0  # impatience_in_traffic
            scores[2] = 1.5  # strong_warning
        else:
            scores[3] = 3.0  # panic_emergency
            scores[2] = 1.0  # strong_warning

        # Penalize everything if energy is extremely low -> ambient
        if energy < -60:
            scores = np.zeros_like(scores)
            scores[-1] = 3.0  # ambient_noise/not_a_honk

        scores += 0.2 * (energy / 50.0)
        scores += 0.3 * np.random.randn(len(scores)).astype(np.float32)
        return softmax(scores)


class DriverEmotionNet:
    """
    Stub implementation of driver facial emotion classifier.
    Input: face crop (BGR image) or None.
    Output: probability distribution over EMOTION_LABELS.
    """

    def __init__(self):
        # TODO: load real face emotion model here
        pass

    def predict_proba(self, face_bgr: Optional[np.ndarray]) -> np.ndarray:
        if face_bgr is None:
            # No face visible: mostly focused/neutral
            scores = np.array([1.0, 1.5, 0.5, 0.2, 0.2, 0.6], dtype=np.float32)
            return softmax(scores)

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        scores = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
        # Very low variation -> calm / drowsy
        if std_val < 20:
            scores[0] = 2.0  # calm
            scores[5] = 1.4  # drowsy/low_energy
        # Bright and varied -> stressed / angry
        elif mean_val > 130:
            scores[2] = 2.0  # stressed
            scores[3] = 1.5  # angry
        else:
            scores[1] = 2.0  # focused
            scores[2] = 1.5  # stressed
            scores[4] = 0.8  # fearful

        scores += 0.3 * np.random.randn(len(scores)).astype(np.float32)
        return softmax(scores)


# ----------------------------------------------------------------------
# Fusion & output structures
# ----------------------------------------------------------------------

@dataclass
class HonkEvent:
    timestamp: float
    horn_probs: Dict[str, float]
    emotion_probs: Dict[str, float]
    audio_duration_sec: float
    horn_rms: float
    face_detected: bool
    frame_timestamp: Optional[float]


class FusionEngine:
    """
    Simple fusion: align horn event with nearest emotion snapshot.
    Currently just pairs them; later you can add smarter weighting.
    """

    def __init__(self, max_time_diff_sec: float = 1.0):
        self.max_time_diff_sec = max_time_diff_sec

    def find_nearest_emotion(
        self,
        event_ts: float,
        emotion_buffer: "list[Tuple[float, np.ndarray]]",
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        if not emotion_buffer:
            return None, None
        # Pick frame with minimum |t - event_ts|
        best_ts, best_vec = min(
            emotion_buffer,
            key=lambda item: abs(item[0] - event_ts),
        )
        if abs(best_ts - event_ts) <= self.max_time_diff_sec:
            return best_ts, best_vec
        return None, None

    def build_event(
        self,
        event_ts: float,
        horn_probs: np.ndarray,
        horn_duration: float,
        horn_rms: float,
        emotion_ts: Optional[float],
        emotion_probs: Optional[np.ndarray],
    ) -> HonkEvent:
        if emotion_probs is None:
            # fallback: uniform if no emotion available
            emotion_probs = np.ones(len(EMOTION_LABELS), dtype=np.float32)
            emotion_probs /= emotion_probs.sum()

        horn_dict = {
            label: float(p) for label, p in zip(HORN_LABELS, horn_probs)
        }
        emo_dict = {
            label: float(p) for label, p in zip(EMOTION_LABELS, emotion_probs)
        }

        return HonkEvent(
            timestamp=event_ts,
            horn_probs=horn_dict,
            emotion_probs=emo_dict,
            audio_duration_sec=horn_duration,
            horn_rms=horn_rms,
            face_detected=emotion_ts is not None,
            frame_timestamp=emotion_ts,
        )


# ----------------------------------------------------------------------
# Main application loop
# ----------------------------------------------------------------------

class HonkSenseApp:
    """
    Main real-time app:
    - Captures audio & video.
    - Detects horn events, classifies intent and emotion.
    - Fuses & prints JSON-like output, runs until user quits.
    """

    def __init__(self, audio_device: Optional[int] = None, cam_index: int = 0):
        # Streams
        self.audio_stream = AudioStream(
            sample_rate=AUDIO_SAMPLE_RATE,
            window_sec=AUDIO_WINDOW_SEC,
            hop_sec=AUDIO_HOP_SEC,
        )
        self.video_stream = VideoStream(cam_index=cam_index)

        # Processing
        self.horn_detector = HornDetector(
            sample_rate=AUDIO_SAMPLE_RATE,
            rms_thresh=HORN_RMS_THRESH,
            min_sec=HORN_MIN_SEC,
        )
        self.face_extractor = FaceExtractor()

        # Models
        self.horn_model = HornIntentNet()
        self.emotion_model = DriverEmotionNet()
        self.fusion = FusionEngine(max_time_diff_sec=1.0)

        self.audio_device = audio_device

        # Emotion buffer: (timestamp, emotion_probs)
        self.emotion_buffer: list[Tuple[float, np.ndarray]] = []

    def _format_probs(self, probs: Dict[str, float]) -> str:
        lines = []
        for name, p in probs.items():
            lines.append(f"  - {name:<22}: {p:0.2f}")
        return "\n".join(lines)

    def _print_event(self, event: HonkEvent):
        ts_str = to_iso_time(event.timestamp)
        print(f"\n[{ts_str}]")
        print(
            f"Horn detected (RMS={event.horn_rms:0.3f}, "
            f"duration={event.audio_duration_sec:0.2f}s)"
        )
        print("Horn intent:")
        print(self._format_probs(event.horn_probs))
        print("\nDriver emotion:")
        print(self._format_probs(event.emotion_probs))

    def run(self):
        print("Starting HonkSense real-time loop.")
        print("Press 'q' in the video window or Ctrl+C in terminal to exit.\n")

        try:
            self.audio_stream.start(device=self.audio_device)
            audio_ok = True
        except Exception as exc:
            print(f"Warning: Could not start audio input: {exc}")
            print("Honk detection will be disabled; video/emotion will still run.")
            audio_ok = False

        try:
            while True:
                # 1. Read one audio window (if audio is available)
                if audio_ok:
                    audio_item = self.audio_stream.read(timeout=0.1)
                    if audio_item is not None:
                        ts_audio, audio_window = audio_item

                        # Run horn detection on this window
                        horn_result = self.horn_detector.process(ts_audio, audio_window)
                        if horn_result is not None:
                            event_ts, segment = horn_result
                            if len(segment) > 0:
                                # Extract log-mel features
                                logmel = extract_logmel(segment, sr=AUDIO_SAMPLE_RATE)
                                horn_probs = self.horn_model.predict_proba(logmel)

                                # Compute simple features
                                horn_rms = float(
                                    np.sqrt(np.mean(segment**2) + 1e-9)
                                )
                                horn_duration = len(segment) / AUDIO_SAMPLE_RATE

                                # Find nearest emotion snapshot
                                emo_ts, emo_vec = self.fusion.find_nearest_emotion(
                                    event_ts, self.emotion_buffer
                                )

                                event = self.fusion.build_event(
                                    event_ts=event_ts,
                                    horn_probs=horn_probs,
                                    horn_duration=horn_duration,
                                    horn_rms=horn_rms,
                                    emotion_ts=emo_ts,
                                    emotion_probs=emo_vec,
                                )
                                self._print_event(event)

                # 2. Read latest video frame
                video_item = self.video_stream.read()
                if video_item is not None:
                    ts_frame, frame = video_item

                    face = self.face_extractor.extract_face(frame)
                    emotion_probs = self.emotion_model.predict_proba(face)
                    self.emotion_buffer.append((ts_frame, emotion_probs))

                    # Keep emotion buffer small (e.g., last 5 seconds)
                    cutoff = now_timestamp() - 5.0
                    self.emotion_buffer = [
                        (t, v) for (t, v) in self.emotion_buffer if t >= cutoff
                    ]

                    # Draw simple visualization on frame
                    if face is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_extractor.cascade.detectMultiScale(
                            gray, 1.3, 5
                        )
                        if len(faces) > 0:
                            x, y, w, h = max(
                                faces, key=lambda f: f[2] * f[3]
                            )
                            cv2.rectangle(
                                frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                            )

                    cv2.imshow("HonkSense - Driver Camera", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except KeyboardInterrupt:
            print("\nStopped by user (KeyboardInterrupt).")
        finally:
            # Safely stop/cleanup even if devices failed to start
            try:
                self.audio_stream.stop()
            except Exception:
                pass
            try:
                self.video_stream.release()
            except Exception:
                pass
            cv2.destroyAllWindows()
            print("HonkSense shut down cleanly.")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Adjust indices if you have multiple devices
    app = HonkSenseApp(audio_device=None, cam_index=0)
    app.run()
