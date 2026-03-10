"""
main.py
Entry point — wires lidar.py + yolo_live.py + fusion.py together.

  python main.py            # real hardware
  python main.py --no-display  # headless (no cv2 window, good for SSH)
"""

import argparse
import os
import time
import threading
import cv2

from lidar_new import LidarReader, _FusionSmoother
from yolo_live import CameraDetector
from fusion    import SensorFusion, FusedObstacle
from typing    import List

PATH_CLEAR_DEG = 20.0   # mirror fusion.PATH_ANGLE_DEG

# ── optional TTS ──────────────────────────────────────────────────────────────
try:
    import pyttsx3
    _tts = pyttsx3.init()
    _tts.setProperty("rate", 165)
    TTS_OK = True
except Exception:
    TTS_OK = False

TTS_COOLDOWN_S  = 3.0
_tts_last: dict = {}
_tts_lock       = threading.Lock()
TTS_CLEAR_PHRASE = "path clear"
_tts_last_phrase = ""
_tts_last_time   = 0.0
_tts_clear_spoken = False


def speak(phrase: str):
    if not TTS_OK:
        return
    now = time.time()
    with _tts_lock:
        if now - _tts_last.get(phrase, 0) < TTS_COOLDOWN_S:
            return
        _tts_last[phrase] = now
    threading.Thread(target=_say, args=(phrase,), daemon=True).start()


def _say(phrase: str):
    try:
        _tts.say(phrase)
        _tts.runAndWait()
    except Exception:
        pass


def _suggest_path(obstacles: List[FusedObstacle]) -> str:
    """Recommend a direction based on the current fused obstacle set."""
    relevant = [o for o in obstacles if o.distance_m <= 2.5]
    if not relevant:
        return "Path clear — continue ahead"

    ahead_blocked = any(o for o in relevant if o.in_path)
    if not ahead_blocked:
        return "Continue ahead"

    closest_ahead = min((o for o in relevant if o.in_path), key=lambda o: o.distance_m)
    if closest_ahead.distance_m < 0.5:
        return "Stop — obstacle too close"

    left_obs  = [o for o in relevant if o.angle_deg < -PATH_CLEAR_DEG]
    right_obs = [o for o in relevant if o.angle_deg >  PATH_CLEAR_DEG]
    left_clear  = min((o.distance_m for o in left_obs),  default=6.0)
    right_clear = min((o.distance_m for o in right_obs), default=6.0)

    if left_clear >= 1.0 or right_clear >= 1.0:
        if left_clear >= right_clear:
            return f"Bear left — {left_clear:.1f}m clearance on left"
        else:
            return f"Bear right — {right_clear:.1f}m clearance on right"
    return "Stop — all directions blocked"


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    global _tts_last_phrase, _tts_last_time, _tts_clear_spoken
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidar-port",  default="/dev/ttyUSB0")
    parser.add_argument("--no-display",  action="store_true")
    parser.add_argument("--loop-hz",     default=10, type=int)
    args = parser.parse_args()

    if not args.no_display:
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    lidar     = LidarReader(port=args.lidar_port)
    camera    = CameraDetector()
    fuser     = SensorFusion()
    smoother  = _FusionSmoother()
    interval  = 1.0 / args.loop_hz

    lidar.start()
    camera.start()
    time.sleep(2.0)   # let threads fill their buffers
    print("✅ Running. Ctrl+C to stop.\n")

    last_print = 0.0

    try:
        while True:
            t0 = time.time()

            clusters   = lidar.get_clusters()
            detections = camera.get_detections()
            obstacles  = smoother.smooth(fuser.fuse(detections, clusters))
            alerts     = lidar.get_pending_alerts()
            for msg in alerts:
                print(f"\033[91m[ALERT] {msg}\033[0m", flush=True)

            # ── TTS for critical / in-path obstacles (every cycle) ───────────
            now = time.time()
            if obstacles:
                _tts_clear_spoken = False
                closest = min(obstacles, key=lambda o: o.distance_m)
                phrase = closest.tts_phrase
                if phrase != _tts_last_phrase:
                    if now - _tts_last_time >= 1.5:
                        print(f"[TTS] {phrase}", flush=True)
                        speak(phrase)
                        _tts_last_phrase = phrase
                        _tts_last_time = now
                elif now - _tts_last_time >= TTS_COOLDOWN_S:
                    print(f"[TTS] {phrase}", flush=True)
                    speak(phrase)
                    _tts_last_time = now
            else:
                if not _tts_clear_spoken and now - _tts_last_time >= 1.5:
                    print(f"[TTS] {TTS_CLEAR_PHRASE}", flush=True)
                    speak(TTS_CLEAR_PHRASE)
                    _tts_last_phrase = TTS_CLEAR_PHRASE
                    _tts_last_time = now
                    _tts_clear_spoken = True

            # ── Throttled console output (1 Hz) ─────────────────────────────
            if t0 - last_print >= 1.0:
                last_print = t0
                print()
                print(time.strftime("%H:%M:%S"))
                if obstacles:
                    print(f"obstacles detected: {len(obstacles)}")
                    for i, obs in enumerate(obstacles, 1):
                        print(f"  object {i}  {obs.class_name} — "
                              f"{obs.distance_m:.2f}m, "
                              f"{obs.angle_deg:+.0f}° {obs.direction}")
                    print(f"path: {_suggest_path(obstacles)}")
                else:
                    print("obstacles detected: 0")
                    print("path: Path clear — continue ahead")

            # ── Display window ────────────────────────────────────────────────
            if not args.no_display:
                frame = camera.get_annotated_frame()
                if frame is not None:
                    cv2.imshow("Obstacle Detection", frame)
                    if cv2.waitKey(1) == ord("q"):
                        break

            time.sleep(max(0.0, interval - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        lidar.stop()
        camera.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
