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

from lidar_new import (
    LidarReader,
    _FusionSmoother,
    ALERT_ENTER_M,
    DANGER_DIST_M,
    SAFE_DIST_M,
)
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

TTS_UPDATE_INTERVAL_S = 3.0
TTS_DANGER_COOLDOWN_S = 1.0
TTS_ALERT_DIST_M = 3.0
_tts_last_time = 0.0
_tts_lock = threading.Lock()
_last_tts_line = "SILENT"
_last_tts_line_time = 0.0


def speak(phrase: str):
    if not TTS_OK:
        return
    threading.Thread(target=_say, args=(phrase,), daemon=True).start()


def _say(phrase: str):
    try:
        with _tts_lock:
            _tts.say(phrase)
            _tts.runAndWait()
    except Exception:
        pass


def _side_from_angle(angle_deg: float) -> str:
    if angle_deg > PATH_CLEAR_DEG:
        return "RIGHT"
    if angle_deg < -PATH_CLEAR_DEG:
        return "LEFT"
    return "FRONT"


def _side_clearances(obstacles: List[FusedObstacle]) -> tuple[float, float]:
    left = min(
        (o.distance_m for o in obstacles if o.angle_deg < -PATH_CLEAR_DEG),
        default=SAFE_DIST_M + 1.0,
    )
    right = min(
        (o.distance_m for o in obstacles if o.angle_deg > PATH_CLEAR_DEG),
        default=SAFE_DIST_M + 1.0,
    )
    return left, right


def _best_turn_side(obstacles: List[FusedObstacle]) -> str:
    left_clear, right_clear = _side_clearances(obstacles)
    if left_clear >= right_clear + 0.3:
        return "LEFT"
    if right_clear >= left_clear + 0.3:
        return "RIGHT"
    return "STRAIGHT"


def _turn_intensity(angle_deg: float, distance_m: float, side: str) -> str:
    if side == "STRAIGHT":
        return "STRAIGHT"
    if abs(angle_deg) >= 35:
        return "SHARP"
    if abs(angle_deg) >= PATH_CLEAR_DEG:
        return "SLIGHTLY"
    return "SHARP" if distance_m < (DANGER_DIST_M + 0.5) else "SLIGHTLY"


def _go_phrase(
    obstacle: FusedObstacle,
    obstacles: List[FusedObstacle],
) -> str:
    side = _side_from_angle(obstacle.angle_deg)
    if side == "LEFT":
        return f"{_turn_intensity(obstacle.angle_deg, obstacle.distance_m, 'RIGHT')} RIGHT"
    if side == "RIGHT":
        return f"{_turn_intensity(obstacle.angle_deg, obstacle.distance_m, 'LEFT')} LEFT"
    best = _best_turn_side(obstacles)
    if best == "STRAIGHT":
        return "STRAIGHT"
    intensity = "SHARP" if obstacle.distance_m < 2.0 else "SLIGHTLY"
    return f"{intensity} {best}"


def _lean_phrase(
    obstacle: FusedObstacle,
    obstacles: List[FusedObstacle],
) -> str:
    side = _side_from_angle(obstacle.angle_deg)
    if side == "LEFT":
        return "Lean RIGHT"
    if side == "RIGHT":
        return "Lean LEFT"
    best = _best_turn_side(obstacles)
    if best == "LEFT":
        return "Lean LEFT"
    if best == "RIGHT":
        return "Lean RIGHT"
    return "GO STRAIGHT"


def _tts_message(
    obstacle: FusedObstacle,
    obstacles: List[FusedObstacle],
    danger_close: bool,
) -> str:
    name = obstacle.class_name or "obstacle"
    side = _side_from_angle(obstacle.angle_deg)

    if danger_close:
        go = _go_phrase(obstacle, obstacles)
        return f"STOP. Go {go}."

    if obstacle.distance_m < DANGER_DIST_M:
        return "STOP, danger close."

    if obstacle.distance_m < TTS_ALERT_DIST_M:
        go = _go_phrase(obstacle, obstacles)
        return f"ALERT! {name} on {side}. Go {go}."

    if obstacle.distance_m < SAFE_DIST_M:
        lean = _lean_phrase(obstacle, obstacles)
        return f"{name} on {side}. {lean}."

    return ""


# ── main loop ─────────────────────────────────────────────────────────────────

def main():
    global _tts_last_time, _last_tts_line, _last_tts_line_time
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

            # ── TTS alert system ────────────────────────────────────────────
            now = time.time()
            alert_range = [o for o in obstacles if o.distance_m < SAFE_DIST_M]
            danger_close = any(o.distance_m < ALERT_ENTER_M for o in alert_range)

            if danger_close and (now - _tts_last_time >= TTS_DANGER_COOLDOWN_S):
                closest = min(alert_range, key=lambda o: o.distance_m)
                phrase = _tts_message(closest, alert_range, danger_close=True)
                if phrase:
                    speak(phrase)
                    _tts_last_time = now
                    _last_tts_line = phrase
                    _last_tts_line_time = now
            elif alert_range and (now - _tts_last_time >= TTS_UPDATE_INTERVAL_S):
                closest = min(alert_range, key=lambda o: o.distance_m)
                phrase = _tts_message(closest, alert_range, danger_close=False)
                if phrase:
                    speak(phrase)
                    _tts_last_time = now
                    _last_tts_line = phrase
                    _last_tts_line_time = now

            # ── Throttled console output (1 Hz) ─────────────────────────────
            if t0 - last_print >= 1.0:
                prev_print = last_print
                last_print = t0
                print()
                print(time.strftime("%H:%M:%S"))
                if obstacles:
                    for i, obs in enumerate(obstacles, 1):
                        print(
                            f"obstacle {i} - {obs.class_name}, "
                            f"{obs.distance_m:.2f}m, "
                            f"{obs.angle_deg:+.0f}°"
                        )
                else:
                    print("obstacles: 0")

                tts_line = _last_tts_line if _last_tts_line_time >= prev_print else "SILENT"
                print(f"[TTS] {tts_line}")

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
