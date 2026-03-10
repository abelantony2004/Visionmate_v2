# YOLO + LiDAR Fusion Integration Plan

## Current State

| Component | What it does | Gap |
|---|---|---|
| `ObjectRecogniser` (lidar_new.py) | YOLOv3-tiny via `cv2.dnn`, returns only class name strings | No spatial info — can't say *where* or *how far* the object is |
| `CameraReader` (lidar_new.py) | Plain `cv2.VideoCapture` frame grab | No YOLO, no angle metadata |
| `GuidanceEngine` (lidar_new.py) | Uses raw `LidarCluster` only | Objects appended as text footnote, not fused |
| `yolo_live.py` | YOLOv8n NCNN, threaded, returns dicts with `angle_deg`, `box_xyxy` | Not connected to lidar_new.py |
| `fusion.py` | `SensorFusion.fuse()` maps YOLO angle → LiDAR cluster | Imports from old `lidar.py`, not `lidar_new.py` |

---

## Phase 1 — Fix fusion.py imports (5 min)

`fusion.py` currently imports:
```python
from lidar     import LidarCluster
from yolo_live import pixel_x_to_angle
```
Change to:
```python
from lidar_new import LidarCluster
from yolo_live import pixel_x_to_angle
```
`LidarCluster` in `lidar_new.py` has all the same fields (`cluster_id`, `distance_m`, `angle_deg`, `raw_angles`) plus the new `shape` field which `fusion.py` ignores — fully compatible.

---

## Phase 2 — Replace CameraReader + ObjectRecogniser with CameraDetector

Remove `CameraReader` and `ObjectRecogniser` from `lidar_new.py`. Import and use `CameraDetector` from `yolo_live.py`:

```python
from yolo_live import CameraDetector
```

`CameraDetector` already does everything both classes did, but better:
- Threaded (same pattern)
- YOLOv8n with `yolov8n.pt` (already in workspace)
- Returns per-detection `angle_deg` mapped from pixel X → real horizontal angle
- Returns annotated frames with drawn boxes

> ⚠️ **Calibration required:** `HFOV_DEG = 66.0` in `yolo_live.py` must match your actual camera.
> - Pi Camera Module 3 wide = 102°
> - Pi Camera Module 3 standard = 66°
> If this is wrong, the angle matching between YOLO detections and LiDAR clusters will be offset.

---

## Phase 3 — Wire SensorFusion into the main loop

In `lidar_new.py` main loop, replace the current 3-step camera block:

**Current (loose):**
```python
frame = camera.get_frame()
if frame is not None and zone_data['FRONT']['status'] != 'CLEAR':
    objects = recogniser.identify(frame)
guidance = guidance_e.decide(zone_data, clusters, objects)
```

**New (fused):**
```python
from fusion import SensorFusion
...
sensor_fusion = SensorFusion()
...
# in loop:
detections = detector.get_detections()                   # from CameraDetector
fused      = sensor_fusion.fuse(detections, clusters)    # List[FusedObstacle]
guidance   = guidance_e.decide(zone_data, fused)
```

---

## Phase 4 — Update GuidanceEngine to consume FusedObstacle

`GuidanceEngine.decide()` currently takes `List[LidarCluster]` for the obstacle argument. Adapt the signature:

```python
def decide(self, zone_data, obstacles: List[FusedObstacle]) -> dict:
```

Use `obstacle.class_name` for the alert message instead of the generic "obstacle":
```
# Before: "Obstacle ahead, 1.2m."
# After:  "Person ahead, 1.2m → Turn left 20°."
```

`FusedObstacle.tts_phrase` (already in `fusion.py`) generates the voice-ready string:
```python
obs.tts_phrase  # → "person, ahead, 1.2 metres"
```

---

## Phase 5 — Update ZoneAnalyser

Change `ZoneAnalyser.analyse()` to accept either `LidarCluster` or `FusedObstacle`.
Both have `angle_deg` and `distance_m` — duck-typing works, only the type hint needs updating.

---

## Phase 6 — Enhanced alert messages in main loop

Replace the current plain-text alert block with `FusedObstacle.alert_message` and `FusedObstacle.tts_phrase`:

```python
for obs in fused[:3]:   # top 3 by priority
    print(obs.alert_message)
    # → "[CRITICAL ] person            1.20m ahead — PATH BLOCKED"
```

For TTS (espeak / pyttsx3):
```python
speak(obs.tts_phrase)
# → "person, ahead, 1.2 metres"
```

---

## Phase 7 — Display window

Replace `camera.get_annotated_frame()` with `detector.get_annotated_frame()` — already draws YOLO boxes + FPS overlay.

Overlay LiDAR distance on each YOLO box:
```python
for obs in fused:
    x1, y1 = int(obs.box_xyxy[0]), int(obs.box_xyxy[1])
    cv2.putText(frame, f"{obs.distance_m:.1f}m", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
```

---

## Dependency Changes

| Package | Current | After |
|---|---|---|
| `rplidar-roboticia` | ✓ | ✓ |
| `opencv-python` | ✓ (`cv2.dnn` YOLO) | ✓ (display only) |
| `ultralytics` | partial (`yolo_live.py` exists) | ✓ must be installed |
| `numpy` | ✓ | ✓ |

```bash
pip install ultralytics
```

---

## Execution Order (file changes summary)

| # | File | Change |
|---|---|---|
| 1 | `fusion.py` | Fix 2 import lines (`lidar` → `lidar_new`) |
| 2 | `lidar_new.py` | Remove `CameraReader` and `ObjectRecogniser` classes |
| 3 | `lidar_new.py` | Add imports: `CameraDetector`, `SensorFusion`, `FusedObstacle` |
| 4 | `lidar_new.py` | Update `GuidanceEngine.decide()` signature + message builder |
| 5 | `lidar_new.py` | Update `main()` loop to use fused pipeline |
| 6 | `yolo_live.py` | Verify/calibrate `HFOV_DEG` for your exact camera module |
