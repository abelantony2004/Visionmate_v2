"""
=============================================================
  BlindNav LIVE — Wearable 180° Obstacle Detection & Guidance
  Hardware : RPLiDAR A1/A2/A3  +  Raspberry Pi Camera
=============================================================

  Paper implemented:
    "The Obstacle Detection and Obstacle Avoidance Algorithm
     Based on 2-D Lidar" — ICIA 2015

  Paper algorithms used:
    • Median-of-5 stable distance         (Section III-A)
    • Gap-based laser-point cloud segment (Section III-B  eq.3–6)
    • Circular / Linear / Rectangle       (Section III-C  eq.7)
      cluster shape classification
    • Minimum cost function direction     (Section IV     eq.8–10)
    • Speed-zone based urgency            (Section IV     eq.11)
    • Destination-unreachable guard       (Section V      Fig.14–15)

  Reference code patterns used (lidar.py):
    • Per-cluster hysteresis state machine  (ENTER/EXIT distances)
    • Direction smoothing via rolling deque (DIRECTION_SMOOTH_LEN)
    • Intelligent trigger logic             (state / distance / reminder)
    • Median-of-5-closest stable distance
    • Threaded background scan loop
    • Threaded cv2.VideoCapture camera      (main.py / yolo_live.py)

  INSTALL (run once on Raspberry Pi):
  ------------------------------------
    sudo apt update
    sudo apt install -y python3-pip python3-opencv
    pip3 install rplidar-roboticia numpy opencv-python

  OPTIONAL YOLO object detection:
    Download into same folder:
      yolov3-tiny.weights   → https://pjreddie.com/media/files/yolov3-tiny.weights
      yolov3-tiny.cfg       → darknet/cfg/yolov3-tiny.cfg
      coco.names            → darknet/data/coco.names

  RUN:
    python3 blindnav_live.py                # with camera window
    python3 blindnav_live.py --no-display   # headless / SSH
    python3 blindnav_live.py --lidar-port /dev/ttyUSB1

=============================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import argparse
import time
import threading
import math
import bisect
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING, Union
from statistics import median

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import cv2
from yolo_live import CameraDetector

if TYPE_CHECKING:
    from fusion import FusedObstacle


# =============================================================================
#  CONFIGURATION — all tunable constants in one place
# =============================================================================

# ── LiDAR port ────────────────────────────────────────────────────────────────
LIDAR_PORT = '/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0'
LIDAR_BAUDRATE  = 115200
LIDAR_MOUNT_OFFSET_DEG = 0.0   # rotate if your 0° mark ≠ forward

# ── Front 180° scan geometry (paper: -5° to +185°, here ±90° of forward) ─────
FORWARD_ARC_DEG = 180.0        # total sweep kept
MIN_DIST_M      = 0.10         # ignore readings closer than this (sensor noise)
MAX_DIST_M      = 6.0          # ignore readings farther than this

# ── Angular zones within the 180° front arc ───────────────────────────────────
# Forward = 0°,  Right = +90°,  Left = -90°
ZONES: Dict[str, Tuple[float, float]] = {
    'FAR_LEFT'  : (-90.0, -50.0),
    'LEFT'      : (-50.0, -15.0),
    'FRONT'     : (-15.0,  15.0),
    'RIGHT'     : ( 15.0,  50.0),
    'FAR_RIGHT' : ( 50.0,  90.0),
}

# ── Distance thresholds — paper eq.11 (speed-zone logic) ─────────────────────
STOP_ENTER_M    = 1.0          # enter STOP when front < 1.0 m
STOP_EXIT_M     = 1.5          # exit  STOP when front > 1.5 m (hysteresis)
DANGER_DIST_M   = 1.5          # < 1.5 m → turn NOW
WARN_DIST_M     = 3.5          # < 3.5 m → slow / steer
SAFE_DIST_M     = 6.0          # ≥ 6.0 m → clear ahead

# ── Alert hysteresis ─────────────────────────────────────────────────────────
ALERT_ENTER_M  = 0.5          # trigger alerts only within 0.5 m
ALERT_EXIT_M   = 0.6          # small hysteresis gap to prevent flicker

# ── Trigger / alert timing (from reference lidar.py) ─────────────────────────
DIST_CHANGE_THR = 0.7          # m — retrigger if distance jumps this much
MIN_ALERT_INTERVAL = 3.0       # s — hard minimum between any two alerts
REMINDER_INTERVAL  = 10.0       # s — repeat if situation unchanged

# ── Direction smoothing (from reference lidar.py) ─────────────────────────────
DIRECTION_SMOOTH_LEN = 8       # rolling-average window (frames) per-side
DIRECTION_LOCK_FRAMES = 15     # hold a direction for at least this many frames
DIRECTION_CONFIRM = 3          # need N consecutive same-category to change

# ── Preprocessing (paper Section III-B, eq.3–6) ─────────────────────────────
ROBOT_WIDTH_M   = 0.5          # width of robot/person (paper: Width)
AMPLIFY_K       = 1.5          # amplification factor k (paper eq.5)
MIN_CLUSTER_PTS = 3            # discard noise clusters smaller than this

# ── Cluster shape classification thresholds (paper Section III-C) ─────────────
MAX_PTS_CIRCULAR   = 5         # ≤ 5 points → circular cluster
LINEAR_RATIO       = 0.2       # Dmax < 0.2·|S| → linear;  else → rectangle

# ── Minimum cost function (paper Section IV, eq.9–10) ─────────────────────────
SEARCH_RANGE_DEG   = 90        # search ±90° around desired heading
SEARCH_STEP_DEG    = 1         # 1° step (paper: "take 1° as the interval")
PATH_ANGLE_DEG     = 20.0      # ±20° counts as "straight ahead"

# ── Destination-unreachable guard (paper Section V, Fig.14–15) ───────────────
GOAL_RADIUS_M      = 1.0       # circular area around target — enable final check

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_INDEX   = 0                # 0 = Pi Camera / first device; 1 = USB webcam
CAM_W       = 640
CAM_H       = 480
CAM_FPS     = 10

# ── YOLO object detection ─────────────────────────────────────────────────────
USE_YOLO     = True
YOLO_WEIGHTS = 'yolov3-tiny.weights'
YOLO_CFG     = 'yolov3-tiny.cfg'
YOLO_NAMES   = 'coco.names'
YOLO_CONF    = 0.5


# =============================================================================
#  HELPER — angle normalisation
# =============================================================================

def normalise_angle(angle_deg: float) -> float:
    """
    Wrap RPLiDAR raw angle (0–360°) → (-180°, +180°] with mount offset.
    Convention: 0° = forward, +90° = right, -90° = left.
    """
    a = (angle_deg + LIDAR_MOUNT_OFFSET_DEG) % 360.0
    if a > 180.0:
        a -= 360.0
    return a


# =============================================================================
#  DATA STRUCTURES
# =============================================================================

@dataclass
class ClusterShape:
    """
    Result of paper Section III-C shape classification.
    shape : 'circular' | 'linear' | 'rectangle'
    For circular  → centre (cx, cy), radius r
    For linear    → endpoints p1, p2
    For rectangle → four endpoints p1..p4
    """
    shape:  str
    centre: Optional[Tuple[float, float]] = None
    radius: Optional[float]               = None
    points: List[Tuple[float, float]]     = field(default_factory=list)


@dataclass
class LidarCluster:
    """One detected obstacle cluster — mirrors reference lidar.py LidarCluster."""
    cluster_id:   int
    distance_m:   float          # median of 5 closest points (paper §III-A)
    angle_deg:    float          # smoothed angle of closest point
    point_count:  int
    shape:        ClusterShape   # paper §III-C classification
    raw_angles:   List[float] = field(default_factory=list)
    raw_ranges:   List[float] = field(default_factory=list)

    @property
    def direction_label(self) -> str:
        if self.angle_deg >  PATH_ANGLE_DEG:
            return "RIGHT"
        elif self.angle_deg < -PATH_ANGLE_DEG:
            return "LEFT"
        return "STRAIGHT"

    @property
    def in_path(self) -> bool:
        return abs(self.angle_deg) <= PATH_ANGLE_DEG

    @property
    def urgency(self) -> str:
        if self.distance_m < DANGER_DIST_M:
            return "critical"
        elif self.distance_m < WARN_DIST_M:
            return "warn"
        return "safe"

    @property
    def alert_message(self) -> str:
        d = f"{self.distance_m:.1f}m"
        if self.direction_label == "RIGHT":
            return f"Obstacle to the right, {d}."
        elif self.direction_label == "LEFT":
            return f"Obstacle to the left, {d}."
        return f"Obstacle ahead, {d}."


# =============================================================================
#  CLUSTER STATE MACHINE  (from reference lidar.py _ClusterState)
# =============================================================================

class _ClusterState:
    """
    Per-cluster stateful tracking:
      • Hysteresis    — ENTER/EXIT distance thresholds
      • Smoothing     — rolling deque for angle averaging
      • Trigger logic — fires alert only on meaningful change
    Directly ported from reference lidar.py with paper distances.
    """
    def __init__(self):
        self.active               = False
        self.direction_window     = deque(maxlen=DIRECTION_SMOOTH_LEN)
        self.last_direction_state: Optional[str]  = None
        self.last_distance:        Optional[float] = None
        self.last_alert_time:      float = 0.0

    def update(self, raw_angle: float, distance: float, now: float) -> bool:
        """
        Feed new measurement. Returns True when alert should fire.
        Implements reference lidar.py hysteresis + intelligent trigger.
        """
        # ── Hysteresis ────────────────────────────────────────
        if not self.active:
            in_danger = distance < ALERT_ENTER_M
        else:
            in_danger = distance < ALERT_EXIT_M

        if not in_danger:
            self.active               = False
            self.last_direction_state = None
            self.last_distance        = None
            return False

        # ── Direction smoothing ────────────────────────────────
        self.direction_window.append(raw_angle)
        smoothed = float(np.mean(self.direction_window))
        direction_state = (
            "RIGHT"    if smoothed >  PATH_ANGLE_DEG else
            "LEFT"     if smoothed < -PATH_ANGLE_DEG else
            "STRAIGHT"
        )

        # ── Intelligent trigger (reference lidar.py logic) ────
        trigger = (
            not self.active
            or self.last_direction_state != direction_state
            or (self.last_distance is not None
                and abs(distance - self.last_distance) > DIST_CHANGE_THR)
            or (now - self.last_alert_time) > REMINDER_INTERVAL
        )

        if trigger and (now - self.last_alert_time) > MIN_ALERT_INTERVAL:
            self.active               = True
            self.last_direction_state = direction_state
            self.last_distance        = distance
            self.last_alert_time      = now
            return True

        return False


# =============================================================================
#  PAPER ALGORITHM — Section III-A  3×3 Median filter
# =============================================================================

def median_filter_3x3(
    scan_history: List[List[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    """
    Paper Section III-A — 3×3 median filtering window.

    Build the window from 3 time-steps (t-1, t, t+1) × 3 spatial
    neighbours (i-1, i, i+1).  Take the median of those 9 values
    as R(t, i).  When fewer than 3 scans are available, fall back
    to spatial-only 1×3 median on the latest scan.

    Each scan is a list of (angle_deg, distance_m) sorted by angle.
    We align scans by nearest-angle matching so they share a common
    index space.
    """
    if not scan_history:
        return []

    current = scan_history[-1]  # time t
    if not current:
        return []

    n = len(current)
    angles = [p[0] for p in current]
    result = []

    # Build distance arrays for t-1, t, t+1 aligned to current angles
    dist_rows: List[List[float]] = []
    for scan in scan_history[-3:]:
        # For each angle in current, find nearest-angle match in this scan
        row = []
        scan_sorted = sorted(scan, key=lambda p: p[0]) if scan else []
        sa = [p[0] for p in scan_sorted]
        sd = [p[1] for p in scan_sorted]
        for a in angles:
            if not sa:
                row.append(0.0)
                continue
            # Binary search for nearest angle
            idx = bisect.bisect_left(sa, a)
            best = idx
            if idx >= len(sa):
                best = len(sa) - 1
            elif idx > 0:
                if abs(sa[idx-1] - a) < abs(sa[idx] - a):
                    best = idx - 1
            # Only match if within 2° (same angular bin)
            if abs(sa[best] - a) <= 2.0:
                row.append(sd[best])
            else:
                row.append(0.0)  # no match → 0 (will be filtered)
        dist_rows.append(row)

    # Pad to 3 rows if fewer scans available
    while len(dist_rows) < 3:
        dist_rows.insert(0, dist_rows[0][:])

    for i in range(n):
        window_vals = []
        for t_row in dist_rows:               # 3 time steps
            for di in (-1, 0, 1):             # 3 spatial neighbours
                j = max(0, min(n - 1, i + di))
                v = t_row[j]
                if v > 0:                     # ignore zero/missing
                    window_vals.append(v)
        if window_vals:
            window_vals.sort()
            med = window_vals[len(window_vals) // 2]
            result.append((angles[i], med))
        # else: point filtered out entirely

    return result


# =============================================================================
#  PAPER ALGORITHM — Section III-B  Preprocessing (segment + merge)
# =============================================================================

def segment_and_merge(
    scan: List[Tuple[float, float]],   # [(angle_deg, dist_m)]  sorted by angle
) -> List[List[Tuple[float, float]]]:
    """
    Paper Section III-B preprocessing pipeline:

    Step (i)  — segment: find contiguous blocks of non-zero readings
                         (paper eq.3, eq.4)
    Step (ii) — split:   Cartesian distance between adjacent points
                         > k × Width → break block  (paper eq.5)
    Step (iii)— merge:   gap between adjacent blocks < Width
                         → robot cannot pass → merge  (paper eq.6)
                         with linear distance interpolation.

    Returns list of point-groups, each group = one obstacle candidate.
    """
    if not scan:
        return []

    width = ROBOT_WIDTH_M
    k     = AMPLIFY_K

    # ── Step (i) — segment contiguous non-zero blocks (eq.3/eq.4) ─────
    # In the paper, R(i)==0 means no return.  Our scan is already filtered
    # to valid readings, so we segment by angular continuity: a gap > 3°
    # between adjacent points indicates a new block (adapts eq.3/eq.4 to
    # a variable-resolution RPLiDAR whose angles are not uniformly spaced).
    ANG_BREAK_DEG = 3.0
    blocks: List[List[Tuple[float, float]]] = [[scan[0]]]
    for i in range(1, len(scan)):
        if abs(scan[i][0] - scan[i-1][0]) > ANG_BREAK_DEG:
            blocks.append([scan[i]])
        else:
            blocks[-1].append(scan[i])

    # ── Step (ii) — split where Cartesian dist > k·Width (eq.5) ───────
    split_blocks: List[List[Tuple[float, float]]] = []
    for block in blocks:
        current = [block[0]]
        for i in range(1, len(block)):
            a0, d0 = block[i - 1]
            a1, d1 = block[i]
            x0 = d0 * math.cos(math.radians(a0))
            y0 = d0 * math.sin(math.radians(a0))
            x1 = d1 * math.cos(math.radians(a1))
            y1 = d1 * math.sin(math.radians(a1))
            cart_dist = math.hypot(x1 - x0, y1 - y0)
            if cart_dist > k * width:
                split_blocks.append(current)
                current = [block[i]]
            else:
                current.append(block[i])
        split_blocks.append(current)

    # ── Step (iii) — merge blocks whose gap < Width (eq.6) ────────────
    if len(split_blocks) < 2:
        return split_blocks

    merged: List[List[Tuple[float, float]]] = [split_blocks[0]]
    for i in range(1, len(split_blocks)):
        prev_end   = merged[-1][-1]      # point p
        curr_start = split_blocks[i][0]   # point q
        x0 = prev_end[1]   * math.cos(math.radians(prev_end[0]))
        y0 = prev_end[1]   * math.sin(math.radians(prev_end[0]))
        x1 = curr_start[1] * math.cos(math.radians(curr_start[0]))
        y1 = curr_start[1] * math.sin(math.radians(curr_start[0]))
        gap_L = math.hypot(x1 - x0, y1 - y0)

        if gap_L < width:
            # Cannot pass — merge with interpolation (eq.6)
            p_dist = prev_end[1]
            q_dist = curr_start[1]
            num    = max(1, int(gap_L / 0.05))
            interp = [
                (prev_end[0] + (curr_start[0] - prev_end[0]) * j / num,
                 p_dist + (q_dist - p_dist) * j / num)
                for j in range(1, num)
            ]
            merged[-1].extend(interp)
            merged[-1].extend(split_blocks[i])
        else:
            merged.append(split_blocks[i])

    return merged


# =============================================================================
#  PAPER ALGORITHM — Section III-C  Cluster shape classification
# =============================================================================

def classify_cluster(points: List[Tuple[float, float]]) -> ClusterShape:
    """
    Paper Section III-C — three clustering methods:

    Circular   (eq.7): x ≤ MAX_PTS_CIRCULAR points
                       → centre = midpoint of p and q, radius = max(Dj)
    Linear           : Dmax < LINEAR_RATIO · |S|
                       → return two endpoints
    Rectangle        : Dmax ≥ LINEAR_RATIO · |S|
                       → return bounding four corners (treated as 4 lines)
    """
    # Convert polar → Cartesian
    cart = [(d*math.cos(math.radians(a)), d*math.sin(math.radians(a)))
            for a, d in points]
    n = len(cart)

    # ── Circular (eq.7) ───────────────────────────────────────
    if n <= MAX_PTS_CIRCULAR:
        xo = (cart[0][0] + cart[-1][0]) / 2
        yo = (cart[0][1] + cart[-1][1]) / 2
        r  = max(math.hypot(cx-xo, cy-yo) for cx, cy in cart)
        return ClusterShape(
            shape='circular',
            centre=(xo, yo),
            radius=r,
            points=cart
        )

    # ── Line from first to last point ────────────────────────
    x0, y0 = cart[0]
    x1, y1 = cart[-1]
    s_dist  = math.hypot(x1-x0, y1-y0)  # |S| in paper

    if s_dist < 1e-6:
        return ClusterShape(shape='circular',
                            centre=(x0, y0), radius=0.01, points=cart)

    # Distance from each interior point to the line (paper §III-C)
    def point_to_line_dist(px, py):
        # Signed distance from (px,py) to line through (x0,y0)–(x1,y1)
        return abs((y1-y0)*px - (x1-x0)*py + x1*y0 - y1*x0) / s_dist

    dmax = max(point_to_line_dist(cx, cy) for cx, cy in cart[1:-1]) \
           if n > 2 else 0.0

    # ── Linear ────────────────────────────────────────────────
    if dmax < LINEAR_RATIO * s_dist:
        return ClusterShape(
            shape='linear',
            points=[cart[0], cart[-1]]
        )

    # ── Rectangle ─────────────────────────────────────────────
    xs = [c[0] for c in cart]
    ys = [c[1] for c in cart]
    corners = [
        (min(xs), min(ys)),
        (max(xs), min(ys)),
        (max(xs), max(ys)),
        (min(xs), max(ys)),
    ]
    return ClusterShape(shape='rectangle', points=corners)


# =============================================================================
#  LIDAR READER  (architecture from reference lidar.py)
# =============================================================================

class LidarReader:
    """
    Background-threaded RPLidar reader.
    Implements the full paper pipeline per scan cycle:
      1. Filter to forward 180° arc
      2. Median filter (paper §III-A)
      3. Segment + merge (paper §III-B)
      4. Classify cluster shape (paper §III-C)
      5. Per-cluster hysteresis + smoothing (reference lidar.py)

    Public API:
        lidar.start()
        clusters  = lidar.get_clusters()   → List[LidarCluster]
        raw_scan  = lidar.get_raw_scan()   → [(angle_deg, dist_m)]
        alerts    = lidar.get_pending_alerts()  → List[str]
        dist      = lidar.distance_at_angle(angle_deg)
        lidar.stop()
    """

    def __init__(self, port: str = LIDAR_PORT, baudrate: int = LIDAR_BAUDRATE):
        self.port     = port
        self.baudrate = baudrate

        self._lock              = threading.Lock()
        self._raw_scan:    List[Tuple[float,float]]  = []
        self._clusters:    List[LidarCluster]        = []
        self._alerts:      List[str]                 = []
        self._cluster_states: Dict[int, _ClusterState] = {}
        # Scan history for 3×3 temporal median filter (paper §III-A)
        self._scan_history: List[List[Tuple[float, float]]] = []

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        print(f"[LiDAR] Starting on {self.port} — warming up motor...")
        # Wait up to 5 s for first scan data to arrive
        for _ in range(50):
            time.sleep(0.1)
            with self._lock:
                got_data = len(self._raw_scan) > 0
            if got_data:
                break
        print(f"[LiDAR] Ready — scanning front {int(FORWARD_ARC_DEG)}°")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=4)

    def get_clusters(self) -> List[LidarCluster]:
        """Latest obstacle clusters, sorted nearest-first."""
        with self._lock:
            return list(self._clusters)

    def get_raw_scan(self) -> List[Tuple[float,float]]:
        """Latest filtered scan as [(angle_deg, dist_m)]."""
        with self._lock:
            return list(self._raw_scan)

    def get_pending_alerts(self) -> List[str]:
        """Drain and return alert messages that fired this cycle."""
        with self._lock:
            alerts = list(self._alerts)
            self._alerts.clear()
            return alerts

    def distance_at_angle(
        self, target_angle: float, tolerance_deg: float = 6.0
    ) -> Optional[float]:
        """Nearest distance reading within ±tolerance_deg of target_angle."""
        scan = self.get_raw_scan()
        hits = [d for a, d in scan if abs(a - target_angle) <= tolerance_deg]
        return min(hits) if hits else None

    # ── background thread ──────────────────────────────────────────────────────

    def _run(self):
        try:
            from rplidar import RPLidar
        except ImportError:
            raise ImportError(
                "pip install rplidar-roboticia --break-system-packages"
            )

        while self._running:
            lidar = None
            try:
                lidar = RPLidar(self.port, baudrate=self.baudrate)
                lidar.connect()
                print(f"[LiDAR] Connected to {self.port}")
                for scan in lidar.iter_scans(max_buf_meas=500):
                    if not self._running:
                        break
                    self._process_scan(scan)
            except Exception as e:
                print(f"[LiDAR] Error: {e} — retrying in 2 s")
                time.sleep(2)
            finally:
                if lidar:
                    try:
                        lidar.stop()
                        lidar.disconnect()
                    except Exception:
                        pass

    def _process_scan(self, raw_scan: list):
        now  = time.time()
        half = FORWARD_ARC_DEG / 2.0

        # ── Step 1: filter to forward 180° + valid range ──────
        points: List[Tuple[float,float]] = []
        for quality, angle, dist_mm in raw_scan:
            if quality == 0:
                continue
            dist_m = dist_mm / 1000.0
            if not (MIN_DIST_M <= dist_m <= MAX_DIST_M):
                continue
            norm = normalise_angle(angle)
            if abs(norm) <= half:
                points.append((norm, dist_m))

        if not points:
            return
        points.sort(key=lambda p: p[0])

        # ── Step 2: 3×3 Median filter (paper §III-A) ─────────
        self._scan_history.append(points)
        if len(self._scan_history) > 3:
            self._scan_history = self._scan_history[-3:]
        filtered = median_filter_3x3(self._scan_history)
        if not filtered:
            return

        # ── Step 3: Segment + merge (paper §III-B) ────────────
        raw_groups = segment_and_merge(filtered)

        # ── Step 4 & 5: Classify + state machine ──────────────
        new_clusters: List[LidarCluster] = []
        new_alerts:   List[str]          = []

        for cid, pts in enumerate(raw_groups):
            if len(pts) < MIN_CLUSTER_PTS:
                continue

            angles = [p[0] for p in pts]
            dists  = [p[1] for p in pts]

            # Stable distance: median of 5 closest (paper §III-A,
            #                                        reference lidar.py)
            sorted_dists = sorted(dists)
            stable_dist  = float(np.median(sorted_dists[:5]))

            # Angle of closest point
            closest_idx = int(np.argmin(dists))
            raw_angle   = angles[closest_idx]

            # Cluster shape classification (paper §III-C)
            shape = classify_cluster(pts)

            # Per-cluster state machine (reference lidar.py)
            if cid not in self._cluster_states:
                self._cluster_states[cid] = _ClusterState()
            state   = self._cluster_states[cid]
            trigger = state.update(raw_angle, stable_dist, now)

            smoothed_angle = (float(np.mean(state.direction_window))
                              if state.direction_window else raw_angle)

            cluster = LidarCluster(
                cluster_id  = cid,
                distance_m  = stable_dist,
                angle_deg   = smoothed_angle,
                point_count = len(pts),
                shape       = shape,
                raw_angles  = angles,
                raw_ranges  = dists,
            )
            new_clusters.append(cluster)
            if trigger:
                new_alerts.append(cluster.alert_message)

        # Prune stale cluster states
        active_ids = {c.cluster_id for c in new_clusters}
        for k in [k for k in self._cluster_states if k not in active_ids]:
            del self._cluster_states[k]

        new_clusters.sort(key=lambda c: c.distance_m)

        with self._lock:
            self._raw_scan = filtered
            self._clusters = new_clusters
            self._alerts.extend(new_alerts)


# =============================================================================
#  ZONE ANALYSER  —  maps clusters → 5 angular zones
# =============================================================================

class ZoneAnalyser:
    """
    Maps LidarCluster list into the 5 named angular zones.
    Each zone reports: min_dist (m) and status (CLEAR/SAFE/WARN/DANGER).
    """

    def analyse(self, clusters: List[Union[LidarCluster, "FusedObstacle"]]) -> Dict[str, dict]:
        zone_data = {
            name: {'min_dist': float('inf'), 'status': 'CLEAR',
                   'clusters': []}
            for name in ZONES
        }

        for cluster in clusters:
            for zone_name, (z_min, z_max) in ZONES.items():
                if z_min <= cluster.angle_deg <= z_max:
                    zone_data[zone_name]['clusters'].append(cluster)
                    if cluster.distance_m < zone_data[zone_name]['min_dist']:
                        zone_data[zone_name]['min_dist'] = cluster.distance_m

        for name, data in zone_data.items():
            d = data['min_dist']
            data['status'] = (
                'CLEAR'  if d == float('inf') else
                'DANGER' if d < DANGER_DIST_M else
                'WARN'   if d < WARN_DIST_M   else
                'SAFE'
            )

        return zone_data


# =============================================================================
#  GUIDANCE ENGINE  — paper Section IV (eq.8–11) + destination guard (Fig.14)
# =============================================================================

class GuidanceEngine:
    """
    Implements the paper's minimum cost function:

        α = atan2(yD - yr, xD - xr)          eq.8  desired heading
        β_i = α + i,  i ∈ [-90, 90]          eq.9  candidate directions
        β = min{ |β_i - α| }                  eq.10 cheapest safe direction

    Without a real destination we use α = 0° (straight ahead) as the
    desired heading, so the cost function simply picks the direction
    closest to forward that has no obstacle.

    Speed zone logic (eq.11):
        dist ≥ Lsafe   → full speed (SAFE)
        Ldanger ≤ dist < Lsafe → proportional slow (WARN)
        dist < Ldanger → minimum speed (DANGER / STOP)

    Destination-unreachable guard (Fig.14–15):
        When inside GOAL_RADIUS_M of the target, only avoid obstacles
        that are directly between robot and target; ignore those behind.

    Direction hysteresis:
        Once a turn direction (left/right) is chosen, it is maintained
        for subsequent scans unless the other side is significantly
        better (SWITCH_MARGIN_DEG° less deviation). This prevents
        flip-flopping between Walk LEFT / Walk RIGHT when both sides
        have similar clearance.
    """

    # Hysteresis: only switch sides if the new side is this many degrees
    # closer to forward than the current preferred side.
    SWITCH_MARGIN_DEG = 20

    # Number of consecutive "forward" results needed before we drop
    # the left/right preference.  Prevents flip-flop when front is
    # borderline between clear and blocked.
    FORWARD_CONFIRM_COUNT = 5

    def __init__(self):
        # Track last chosen turn direction for hysteresis
        self._last_turn_sign: Optional[int] = None   # -1=left, +1=right, None=no preference
        self._fwd_streak = 0                          # consecutive forward results

        # STOP state with hysteresis (enter at STOP_ENTER_M, exit at STOP_EXIT_M)
        self._in_stop = False

        # Lock-based direction stabiliser
        self._locked_category: Optional[str] = None   # 'FORWARD'/'LEFT'/'RIGHT'/'STOP'
        self._lock_counter = 0                         # frames remaining in lock
        # Per-side angle histories (never mix left/right angles)
        self._left_angles:  deque = deque(maxlen=DIRECTION_SMOOTH_LEN)
        self._right_angles: deque = deque(maxlen=DIRECTION_SMOOTH_LEN)
        # Consecutive-same-category counter for breaking lock
        self._pending_cat: Optional[str] = None
        self._pending_count = 0

    def decide(
        self,
        zone_data: Dict[str, dict],
        obstacles: List[Union["FusedObstacle", LidarCluster]],
    ) -> dict:

        dF  = zone_data['FRONT']['min_dist']
        dL  = zone_data['LEFT']['min_dist']
        dR  = zone_data['RIGHT']['min_dist']
        dFL = zone_data['FAR_LEFT']['min_dist']
        dFR = zone_data['FAR_RIGHT']['min_dist']

        # Object label from fused detections (skip generic "obstacle")
        obj = ''
        if obstacles:
            names: List[str] = []
            for o in obstacles:
                name = getattr(o, "class_name", "")
                if not name or name == "obstacle":
                    continue
                if name not in names:
                    names.append(name)
                if len(names) >= 2:
                    break
            if names:
                obj = f" {', '.join(names)} detected."

        # ── Minimum cost function (eq.9–10) ───────────────────

        # ── Smart side selection: always steer AWAY from obstacles ──
        # Compute combined clearance for each broad side.
        left_clearance  = min(dL, dFL)
        right_clearance = min(dR, dFR)

        # Determine which side has obstacles closest to the user.
        # If obstacles are primarily on one side, force the cost
        # function to search the OPPOSITE side first.
        if obstacles:
            nearest = min(obstacles, key=lambda o: o.distance_m)
            nearest_ang = nearest.angle_deg

            if nearest_ang < -PATH_ANGLE_DEG:
                # Nearest obstacle is on the LEFT → prefer RIGHT
                if right_clearance > left_clearance:
                    self._last_turn_sign = 1   # force right
            elif nearest_ang > PATH_ANGLE_DEG:
                # Nearest obstacle is on the RIGHT → prefer LEFT
                if left_clearance > right_clearance:
                    self._last_turn_sign = -1  # force left
            else:
                # Nearest obstacle is AHEAD → pick the side with more room
                if left_clearance > right_clearance + 0.3:
                    self._last_turn_sign = -1
                elif right_clearance > left_clearance + 0.3:
                    self._last_turn_sign = 1
                # else: keep existing preference (no strong bias)

        best_dir = self._cost_function_search(zone_data, obstacles)

        # ── Final sanity: never direct user into the nearest obstacle ──
        if best_dir is not None and obstacles:
            nearest = min(obstacles, key=lambda o: o.distance_m)
            nearest_ang = nearest.angle_deg
            # Check if suggested direction is toward the obstacle
            if nearest_ang < -PATH_ANGLE_DEG and best_dir < -PATH_ANGLE_DEG:
                # Suggesting LEFT but obstacle is LEFT → try RIGHT
                self._last_turn_sign = 1
                best_dir = self._cost_function_search(zone_data, obstacles)
            elif nearest_ang > PATH_ANGLE_DEG and best_dir > PATH_ANGLE_DEG:
                # Suggesting RIGHT but obstacle is RIGHT → try LEFT
                self._last_turn_sign = -1
                best_dir = self._cost_function_search(zone_data, obstacles)

        # ── STOP state with hysteresis ────────────────────────
        # Enter STOP when dF < STOP_ENTER_M, exit only when dF > STOP_EXIT_M.
        # This prevents flip-flop when dF hovers around the threshold.
        if self._in_stop:
            if dF > STOP_EXIT_M:
                self._in_stop = False
            # else: stay in STOP
        else:
            if dF < STOP_ENTER_M:
                self._in_stop = True

        # ── Speed zone classification (eq.11) ─────────────────
        nearest_any = min(dF, dL, dR, dFL, dFR)

        if nearest_any >= SAFE_DIST_M:
            urgency = 'SAFE'
            speed_pct = 1.0
        elif nearest_any >= DANGER_DIST_M:
            urgency = 'WARN'
            speed_pct = 0.2 + ((nearest_any - DANGER_DIST_M)
                               / (SAFE_DIST_M - DANGER_DIST_M)) * 0.8
        else:
            urgency = 'DANGER'
            speed_pct = 0.2

        # Shape info for richer message (LiDAR-only obstacles have shapes)
        shape_str = ''
        in_path = [o for o in obstacles if getattr(o, "in_path", False)]
        if in_path:
            shape = getattr(in_path[0], "shape", None)
            if shape is not None and getattr(shape, "shape", None):
                shape_str = f' ({shape.shape} obstacle)'

        # Nearest obstacle distance for messages
        nearest_dist = min(
            (o.distance_m for o in obstacles),
            default=float('inf'),
        )

        # ── Determine raw direction category ──────────────────
        if best_dir is None:
            raw_cat = 'STOP'
        elif abs(best_dir) <= PATH_ANGLE_DEG:
            raw_cat = 'FORWARD'
        elif best_dir < 0:
            raw_cat = 'LEFT'
        else:
            raw_cat = 'RIGHT'

        # ── Per-side angle smoothing ─────────────────────────
        # Track angles separately for LEFT and RIGHT so averaging
        # never mixes negative and positive angles (which would
        # produce ~0° = false FORWARD).
        if best_dir is not None:
            if best_dir < 0:
                self._left_angles.append(best_dir)
            elif best_dir > 0:
                self._right_angles.append(best_dir)

        # Pick smoothed angle from the correct side
        if raw_cat == 'LEFT' and self._left_angles:
            smooth_angle = sum(self._left_angles) / len(self._left_angles)
        elif raw_cat == 'RIGHT' and self._right_angles:
            smooth_angle = sum(self._right_angles) / len(self._right_angles)
        else:
            smooth_angle = best_dir

        # ── Lock-based stabilisation ──────────────────────────
        stable_cat = self._apply_lock(raw_cat)

        # Adjust angle if lock overrode the category
        if stable_cat != raw_cat and smooth_angle is not None:
            if stable_cat == 'LEFT':
                # Use left-side average, or flip sign
                if self._left_angles:
                    smooth_angle = sum(self._left_angles) / len(self._left_angles)
                else:
                    smooth_angle = -abs(smooth_angle)
            elif stable_cat == 'RIGHT':
                if self._right_angles:
                    smooth_angle = sum(self._right_angles) / len(self._right_angles)
                else:
                    smooth_angle = abs(smooth_angle)
            elif stable_cat == 'FORWARD':
                smooth_angle = 0.0

        sm_abs = abs(smooth_angle) if smooth_angle is not None else 0

        # ── Build output ──────────────────────────────────────
        if self._in_stop:
            # In STOP state — always prefix with Stop!
            speed_pct = 0.0
            if stable_cat == 'LEFT':
                alt = f" Turn left {sm_abs:.0f}°."
                direction = 'STOP_LEFT'
            elif stable_cat == 'RIGHT':
                alt = f" Turn right {sm_abs:.0f}°."
                direction = 'STOP_RIGHT'
            elif stable_cat == 'FORWARD':
                alt = ""
                direction = 'STOP'
            else:
                alt = " All directions blocked."
                direction = 'STOP'
            return dict(
                instruction = f"Stop!{obj} Obstacle {dF:.1f}m ahead.{alt}",
                severity    = 'DANGER',
                direction   = direction,
                speed_pct   = 0.0,
                best_angle  = smooth_angle,
            )

        if stable_cat == 'STOP':
            return dict(
                instruction = f"Stop!{obj}{shape_str} All directions blocked.",
                severity    = 'DANGER',
                direction   = 'STOP',
                speed_pct   = speed_pct,
                best_angle  = None,
            )

        if stable_cat == 'FORWARD':
            return dict(
                instruction = f"Path clear.{obj}",
                severity    = urgency,
                direction   = 'FORWARD',
                speed_pct   = speed_pct,
                best_angle  = smooth_angle,
            )

        if stable_cat == 'LEFT':
            label = 'TURN LEFT' if urgency == 'DANGER' else 'VEER LEFT'
            nd = (f"{nearest_dist:.1f}" if nearest_dist < float('inf')
                  else f">{SAFE_DIST_M:.0f}")
            return dict(
                instruction = (f"{shape_str} Obstacle {nd}m. "
                               f"Turn left {sm_abs:.0f}°.{obj}"),
                severity    = urgency,
                direction   = label,
                speed_pct   = speed_pct,
                best_angle  = smooth_angle,
            )

        # stable_cat == 'RIGHT'
        label = 'TURN RIGHT' if urgency == 'DANGER' else 'VEER RIGHT'
        nd = (f"{nearest_dist:.1f}" if nearest_dist < float('inf')
              else f">{SAFE_DIST_M:.0f}")
        return dict(
            instruction = (f"{shape_str} Obstacle {nd}m. "
                           f"Turn right {sm_abs:.0f}°.{obj}"),
            severity    = urgency,
            direction   = label,
            speed_pct   = speed_pct,
            best_angle  = smooth_angle,
        )

    def _cost_function_search(
        self,
        zone_data: Dict[str, dict],
        obstacles: List[Union["FusedObstacle", LidarCluster]],
    ) -> Optional[float]:
        """
        Paper Section IV, eq.9–10 — TIERED minimum-cost search.

        Instead of binary blocked/unblocked, tracks the minimum obstacle
        distance at each candidate angle.  Searches in three tiers:

          Tier 1 — Fully clear:  no obstacle within WARN_DIST_M  (3.5 m)
          Tier 2 — Passable:     no obstacle within DANGER_DIST_M (1.5 m)
          Tier 3 — Emergency:    direction with maximum clearance

        Only returns None (STOP) when maximum clearance < 0.3 m.

        This prevents false STOP when walls at ~1.8 m bracket the user
        but there is still > 1.5 m of walkable space to one side.
        """
        PERSON_HALF_WIDTH_M = 0.35  # safety padding on each side
        EMERGENCY_MIN_M     = 0.3   # below this in every direction → STOP

        # Track minimum obstacle distance at each angle (-90..+90 → idx 0..180)
        min_dist_at = [float('inf')] * 181

        # NOTE: We do NOT seed from zone-level distances here.
        # Zone-seeding was too aggressive — one cluster at the edge of the
        # FRONT zone would block all 30° of forward.  Instead we rely
        # solely on actual cluster angular extents (below) which give
        # precise per-degree blocking.

        for c in obstacles:
            if c.distance_m >= MAX_DIST_M:
                continue

            # If this is a LiDAR cluster, use its angular extent.
            raw_angles = getattr(c, "raw_angles", None)
            shape = getattr(c, "shape", None)
            if raw_angles:
                a_lo = a_hi = None

                if shape is not None and getattr(shape, "shape", None) == 'circular' \
                        and shape.centre and shape.radius is not None:
                    cx, cy = shape.centre
                    r = shape.radius + PERSON_HALF_WIDTH_M
                    dist_to_centre = math.hypot(cx, cy)
                    if dist_to_centre < 1e-6:
                        for idx in range(181):
                            min_dist_at[idx] = min(min_dist_at[idx], c.distance_m)
                        continue
                    if r >= dist_to_centre:
                        half_ang = 90.0
                    else:
                        half_ang = math.degrees(math.asin(r / dist_to_centre))
                    centre_ang = math.degrees(math.atan2(cy, cx))
                    a_lo = int(math.floor(centre_ang - half_ang))
                    a_hi = int(math.ceil(centre_ang + half_ang))

                elif shape is not None and getattr(shape, "shape", None) == 'linear' \
                        and len(shape.points) >= 2:
                    pts_ang = [math.degrees(math.atan2(py, px))
                               for px, py in shape.points]
                    margin = math.degrees(math.atan2(
                        PERSON_HALF_WIDTH_M, max(c.distance_m, 0.1)))
                    a_lo = int(math.floor(min(pts_ang) - margin))
                    a_hi = int(math.ceil(max(pts_ang) + margin))

                elif shape is not None and getattr(shape, "shape", None) == 'rectangle' \
                        and len(shape.points) >= 4:
                    pts_ang = [math.degrees(math.atan2(py, px))
                               for px, py in shape.points]
                    margin = math.degrees(math.atan2(
                        PERSON_HALF_WIDTH_M, max(c.distance_m, 0.1)))
                    a_lo = int(math.floor(min(pts_ang) - margin))
                    a_hi = int(math.ceil(max(pts_ang) + margin))

                else:
                    margin = math.degrees(math.atan2(
                        PERSON_HALF_WIDTH_M, max(c.distance_m, 0.1)))
                    a_lo = int(math.floor(min(raw_angles) - margin))
                    a_hi = int(math.ceil(max(raw_angles) + margin))

                if a_lo is not None and a_hi is not None:
                    for a in range(max(a_lo, -90), min(a_hi, 90) + 1):
                        idx = a + 90
                        if 0 <= idx < 181:
                            min_dist_at[idx] = min(min_dist_at[idx], c.distance_m)
                continue

            # Fused-only obstacle without LiDAR shape: treat as a point with margin.
            margin = math.degrees(math.atan2(
                PERSON_HALF_WIDTH_M, max(c.distance_m, 0.1)))
            a_lo = int(math.floor(c.angle_deg - margin))
            a_hi = int(math.ceil(c.angle_deg + margin))
            for a in range(max(a_lo, -90), min(a_hi, 90) + 1):
                idx = a + 90
                if 0 <= idx < 181:
                    min_dist_at[idx] = min(min_dist_at[idx], c.distance_m)

        # ── Helper: find best passable angle on a specific side ──────────
        def _best_on_side(sign: int, threshold: float) -> Optional[float]:
            """
            Search for the closest-to-forward passable angle on the given
            side (sign: -1=left, +1=right).  Returns angle or None.
            """
            for i in range(1, SEARCH_RANGE_DEG + 1):
                a = sign * i
                idx = a + 90
                if 0 <= idx < 181 and min_dist_at[idx] >= threshold:
                    return float(a)
            return None

        # ── Helper: apply hysteresis to a raw result ───────────────────
        def _apply_hysteresis(raw_angle: float, threshold: float) -> float:
            """
            If raw_angle is on the opposite side from _last_turn_sign,
            check whether the preferred side still has a passable path.
            If yes, return the preferred-side angle instead (sticky).
            Only allow switching when preferred side is fully blocked
            or the new side is significantly better (SWITCH_MARGIN_DEG).
            """
            if abs(raw_angle) <= PATH_ANGLE_DEG:
                return raw_angle  # forward — no side to stick to

            new_sign = -1 if raw_angle < 0 else 1

            if self._last_turn_sign is None or new_sign == self._last_turn_sign:
                return raw_angle  # same side or no preference — keep it

            # Raw result is on the OPPOSITE side from preference.
            # Try to find a passable angle on the preferred side.
            pref = _best_on_side(self._last_turn_sign, threshold)
            if pref is None:
                return raw_angle  # preferred side fully blocked — must switch

            # Preferred side has a path. Only switch if new side is
            # significantly closer to forward.
            if abs(raw_angle) < abs(pref) - self.SWITCH_MARGIN_DEG:
                return raw_angle  # new side is much better — switch

            return pref  # stick with preferred side

        # ── Tier search helper ─────────────────────────────────────────
        def _tier_search(threshold: float) -> Optional[float]:
            """Search outward from 0° for first angle with clearance >= threshold."""
            for i in range(0, SEARCH_RANGE_DEG + 1, SEARCH_STEP_DEG):
                if i == 0:
                    candidates = [0]
                elif self._last_turn_sign is not None and self._last_turn_sign > 0:
                    candidates = [i, -i]    # prefer right
                else:
                    candidates = [-i, i]    # prefer left (or default)
                for candidate in candidates:
                    idx = candidate + 90
                    if 0 <= idx < 181 and min_dist_at[idx] >= threshold:
                        return float(candidate)
            return None

        # ── Tier 1: Fully clear — no obstacle within WARN_DIST_M ──────
        raw = _tier_search(WARN_DIST_M)
        if raw is not None:
            result = _apply_hysteresis(raw, WARN_DIST_M)
            self._update_turn_sign(result)
            return result

        # ── Tier 2: Passable — no obstacle within DANGER_DIST_M ───────
        raw = _tier_search(DANGER_DIST_M)
        if raw is not None:
            result = _apply_hysteresis(raw, DANGER_DIST_M)
            self._update_turn_sign(result)
            return result

        # ── Tier 3: Emergency — direction with maximum clearance ───────
        best_idx  = max(range(181), key=lambda i: min_dist_at[i])
        best_dist = min_dist_at[best_idx]
        if best_dist >= EMERGENCY_MIN_M:
            raw = float(best_idx - 90)
            result = _apply_hysteresis(raw, EMERGENCY_MIN_M)
            self._update_turn_sign(result)
            return result

        self._last_turn_sign = None
        return None   # truly boxed in from all sides

    def _update_turn_sign(self, chosen_angle: float):
        """
        Update direction hysteresis state.
        Forward → only reset preference after FORWARD_CONFIRM_COUNT
        consecutive forward results (prevents flip-flop).
        Left/right → immediately lock to that side.
        """
        if abs(chosen_angle) <= PATH_ANGLE_DEG:
            self._fwd_streak += 1
            if self._fwd_streak >= self.FORWARD_CONFIRM_COUNT:
                self._last_turn_sign = None
        else:
            self._fwd_streak = 0
            self._last_turn_sign = -1 if chosen_angle < 0 else 1

    def _apply_lock(self, raw_cat: str) -> str:
        """
        Lock-based direction stabiliser.

        Once a direction category is emitted, hold it for at least
        DIRECTION_LOCK_FRAMES frames.  To change direction after a
        lock expires, the new category must appear DIRECTION_CONFIRM
        times in a row.  This prevents single-frame noise from
        flipping the output.

        Returns the stabilised category.
        """
        # ── First call: initialise ────────────────────────
        if self._locked_category is None:
            self._locked_category = raw_cat
            self._lock_counter = DIRECTION_LOCK_FRAMES
            return raw_cat

        # ── Active lock: count down and keep current direction ──
        if self._lock_counter > 0:
            self._lock_counter -= 1
            # Track if a new direction is building up
            if raw_cat != self._locked_category:
                if raw_cat == self._pending_cat:
                    self._pending_count += 1
                else:
                    self._pending_cat = raw_cat
                    self._pending_count = 1
            else:
                # Same as locked — reset any pending change
                self._pending_cat = None
                self._pending_count = 0
            return self._locked_category

        # ── Lock expired ─────────────────────────────────
        if raw_cat == self._locked_category:
            # Same direction — keep it, no need to re-lock
            self._pending_cat = None
            self._pending_count = 0
            return self._locked_category

        # Different direction requested — require confirmation
        if raw_cat == self._pending_cat:
            self._pending_count += 1
        else:
            self._pending_cat = raw_cat
            self._pending_count = 1

        if self._pending_count >= DIRECTION_CONFIRM:
            # Confirmed: switch to new direction and lock it
            self._locked_category = raw_cat
            self._lock_counter = DIRECTION_LOCK_FRAMES
            self._pending_cat = None
            self._pending_count = 0
            # Clear opposite-side angle history to avoid stale data
            if raw_cat == 'LEFT':
                self._right_angles.clear()
            elif raw_cat == 'RIGHT':
                self._left_angles.clear()
            elif raw_cat == 'FORWARD':
                self._left_angles.clear()
                self._right_angles.clear()
            return raw_cat

        # Not enough confirmations yet — keep old direction
        return self._locked_category


# =============================================================================
#  TERMINAL DISPLAY
# =============================================================================

def print_status(
    zone_data:  Dict[str, dict],
    clusters:   List[LidarCluster],
    guidance:   dict,
    objects:    List[str],
    count:      int,
    n_pts:      int,
):
    # Use ANSI cursor-home trick to overwrite in place instead of os.system('clear')
    # This prevents wiping [GUIDE] / [ALERT] lines printed above
    C = {
        'DANGER': '\033[91m', 'WARN':  '\033[93m',
        'SAFE':   '\033[92m', 'CLEAR': '\033[94m',
        'RESET':  '\033[0m',  'BOLD':  '\033[1m',
        'DIM':    '\033[2m',
    }

    def bar(status, dist_m, w=10):
        f = 0 if dist_m == float('inf') else \
            int((1 - min(dist_m, SAFE_DIST_M) / SAFE_DIST_M) * w)
        return C.get(status, '') + ('█' * f) + C['DIM'] + ('░' * (w - f)) + C['RESET']

    sc = C[guidance['severity']]

    lines = []
    lines.append(f"{C['BOLD']}{'='*58}{C['RESET']}")
    lines.append(f"{C['BOLD']}  BlindNav  |  scans:{count}  pts:{n_pts}  clusters:{len(clusters)}{C['RESET']}")
    lines.append(f"  Camera : {', '.join(objects) if objects else 'none'}")
    lines.append(f"{'─'*58}")
    lines.append(f"           FL        L      FWD      R       FR")

    # Zone colour bar
    bar_row = "  "
    for name in ['FAR_LEFT', 'LEFT', 'FRONT', 'RIGHT', 'FAR_RIGHT']:
        st = zone_data[name]['status']
        d  = zone_data[name]['min_dist']
        ds = f"{d:.1f}m" if d < float('inf') else " --- "
        bar_row += f"{C.get(st, '')}{ds:^9}{C['RESET']}"
    lines.append(bar_row)

    stat_row = "  "
    for name in ['FAR_LEFT', 'LEFT', 'FRONT', 'RIGHT', 'FAR_RIGHT']:
        st = zone_data[name]['status']
        stat_row += f"{C.get(st, '')}{st:^9}{C['RESET']}"
    lines.append(stat_row)

    lines.append(f"{'─'*58}")

    # Cluster table
    if clusters:
        lines.append(f"  Clusters: (shape  angle   dist  direction)")
        for c in clusters[:4]:
            sh = c.shape.shape[0].upper()
            lines.append(f"    #{c.cluster_id} {sh}  {c.angle_deg:+6.1f}°  "
                         f"{c.distance_m:.2f}m  [{c.direction_label}]  pts={c.point_count}")
    else:
        lines.append("  No clusters in range.")

    lines.append(f"{'─'*58}")
    lines.append(f"  {C['BOLD']}>> {sc}{guidance['direction']}{C['RESET']}")
    if guidance.get('best_angle') is not None:
        lines.append(f"     best angle: {guidance['best_angle']:+.0f}°")
    lines.append(f"  {sc}{guidance['instruction']}{C['RESET']}")
    lines.append(f"{'='*58}")

    print('\n'.join(lines), flush=True)


# =============================================================================
#  FUSION SMOOTHER  — stabilise YOLO+LiDAR distances
# =============================================================================

@dataclass
class _SmoothState:
    hist: deque
    ema_dist: Optional[float] = None
    ema_ang:  Optional[float] = None
    last_seen: float = 0.0


class _FusionSmoother:
    """
    Applies a short median window + EMA with jump clamp per obstacle id.
    Designed to reduce LiDAR distance jitter on fused detections.
    """
    def __init__(
        self,
        window: int = 5,
        alpha: float = 0.45,
        max_jump_m: float = 0.6,
        max_jump_deg: float = 8.0,
        stale_s: float = 1.5,
    ):
        self._window = window
        self._alpha = alpha
        self._max_jump_m = max_jump_m
        self._max_jump_deg = max_jump_deg
        self._stale_s = stale_s
        self._state: Dict[object, _SmoothState] = {}

    def smooth(self, obstacles: List["FusedObstacle"]) -> List["FusedObstacle"]:
        now = time.time()
        # prune stale
        for key in list(self._state.keys()):
            if now - self._state[key].last_seen > self._stale_s:
                del self._state[key]

        for o in obstacles:
            key = o.cluster_id if o.cluster_id is not None else (o.class_name, round(o.angle_deg))
            st = self._state.get(key)
            if st is None:
                st = _SmoothState(hist=deque(maxlen=self._window))
                self._state[key] = st

            st.hist.append(float(o.distance_m))
            med = float(median(st.hist))

            # distance smoothing with jump clamp
            prev = st.ema_dist if st.ema_dist is not None else med
            raw = med
            if abs(raw - prev) > self._max_jump_m:
                raw = prev + math.copysign(self._max_jump_m, raw - prev)
            st.ema_dist = prev + self._alpha * (raw - prev)
            o.distance_m = st.ema_dist

            # light angle smoothing for stability
            prev_ang = st.ema_ang if st.ema_ang is not None else o.angle_deg
            raw_ang = o.angle_deg
            if abs(raw_ang - prev_ang) > self._max_jump_deg:
                raw_ang = prev_ang + math.copysign(self._max_jump_deg, raw_ang - prev_ang)
            st.ema_ang = prev_ang + self._alpha * (raw_ang - prev_ang)
            o.angle_deg = st.ema_ang

            st.last_seen = now

        return obstacles


# =============================================================================
#  MAIN
# =============================================================================

def main():
    import sys
    from fusion import SensorFusion
    # Force line-buffered stdout so every print() appears immediately
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="BlindNav LIVE — RPLiDAR + Pi Camera obstacle guidance")
    parser.add_argument("--lidar-port",  default=LIDAR_PORT)
    parser.add_argument("--no-display",  action="store_true",
                        help="Headless — no cv2 window (good for SSH)")
    parser.add_argument("--loop-hz",     default=10, type=int)
    args = parser.parse_args()

    if not args.no_display:
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    print("\n🦯  BlindNav LIVE starting...\n", flush=True)

    lidar           = LidarReader(port=args.lidar_port)
    detector        = CameraDetector()
    sensor_fusion   = SensorFusion()
    fusion_smoother = _FusionSmoother()
    zone_anal     = ZoneAnalyser()
    guidance_e    = GuidanceEngine()
    interval      = 1.0 / args.loop_hz

    lidar.start()   # blocks until first scan data arrives
    detector.start()

    print("[BlindNav] System ready. Ctrl+C to stop.\n", flush=True)

    last_msg   = ''
    last_msg_t = 0.0
    last_sev   = ''
    last_tts   = ''
    last_tts_t = 0.0
    tts_clear_spoken = False
    # Alert intervals tuned for a blind user — avoid audio overload
    DANGER_INTERVAL = 3.0   # s — critical obstacle: alert every 3 s
    WARN_INTERVAL   = 6.0   # s — moderate obstacle: alert every 6 s
    CHANGE_GRACE    = 1.5   # s — new/changed situation: wait at least this

    try:
        while True:
            t0  = time.time()
            now = t0

            # 1 — clusters (paper pipeline runs in background thread)
            clusters = lidar.get_clusters()
            raw_scan = lidar.get_raw_scan()

            # 2 — map clusters to 5 zones
            zone_data = zone_anal.analyse(clusters)

            # 3 — camera object identification (only when front not clear)
            objects = []
            detections = detector.get_detections()
            if detections and zone_data['FRONT']['status'] != 'CLEAR':
                objects = list({d.get("class_name", "obstacle") for d in detections})

            # 4 — sensor fusion
            fused = sensor_fusion.fuse(detections, clusters)
            fused = fusion_smoother.smooth(fused)

            # 5 — guidance decision (paper eq.8–11)
            guidance = guidance_e.decide(zone_data, list(fused))

            # 6 — drain alerts (imminent threats only)
            alerts = lidar.get_pending_alerts()
            for msg in alerts:
                print(f"\033[91m[ALERT] {msg}\033[0m", flush=True)

            # 7 — report ALL obstacles up to MAX_DIST_M with direction
            reportable = [o for o in fused if o.distance_m <= MAX_DIST_M]
            reportable.sort(key=lambda o: o.distance_m)
            if reportable:
                tts_clear_spoken = False
                nearest   = reportable[0]   # sorted nearest-first
                dist      = nearest.distance_m
                direction = guidance['direction']
                best_ang  = guidance.get('best_angle')
                severity  = guidance.get('severity', 'SAFE')

                # ── Direction label from angle ────────────────
                if nearest.angle_deg < -50:
                    dir_label = "far left"
                elif nearest.angle_deg < -15:
                    dir_label = "left"
                elif nearest.angle_deg <= 15:
                    dir_label = "ahead"
                elif nearest.angle_deg <= 50:
                    dir_label = "right"
                else:
                    dir_label = "far right"

                # ── Avoidance instruction ─────────────────────
                if direction == 'STOP':
                    avoid = "STOP"
                elif direction == 'STOP_LEFT':
                    avoid = f"STOP → Turn LEFT" + (f" {abs(best_ang):.0f}°" if best_ang else "")
                elif direction == 'STOP_RIGHT':
                    avoid = f"STOP → Turn RIGHT" + (f" {abs(best_ang):.0f}°" if best_ang else "")
                elif 'LEFT' in direction:
                    avoid = f"Walk LEFT" + (f" {abs(best_ang):.0f}°" if best_ang else "")
                elif 'RIGHT' in direction:
                    avoid = f"Walk RIGHT" + (f" {abs(best_ang):.0f}°" if best_ang else "")
                else:
                    avoid = "FORWARD"

                # ── Severity tier ─────────────────────────────
                if dist < DANGER_DIST_M:
                    colour = "\033[91m"   # red
                    tag    = "\U0001f534 DANGER"
                elif dist < WARN_DIST_M:
                    colour = "\033[93m"   # yellow
                    tag    = "\U0001f7e1 CAUTION"
                else:
                    colour = "\033[96m"   # cyan
                    tag    = "\U0001f7e2 NOTICE"

                msg = f"{tag} Obstacle {dir_label} {dist:.2f}m \u2192 {avoid}"

                # ── List up to 3 other nearby clusters ────────
                extras = []
                for c in reportable[1:4]:
                    if c.angle_deg < -50:
                        d_lbl = "far left"
                    elif c.angle_deg < -15:
                        d_lbl = "left"
                    elif c.angle_deg <= 15:
                        d_lbl = "ahead"
                    elif c.angle_deg <= 50:
                        d_lbl = "right"
                    else:
                        d_lbl = "far right"
                    extras.append(f"{d_lbl} {c.distance_m:.1f}m")

                if extras:
                    msg += f"  | also: {', '.join(extras)}"

                # ── Pacing — different rates per severity ─────
                situation_changed = (msg != last_msg)
                if dist < DANGER_DIST_M:
                    pace = DANGER_INTERVAL
                elif dist < WARN_DIST_M:
                    pace = WARN_INTERVAL
                else:
                    pace = 8.0   # NOTICE — less frequent
                elapsed = now - last_msg_t

                should_speak = (
                    (situation_changed and elapsed >= CHANGE_GRACE)
                    or (not situation_changed and elapsed >= pace)
                )

                if should_speak:
                    top_fused = reportable[:3]
                    # Summarise fused obstacles
                    for obs in top_fused:
                        print(obs.alert_message, flush=True)
                    # Voice-ready phrases for external TTS
                    # Throttle TTS to match alert pacing
                    tts_phrase = "; ".join([o.tts_phrase for o in top_fused])
                    if (tts_phrase != last_tts and (now - last_tts_t) >= CHANGE_GRACE) or \
                       (tts_phrase == last_tts and (now - last_tts_t) >= pace):
                        print(f"[TTS] {tts_phrase}", flush=True)
                        last_tts = tts_phrase
                        last_tts_t = now

                    print(f"{colour}[!] {msg}\033[0m", flush=True)
                    last_msg   = msg
                    last_msg_t = now
                    last_sev   = severity
            else:
                if last_msg:
                    # Obstacle cleared — notify once
                    if now - last_msg_t >= CHANGE_GRACE:
                        print("\033[92m[\u2713] Path clear \u2014 no obstacles within 6m.\033[0m", flush=True)
                        last_msg   = ''
                        last_msg_t = now
                if not tts_clear_spoken and (now - last_tts_t) >= CHANGE_GRACE:
                    print("[TTS] path clear", flush=True)
                    last_tts = "path clear"
                    last_tts_t = now
                    tts_clear_spoken = True

            # 9 — camera window
            if not args.no_display:
                annotated = detector.get_annotated_frame()
                if annotated is not None:
                    if objects:
                        cv2.putText(annotated, ', '.join(objects), (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 2)
                    sev_bgr = {'DANGER': (0, 0, 255),
                               'WARN':   (0, 165, 255),
                               'SAFE':   (0, 255, 0)
                               }.get(guidance['severity'], (255, 255, 255))
                    cv2.putText(annotated, guidance['direction'],
                                (10, annotated.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, sev_bgr, 3)
                    cv2.putText(annotated, f"clusters:{len(clusters)}",
                                (10, annotated.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (200, 200, 200), 1)
                    cv2.imshow("BlindNav — Camera View", annotated)
                    if cv2.waitKey(1) == ord('q'):
                        break

            time.sleep(max(0.0, interval - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\n  Stopping...\n", flush=True)
    finally:
        lidar.stop()
        detector.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("Goodbye.\n", flush=True)


if __name__ == '__main__':
    main()
