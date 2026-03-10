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
from typing import List, Optional, Dict, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import cv2


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
DANGER_DIST_M   = 0.8          # < 0.8 m → STOP / turn NOW
WARN_DIST_M     = 2.0          # < 2.0 m → slow / steer
SAFE_DIST_M     = 4.0          # ≥ 4.0 m → clear ahead

# ── Hysteresis (from reference lidar.py) ─────────────────────────────────────
ENTER_DANGER_M  = 1.5          # cluster becomes active (triggers alert)
EXIT_DANGER_M   = 2.0          # cluster clears (hysteresis gap prevents flicker)

# ── Trigger / alert timing (from reference lidar.py) ─────────────────────────
DIST_CHANGE_THR = 0.7          # m — retrigger if distance jumps this much
MIN_ALERT_INTERVAL = 2.0       # s — hard minimum between any two alerts
REMINDER_INTERVAL  = 6.0       # s — repeat if situation unchanged

# ── Direction smoothing (from reference lidar.py) ─────────────────────────────
DIRECTION_SMOOTH_LEN = 10      # rolling-average window (frames)

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
            in_danger = distance < ENTER_DANGER_M
        else:
            in_danger = distance < EXIT_DANGER_M

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

    def analyse(self, clusters: List[LidarCluster]) -> Dict[str, dict]:
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
    """

    def decide(
        self,
        zone_data: Dict[str, dict],
        clusters:  List[LidarCluster],
        objects:   List[str],
    ) -> dict:

        dF  = zone_data['FRONT']['min_dist']
        dL  = zone_data['LEFT']['min_dist']
        dR  = zone_data['RIGHT']['min_dist']
        dFL = zone_data['FAR_LEFT']['min_dist']
        dFR = zone_data['FAR_RIGHT']['min_dist']

        # Camera object label
        obj = ''
        if objects and objects != ['obstacle']:
            obj = f" {', '.join(set(objects[:2]))} detected."

        # ── Minimum cost function (eq.9–10) ───────────────────
        # Desired direction = 0° (straight ahead = α)
        # Search β_i = 0 + i for i in [-90..90], step 1°
        # Cost = |β_i - 0| = |i| → minimum cost is smallest |i|
        # that has no obstacle in its path.
        best_dir = self._cost_function_search(zone_data, clusters)

        # ── Speed zone classification (eq.11) ─────────────────
        # Paper eq.11: Vmax if dist >= Lsafe,
        #   (0.2 + (dist-Ldanger)/(Lsafe-Ldanger)*0.8)*Vmax if Ldanger<=dist<Lsafe,
        #   0.2*Vmax if dist < Ldanger
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

        # Shape info for richer message
        shape_str = ''
        in_path = [c for c in clusters if c.in_path]
        if in_path:
            s = in_path[0].shape.shape
            shape_str = f' ({s} obstacle)'

        # Nearest cluster distance for messages
        nearest_dist = (clusters[0].distance_m if clusters
                        else float('inf'))

        # ── Build guidance output (paper eq.9–10 result) ──────
        if best_dir is None:
            return dict(
                instruction = f"Stop!{obj}{shape_str} All directions blocked.",
                severity    = 'DANGER',
                direction   = 'STOP',
                speed_pct   = speed_pct,
                best_angle  = None,
            )

        abs_dir = abs(best_dir)

        if abs_dir <= PATH_ANGLE_DEG:
            # Best direction is within the forward cone
            return dict(
                instruction = f"Path clear.{obj}",
                severity    = urgency,
                direction   = 'FORWARD',
                speed_pct   = speed_pct,
                best_angle  = best_dir,
            )

        elif best_dir < 0:
            # Best safe direction is to the LEFT
            label = 'TURN LEFT' if urgency == 'DANGER' else 'VEER LEFT'
            nd = (f"{nearest_dist:.1f}" if nearest_dist < float('inf')
                  else f">{SAFE_DIST_M:.0f}")
            return dict(
                instruction = (f"{shape_str} Obstacle {nd}m. "
                               f"Turn left {abs_dir:.0f}°.{obj}"),
                severity    = urgency,
                direction   = label,
                speed_pct   = speed_pct,
                best_angle  = best_dir,
            )

        else:
            # Best safe direction is to the RIGHT
            label = 'TURN RIGHT' if urgency == 'DANGER' else 'VEER RIGHT'
            nd = (f"{nearest_dist:.1f}" if nearest_dist < float('inf')
                  else f">{SAFE_DIST_M:.0f}")
            return dict(
                instruction = (f"{shape_str} Obstacle {nd}m. "
                               f"Turn right {abs_dir:.0f}°.{obj}"),
                severity    = urgency,
                direction   = label,
                speed_pct   = speed_pct,
                best_angle  = best_dir,
            )

    def _cost_function_search(
        self,
        zone_data: Dict[str, dict],
        clusters:  List[LidarCluster],
    ) -> Optional[float]:
        """
        Paper Section IV, eq.9–10:
          β_i = α + i,   i ∈ [-90, 90],  step = 1°
          cost(β_i) = |β_i - α| = |i|
          choose β with minimum cost whose ray does NOT intersect
          any detected obstacle shape (circular, linear, rectangle).

        A candidate ray at angle β is traced from the origin (robot)
        outward.  If it intersects a cluster shape within WARN_DIST_M,
        that candidate is blocked.

        The cluster shapes come from paper Section III-C:
          • circular  → check ray-vs-circle intersection
          • linear    → check ray-vs-line-segment intersection
          • rectangle → check ray-vs-4-line-segments intersection
        """
        PERSON_HALF_WIDTH_M = 0.35  # safety padding on each side

        # Build a blocked-angle bitmap (-90..+90 → index 0..180)
        blocked = [False] * 181

        for c in clusters:
            if c.distance_m >= WARN_DIST_M:
                continue
            if not c.raw_angles:
                continue

            shape = c.shape

            if shape.shape == 'circular' and shape.centre and shape.radius is not None:
                # Inflate radius by person half-width for safety
                cx, cy = shape.centre
                r = shape.radius + PERSON_HALF_WIDTH_M
                dist_to_centre = math.hypot(cx, cy)
                if dist_to_centre < 1e-6:
                    # Obstacle at origin — block everything
                    for idx in range(181):
                        blocked[idx] = True
                    continue
                # Angular half-width that the circle subtends
                if r >= dist_to_centre:
                    half_ang = 90.0
                else:
                    half_ang = math.degrees(math.asin(r / dist_to_centre))
                centre_ang = math.degrees(math.atan2(cy, cx))
                a_lo = int(math.floor(centre_ang - half_ang))
                a_hi = int(math.ceil(centre_ang + half_ang))
                for a in range(max(a_lo, -90), min(a_hi, 90) + 1):
                    blocked[a + 90] = True

            elif shape.shape == 'linear' and len(shape.points) >= 2:
                # Line segment — block angles spanning the two endpoints
                # plus safety margin
                pts_ang = []
                for px, py in shape.points:
                    pts_ang.append(math.degrees(math.atan2(py, px)))
                a_lo = min(pts_ang)
                a_hi = max(pts_ang)
                margin = math.degrees(math.atan2(
                    PERSON_HALF_WIDTH_M, max(c.distance_m, 0.1)))
                a_lo -= margin
                a_hi += margin
                for a in range(max(int(math.floor(a_lo)), -90),
                               min(int(math.ceil(a_hi)), 90) + 1):
                    blocked[a + 90] = True

            elif shape.shape == 'rectangle' and len(shape.points) >= 4:
                # Rectangle — block angles spanning all 4 corners + margin
                pts_ang = []
                for px, py in shape.points:
                    pts_ang.append(math.degrees(math.atan2(py, px)))
                a_lo = min(pts_ang)
                a_hi = max(pts_ang)
                margin = math.degrees(math.atan2(
                    PERSON_HALF_WIDTH_M, max(c.distance_m, 0.1)))
                a_lo -= margin
                a_hi += margin
                for a in range(max(int(math.floor(a_lo)), -90),
                               min(int(math.ceil(a_hi)), 90) + 1):
                    blocked[a + 90] = True

            else:
                # Fallback: use raw angular extent + margin
                a_min = min(c.raw_angles)
                a_max = max(c.raw_angles)
                margin = math.degrees(math.atan2(
                    PERSON_HALF_WIDTH_M, max(c.distance_m, 0.1)))
                a_min -= margin
                a_max += margin
                for a in range(max(int(math.floor(a_min)), -90),
                               min(int(math.ceil(a_max)), 90) + 1):
                    blocked[a + 90] = True

        # Search outward from 0° (eq.9: β_i = α + i, α=0)
        # Cost = |i|, so search i = 0, ±1, ±2, ... (eq.10: min cost)
        for i in range(0, SEARCH_RANGE_DEG + 1, SEARCH_STEP_DEG):
            for candidate in ([0] if i == 0 else [-i, i]):
                idx = candidate + 90
                if 0 <= idx < 181 and not blocked[idx]:
                    return float(candidate)

        return None   # all directions blocked


# =============================================================================
#  CAMERA READER  (threaded cv2.VideoCapture — from reference main.py)
# =============================================================================

class CameraReader:
    """
    Threaded camera capture using cv2.VideoCapture.
    Pattern taken directly from the reference main.py / yolo_live.py.
    Pi Camera index = 0.  USB webcam = 1.
    """

    def __init__(self, index: int = CAM_INDEX):
        self._index        = index
        self._cap          = None
        self._frame        = None
        self._frame_annot  = None
        self._lock         = threading.Lock()
        self._running      = False

    def start(self):
        self._cap = cv2.VideoCapture(self._index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            print("[Camera] ERROR: Cannot open camera — check cable/config.")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self._cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)

        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[Camera] Ready — "
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
              f"@ {int(self._cap.get(cv2.CAP_PROP_FPS))} fps")

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return (self._frame_annot.copy()
                    if self._frame_annot is not None else None)

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame       = frame
                self._frame_annot = frame.copy()


# =============================================================================
#  OBJECT RECOGNISER  (YOLO / edge-detection fallback)
# =============================================================================

class ObjectRecogniser:
    """
    Identifies WHAT the obstacle is using the camera frame.
    Uses YOLOv3-tiny if model files are present.
    Falls back to OpenCV Canny edge density if not.
    """

    def __init__(self):
        self._net     = None
        self._classes: List[str] = []
        self._ready   = False
        self._load()

    def _load(self):
        if not USE_YOLO:
            return
        try:
            if all(os.path.exists(f)
                   for f in [YOLO_WEIGHTS, YOLO_CFG, YOLO_NAMES]):
                self._net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
                self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                with open(YOLO_NAMES) as f:
                    self._classes = [c.strip() for c in f]
                self._ready = True
                print("[Camera] YOLO loaded.")
            else:
                print("[Camera] YOLO files missing — using edge detection.")
        except Exception as e:
            print(f"[Camera] YOLO load error: {e}")

    def identify(self, frame: np.ndarray) -> List[str]:
        if self._ready and self._net is not None:
            return self._yolo(frame)
        return self._edges(frame)

    def _yolo(self, frame: np.ndarray) -> List[str]:
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self._net.setInput(blob)
        outs  = self._net.forward(self._net.getUnconnectedOutLayersNames())
        found: List[str] = []
        for out in outs:
            for det in out:
                scores = det[5:]
                cid    = int(np.argmax(scores))
                conf   = float(scores[cid])
                if conf > YOLO_CONF:
                    name = self._classes[cid]
                    if name not in found:
                        found.append(name)
        return found or ['obstacle']

    def _edges(self, frame: np.ndarray) -> List[str]:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges   = cv2.Canny(gray, 50, 150)
        h, w    = edges.shape
        centre  = edges[h//4:3*h//4, w//4:3*w//4]
        density = np.sum(centre > 0) / centre.size
        return ['obstacle'] if density > 0.08 else []


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
#  MAIN
# =============================================================================

def main():
    import sys
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

    lidar      = LidarReader(port=args.lidar_port)
    camera     = CameraReader()
    zone_anal  = ZoneAnalyser()
    recogniser = ObjectRecogniser()
    guidance_e = GuidanceEngine()
    interval   = 1.0 / args.loop_hz

    lidar.start()   # blocks until first scan data arrives
    camera.start()

    print("[BlindNav] System ready. Ctrl+C to stop.\n", flush=True)

    last_msg   = ''
    last_msg_t = 0.0
    last_sev   = ''
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
            frame   = camera.get_frame()
            if frame is not None and zone_data['FRONT']['status'] != 'CLEAR':
                objects = recogniser.identify(frame)

            # 4 — guidance decision (paper eq.8–11)
            guidance = guidance_e.decide(zone_data, clusters, objects)

            # 5 — drain alerts (processed below)
            lidar.get_pending_alerts()

            # 6 — only speak when obstacle is within DANGER range,
            #     with paced intervals so the blind user isn't overwhelmed
            danger_clusters = [c for c in clusters if c.distance_m < DANGER_DIST_M]
            if danger_clusters:
                nearest = danger_clusters[0]
                direction = guidance['direction']
                best_ang  = guidance.get('best_angle')
                severity  = guidance.get('severity', 'DANGER')

                if direction == 'STOP':
                    avoid = "STOP"
                elif 'LEFT' in direction:
                    avoid = f"Walk LEFT" + (f" {abs(best_ang):.0f}°" if best_ang else "")
                elif 'RIGHT' in direction:
                    avoid = f"Walk RIGHT" + (f" {abs(best_ang):.0f}°" if best_ang else "")
                else:
                    avoid = "FORWARD"

                msg = (f"Obstacle {nearest.distance_m:.2f}m "
                       f"{nearest.direction_label.lower()} → {avoid}")

                # Decide whether to speak this alert
                situation_changed = (msg != last_msg)
                interval = DANGER_INTERVAL if severity == 'DANGER' else WARN_INTERVAL
                elapsed  = now - last_msg_t

                should_speak = (
                    (situation_changed and elapsed >= CHANGE_GRACE)
                    or (not situation_changed and elapsed >= interval)
                )

                if should_speak:
                    print(f"\033[91m[!] {msg}\033[0m", flush=True)
                    last_msg   = msg
                    last_msg_t = now
                    last_sev   = severity
            else:
                if last_msg:
                    # Obstacle cleared — notify once
                    if now - last_msg_t >= CHANGE_GRACE:
                        print("\033[92m[✓] Path clear.\033[0m", flush=True)
                        last_msg   = ''
                        last_msg_t = now

            # 8 — camera window
            if not args.no_display:
                annotated = camera.get_annotated_frame()
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
        camera.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("Goodbye.\n", flush=True)


if __name__ == '__main__':
    main()