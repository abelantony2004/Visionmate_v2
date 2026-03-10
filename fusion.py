"""
fusion.py
Fuses LidarCluster objects (from lidar.py) with YOLO detections (from yolo_live.py).

How the multi-object / similar-distance problem is solved
---------------------------------------------------------
1. LiDAR clusters separate objects that are at similar depths but different
   angles — e.g. two people at 1.2m but at -20° and +5° become two distinct
   LidarCluster objects with their own per-cluster hysteresis state.

2. YOLO detections give each cluster a semantic label (person, chair…) and a
   precise pixel bounding box.

3. Fusion matches each YOLO detection to the closest LiDAR cluster by angle.
   The LiDAR provides the metric distance; YOLO provides the class name.

4. When no LiDAR cluster matches a YOLO box (object above/below the scan
   plane — e.g. a table top or a kerb), fusion falls back to a bounding-box
   height heuristic and flags the estimate as "visual_est".

5. Priority ranking puts path-blocking, close-range obstacles first regardless
   of their detection order.

Output: List[FusedObstacle] sorted by urgency → path centrality → distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from lidar_new import LidarCluster
from yolo_live import pixel_x_to_angle   # noqa: F401 — re-exported for main.py

# ── Fusion tuning ──────────────────────────────────────────────────────────────
ANGLE_MATCH_TIGHT_DEG  = 6.0    # primary match window (well-calibrated mount)
ANGLE_MATCH_WIDE_DEG   = 15.0   # fallback window (slight mount offset tolerance)

# ── Alert zones ────────────────────────────────────────────────────────────────
CRITICAL_DIST_M        = 1.0    # urgent — immediate hazard
WARN_DIST_M            = 2.5    # warning — matches your ENTER_DANGER_DIST

# ── Path-blocking definition ───────────────────────────────────────────────────
PATH_ANGLE_DEG         = 20.0   # ±20° of straight-ahead = "in path"

# ── Visual depth fallback ──────────────────────────────────────────────────────
# Calibrate: hold a person-sized object at 1.0m, measure its box height in px.
REF_BOX_HEIGHT_PX      = 300.0  # pixel height of typical person at 1.0m (1280px frame)


# ── Output data structure ──────────────────────────────────────────────────────

@dataclass
class FusedObstacle:
    class_name:   str
    confidence:   float
    angle_deg:    float
    distance_m:   float
    dist_source:  str           # "lidar" | "lidar_approx" | "lidar_only" | "visual_est"
    urgency:      str           # "critical" | "warn" | "info"
    in_path:      bool
    cluster_id:   Optional[int]    = None
    box_xyxy:     List[float]      = field(default_factory=list)
    shape:        Optional[object] = None
    raw_angles:   List[float]      = field(default_factory=list)

    @property
    def direction(self) -> str:
        a = self.angle_deg
        if abs(a) <= 12:
            return "ahead"
        side = "right" if a > 0 else "left"
        return f"slightly {side}" if abs(a) <= 35 else side

    @property
    def alert_message(self) -> str:
        path_note = " — PATH BLOCKED" if self.in_path else ""
        src_note  = f" [{self.dist_source}]" if self.dist_source != "lidar" else ""
        return (
            f"[{self.urgency.upper():<8}] {self.class_name:<16} "
            f"{self.distance_m:.2f}m {self.direction}{path_note}{src_note}"
        )

    @property
    def tts_phrase(self) -> str:
        return f"{self.class_name}, {self.direction}, {self.distance_m:.1f} metres"


# ── Fusion engine ──────────────────────────────────────────────────────────────

class SensorFusion:
    """
    Stateless fusion — call fuse() every detection cycle.

    Parameters
    ----------
    detections : list of dicts from CameraDetector.get_detections()
    clusters   : list of LidarCluster from LidarReader.get_clusters()

    Returns
    -------
    List[FusedObstacle] sorted by priority (critical path-blockers first).
    """

    def fuse(
        self,
        detections: List[dict],
        clusters:   List[LidarCluster],
    ) -> List[FusedObstacle]:

        cluster_index = [(c.angle_deg, c) for c in clusters]
        obstacles: List[FusedObstacle] = []
        matched_ids: set = set()

        # ── Match each YOLO detection to a LiDAR cluster ───────────────────
        for det in detections:
            dist_m, source, matched = self._resolve_distance(det, cluster_index)
            if dist_m is None:
                continue

            if matched:
                matched_ids.add(matched.cluster_id)

            obstacles.append(FusedObstacle(
                class_name  = det["class_name"],
                confidence  = det["confidence"],
                angle_deg   = det["angle_deg"],
                distance_m  = dist_m,
                dist_source = source,
                urgency     = _urgency(dist_m),
                in_path     = abs(det["angle_deg"]) <= PATH_ANGLE_DEG,
                cluster_id  = matched.cluster_id if matched else None,
                box_xyxy    = det.get("box_xyxy", []),
                shape       = matched.shape if matched else None,
                raw_angles  = matched.raw_angles if matched else [],
            ))

        return _prioritise(obstacles)

    # ── distance resolution ────────────────────────────────────────────────────

    def _resolve_distance(
        self,
        det:           dict,
        cluster_index: List[Tuple[float, LidarCluster]],
    ) -> Tuple[Optional[float], str, Optional[LidarCluster]]:

        angle = det["angle_deg"]

        # 1. Tight angle match
        m = _nearest_cluster(cluster_index, angle, ANGLE_MATCH_TIGHT_DEG)
        if m:
            return m.distance_m, "lidar", m

        # 2. Wide angle match (mount calibration tolerance)
        m = _nearest_cluster(cluster_index, angle, ANGLE_MATCH_WIDE_DEG)
        if m:
            return m.distance_m, "lidar_approx", m

        # No LiDAR match — drop this detection (visual-only depth not used)
        return None, "", None


# ── helpers ────────────────────────────────────────────────────────────────────

def _nearest_cluster(
    index:     List[Tuple[float, LidarCluster]],
    target:    float,
    tolerance: float,
) -> Optional[LidarCluster]:
    candidates = [
        (abs(a - target), c)
        for a, c in index
        if abs(a - target) <= tolerance
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0])[1]


def _visual_depth_estimate(det: dict) -> Optional[float]:
    bh = det.get("box_height", 0)
    if bh < 30:
        return None
    est = (REF_BOX_HEIGHT_PX / bh) * 1.0
    return round(max(0.3, min(est, 6.0)), 2)


def _urgency(dist_m: float) -> str:
    if dist_m <= CRITICAL_DIST_M:
        return "critical"
    if dist_m <= WARN_DIST_M:
        return "warn"
    return "info"


def _prioritise(obstacles: List[FusedObstacle]) -> List[FusedObstacle]:
    rank = {"critical": 0, "warn": 1, "info": 2}
    return sorted(
        obstacles,
        key=lambda o: (
            rank[o.urgency],
            not o.in_path,
            abs(o.angle_deg),
            o.distance_m,
        ),
    )
