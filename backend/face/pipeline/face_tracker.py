"""
Face tracking via mediapipe.

Per-frame outputs:
    - timestamp (s)
    - face_present (bool)
    - scoreable (bool)              face_present AND |yaw|<25° AND |pitch|<20°
    - head pose: pitch, yaw, roll (degrees, img2pose-free solvePnP)
    - left_ear, right_ear, avg_ear  raw EAR (Soukupová & Cech); blink threshold
                                    is applied LATER, per-person, in baseline.py
    - blink (bool)                  fixed-threshold flag retained for backward
                                    compatibility only — downstream code now
                                    re-derives events from `avg_ear` against
                                    the adaptive threshold from baseline.
    - bilateral features (used by asymmetry):
        brow_asym, cheek_asym, mouth_corner_asym, eye_open_asym
        — all unitless ratios |L − R| / (|L| + |R| + ε) in [0,1].
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh

# Six-point eye landmarks (P1–P6) for the Soukupová & Cech EAR formula.
LEFT_EYE_6 = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_6 = [263, 387, 385, 362, 380, 373]

# Mouth corners + lip midline.
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# Inner brow points (closest to corrugator) — used for left/right brow height.
LEFT_BROW_INNER = 105       # left medial brow
RIGHT_BROW_INNER = 334      # right medial brow
LEFT_EYE_TOP = 159          # upper-lid midpoint, left
RIGHT_EYE_TOP = 386         # upper-lid midpoint, right

# Upper-cheek (zygomatic) landmarks — proxy for AU6 cheek raise.
LEFT_CHEEK = 117
RIGHT_CHEEK = 346

# Nose tip for midline reference.
NOSE_TIP = 1

# Curated 26-point landmark subset for the simulation overlay. Coordinates are
# stored normalised (0–1) so the frontend can scale them to whatever size the
# <video> element happens to be rendered at. The order is fixed and the groups
# below index INTO this array (not into the raw FaceMesh).
VIZ_LANDMARK_INDICES: List[int] = [
    # 0–5 : left eye, Soukupová & Cech EAR ring
    33, 160, 158, 133, 153, 144,
    # 6–11 : right eye, same ring
    263, 387, 385, 362, 380, 373,
    # 12–14 : left brow (medial → lateral)
    107, 105, 66,
    # 15–17 : right brow (medial → lateral)
    336, 334, 296,
    # 18–21 : mouth corners + lip top + lip bottom
    61, 291, 13, 14,
    # 22 : nose tip
    1,
    # 23 : chin
    152,
    # 24–25 : upper cheeks (left, right)
    117, 346,
]

# Used by the frontend canvas to draw outline polylines per group.
VIZ_LANDMARK_GROUPS: dict = {
    "left_eye":   [0, 1, 2, 3, 4, 5],
    "right_eye":  [6, 7, 8, 9, 10, 11],
    "left_brow":  [12, 13, 14],
    "right_brow": [15, 16, 17],
    "mouth":      [18, 20, 19, 21],   # L-top-R-bottom for a diamond outline
    "nose":       [22],
    "chin":       [23],
    "cheeks":     [24, 25],
}

# 3-D model points (canonical face, mm) used for solvePnP head-pose estimation.
# IMPORTANT: y-coordinates use the OpenCV image convention (y points DOWN).
# Earlier versions used y-up which caused solvePnP to return pitch ≈ ±180°
# even when the face was looking straight at the camera — leading to a
# `scoreable=False` mask on every frame.
MODEL_POINTS_3D = np.array([
    [0.0,    0.0,   0.0],       # nose tip (1)
    [0.0,    63.6, -12.5],      # chin (152)   — below nose, +y is "down"
    [-43.3, -32.7, -26.0],      # left eye outer corner (33)   — above nose
    [43.3,  -32.7, -26.0],      # right eye outer corner (263) — above nose
    [-28.9,  28.9, -24.1],      # left mouth corner (61)       — below nose
    [28.9,   28.9, -24.1],      # right mouth corner (291)     — below nose
], dtype=np.float64)
POSE_IDX = [1, 152, 33, 263, 61, 291]

# Quality gates for "scoreable" frame. OpenFace uses 30°/30°; we run a touch
# stricter at 35°/30° but only enforce when the pose estimate looks reliable.
MAX_YAW_DEG = 35.0
MAX_PITCH_DEG = 30.0
# If any angle exceeds this, treat the pose as unreliable and fall back to
# `face_present` alone (otherwise a buggy solvePnP could silently null every
# frame's scoreable flag, exactly the bug we just hit).
POSE_PLAUSIBLE_DEG = 80.0

# Static EAR threshold used to fill the legacy `blink` column. Real blink
# events are re-derived against an adaptive per-person threshold by
# `baseline.derive_blink_events()`.
LEGACY_EAR_THRESHOLD = 0.21


def _ear(landmarks, idx6, w: int, h: int) -> float:
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idx6])
    vertical = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    return float(vertical / (2.0 * horizontal + 1e-6))


def _wrap_180(angle: float) -> float:
    """Wrap an angle to [-180, 180]."""
    a = (angle + 180.0) % 360.0 - 180.0
    return a


def _fold_to_half(angle: float) -> float:
    """
    Fold an angle into [-90, 90] by reflecting through ±180.

    solvePnP for faces can resolve an equivalent rotation as either
    `pitch ≈ 0` or `pitch ≈ 180` depending on which side of the gimbal
    branch the solver lands on; both are the same physical pose. We
    canonicalise to the small-angle side so the scoreable gate works.
    """
    a = _wrap_180(angle)
    if a > 90.0:
        a = 180.0 - a
    elif a < -90.0:
        a = -180.0 - a
    return a


def _head_pose(landmarks, w: int, h: int) -> Tuple[float, float, float]:
    img_pts = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in POSE_IDX],
        dtype=np.float64,
    )
    focal = w
    cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS_3D, img_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        pitch = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
        yaw = float(np.degrees(np.arctan2(-R[2, 0], sy)))
        roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    else:
        pitch = float(np.degrees(np.arctan2(-R[1, 2], R[1, 1])))
        yaw = float(np.degrees(np.arctan2(-R[2, 0], sy)))
        roll = 0.0
    # Canonicalise to the small-angle branch and the standard [-180,180] range.
    return _fold_to_half(pitch), _fold_to_half(yaw), _wrap_180(roll)


def _unroll(points: np.ndarray, roll_deg: float, centre: np.ndarray) -> np.ndarray:
    """Rotate `points` by −roll around `centre` so that head-tilt is removed."""
    theta = -np.radians(roll_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return (points - centre) @ rot.T + centre


def _bilateral_features(landmarks, w: int, h: int, roll_deg: float) -> dict:
    """
    Compute four roll-corrected left/right asymmetry probes from FaceMesh.

    Returned values are |L − R| / (|L| + |R| + ε) ratios in [0, 1].
    """
    def px(i: int) -> np.ndarray:
        return np.array([landmarks[i].x * w, landmarks[i].y * h], dtype=float)

    nose = px(NOSE_TIP)

    # Roll-correct every landmark we use, so head tilt doesn't create false asym.
    pts = {
        "lbrow":  px(LEFT_BROW_INNER),
        "rbrow":  px(RIGHT_BROW_INNER),
        "leye_t": px(LEFT_EYE_TOP),
        "reye_t": px(RIGHT_EYE_TOP),
        "lcheek": px(LEFT_CHEEK),
        "rcheek": px(RIGHT_CHEEK),
        "lmouth": px(MOUTH_LEFT),
        "rmouth": px(MOUTH_RIGHT),
    }
    arr = np.stack(list(pts.values()))
    arr = _unroll(arr, roll_deg, nose)
    (lbrow, rbrow, leye_t, reye_t,
     lcheek, rcheek, lmouth, rmouth) = [arr[i] for i in range(8)]

    # Reference scale: outer-eye-to-outer-eye distance (proxy for face size).
    ref = float(np.linalg.norm(px(33) - px(263))) + 1e-6

    # ---- brow height: vertical gap brow → eyelid (small = brow furrow / AU4) ----
    lbrow_h = (leye_t[1] - lbrow[1])
    rbrow_h = (reye_t[1] - rbrow[1])
    brow_asym = abs(lbrow_h - rbrow_h) / (abs(lbrow_h) + abs(rbrow_h) + 1e-6)

    # ---- cheek raise: vertical position of upper cheek relative to nose ----
    lcheek_d = nose[1] - lcheek[1]
    rcheek_d = nose[1] - rcheek[1]
    cheek_asym = abs(lcheek_d - rcheek_d) / (abs(lcheek_d) + abs(rcheek_d) + 1e-6)

    # ---- mouth corner height: y-position relative to nose (smirk signal) ----
    lmouth_d = nose[1] - lmouth[1]
    rmouth_d = nose[1] - rmouth[1]
    mouth_corner_asym = abs(lmouth_d - rmouth_d) / (abs(lmouth_d) + abs(rmouth_d) + 1e-6)

    # ---- eye opening: vertical aperture difference (squint asymmetry) ----
    l_open = abs(leye_t[1] - lbrow[1])     # rough proxy
    r_open = abs(reye_t[1] - rbrow[1])
    eye_open_asym = abs(l_open - r_open) / (abs(l_open) + abs(r_open) + 1e-6)

    # Mouth-midline shift (in face-relative units) — kept for UI display.
    midline_x = (lmouth[0] + rmouth[0]) / 2.0
    nose_dx = abs(midline_x - nose[0]) / ref

    return {
        "brow_asym":         float(np.clip(brow_asym, 0.0, 1.0)),
        "cheek_asym":        float(np.clip(cheek_asym, 0.0, 1.0)),
        "mouth_corner_asym": float(np.clip(mouth_corner_asym, 0.0, 1.0)),
        "eye_open_asym":     float(np.clip(eye_open_asym, 0.0, 1.0)),
        "mouth_asym":        float(np.clip(nose_dx, 0.0, 1.0)),  # legacy column
    }


def track_face(frames: List[np.ndarray], fps: int = 12) -> pd.DataFrame:
    """Track face landmarks → DataFrame, one row per frame."""
    records = []

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as mesh:
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            res = mesh.process(frame)
            rec = {
                "frame_idx": i,
                "timestamp": i / fps,
                "face_present": False,
                "scoreable": False,
                "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
                "left_ear": 0.30, "right_ear": 0.30, "avg_ear": 0.30,
                "blink": False,
                "brow_asym": 0.0,
                "cheek_asym": 0.0,
                "mouth_corner_asym": 0.0,
                "eye_open_asym": 0.0,
                "mouth_asym": 0.0,
                "landmarks": [],            # filled when face_present
            }
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                rec["face_present"] = True

                le = _ear(lm, LEFT_EYE_6, w, h)
                re = _ear(lm, RIGHT_EYE_6, w, h)
                avg = (le + re) / 2.0
                rec["left_ear"] = le
                rec["right_ear"] = re
                rec["avg_ear"] = avg
                rec["blink"] = avg < LEGACY_EAR_THRESHOLD     # legacy / fallback

                pose_ok = False
                try:
                    rec["pitch"], rec["yaw"], rec["roll"] = _head_pose(lm, w, h)
                    pose_ok = True
                except cv2.error:
                    pass

                # If pose estimate looks implausible (solver landed on a bad
                # gimbal branch, or numerical degeneracy), trust face_present
                # alone rather than silently rejecting every frame.
                pose_reliable = (
                    pose_ok
                    and abs(rec["pitch"]) <= POSE_PLAUSIBLE_DEG
                    and abs(rec["yaw"]) <= POSE_PLAUSIBLE_DEG
                    and abs(rec["roll"]) <= POSE_PLAUSIBLE_DEG
                )
                if pose_reliable:
                    rec["scoreable"] = (
                        abs(rec["yaw"]) <= MAX_YAW_DEG
                        and abs(rec["pitch"]) <= MAX_PITCH_DEG
                    )
                else:
                    rec["scoreable"] = True   # pose unreliable → don't gate on it

                bilat = _bilateral_features(lm, w, h, rec["roll"])
                rec.update(bilat)

                # Normalized landmarks for the simulation overlay.
                rec["landmarks"] = [
                    [round(float(lm[i].x), 4), round(float(lm[i].y), 4)]
                    for i in VIZ_LANDMARK_INDICES
                ]

            records.append(rec)

    df = pd.DataFrame(records)
    n_face = int(df["face_present"].sum())
    n_score = int(df["scoreable"].sum())
    if n_face:
        present = df[df["face_present"]]
        log.info(
            "Face tracking: face_present=%d/%d, scoreable=%d/%d, "
            "mean pitch=%.1f° yaw=%.1f° roll=%.1f° (only present frames)",
            n_face, len(df), n_score, len(df),
            float(present["pitch"].mean()),
            float(present["yaw"].mean()),
            float(present["roll"].mean()),
        )
    else:
        log.info(
            "Face tracking: face_present=0/%d, scoreable=0/%d", len(df), len(df),
        )
    return df
