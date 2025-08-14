#!/usr/bin/env python3
"""
apriltag_tracker_plus.py
------------------------
AprilTag tracker for a *fixed camera* with extra features:

New vs baseline:
- --homography homography.yaml : Map pixel (x,y) -> planar world (X_m,Y_m) via 3x3 H (from one_shot_plane_calib.py).
- --time {fps,pos_msec}       : Choose timestamp source (frame_idx/fps or CAP_PROP_POS_MSEC).
- Better pose for planar tag: try SOLVEPNP_IPPE_SQUARE when available; fallback to ITERATIVE.
- Preview path drawing fixed for pixel-mode.
- Reprojection error per frame (metric pose) stored as reproj_px.
- Smarter target selection when --id not provided: choose largest-area detection, then stickiness by proximity.
- Optional --write-parquet to write a Parquet copy for faster downstream loading.

Modes:
  * METRIC mode (full): needs --calib + --tag-size  → (x,y,z) meters via solvePnP.
  * HOMOGRAPHY mode: needs --homography           → (X_m,Y_m) meters on plane (no z, no pitch/roll).
  * PIXEL mode: neither provided → centroid/orientation in pixels.

Usage examples:
  python apriltag_tracker_plus.py --video in.mp4 --out track.csv --calib camera.yaml --tag-size 0.05 --backend apriltag
  python apriltag_tracker_plus.py --video in.mp4 --out track.csv --homography homography.yaml --backend apriltag --preview preview.mp4
  python apriltag_tracker_plus.py --video in.mp4 --out track.csv --backend opencv --time pos_msec --smooth 5

Requirements: opencv-python (or opencv-contrib-python for APRILTAG_36h11), numpy, pandas, (optional) apriltag.
"""
import argparse, os, sys, math
import numpy as np
import pandas as pd
import cv2

# ----------------- Utils -----------------
def load_calib(path):
    if not path:
        return None, None
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open calib file: {path}")
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("distortion_coefficients").mat()
    fs.release()
    if K is None or D is None:
        raise RuntimeError("Calibration YAML must contain 'camera_matrix' and 'distortion_coefficients'")
    return K.astype(np.float64), D.astype(np.float64)

def load_H(path):
    if not path: return None
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Failed to open homography file: {path}")
    H = fs.getNode("H").mat()
    fs.release()
    if H is None or H.shape != (3,3):
        raise RuntimeError("homography.yaml missing 3x3 matrix H")
    return H.astype(np.float64)

def euler_zyx_from_R(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        yaw = math.degrees(math.atan2(R[1,0], R[0,0]))
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        roll = math.degrees(math.atan2(R[2,1], R[2,2]))
    else:
        yaw = math.degrees(math.atan2(-R[0,1], R[1,1]))
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        roll = 0.0
    return yaw, pitch, roll

def moving_average(arr, w):
    if w <= 1: return arr
    pad = w // 2
    ypad = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(ypad, kernel, mode="valid")

def get_fps(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return float(fps) if fps and fps > 1e-3 else 30.0

def timestamp(cap, mode, frame_idx, fps):
    if mode == "pos_msec":
        tm = cap.get(cv2.CAP_PROP_POS_MSEC)
        if tm and tm > 0:
            return tm / 1000.0
        # fallback
    return frame_idx / fps

def homography_map(H, pts_xy):
    """pts_xy: Nx2, returns Nx2 mapped in meters."""
    pts = np.hstack([pts_xy, np.ones((len(pts_xy),1))])
    XYw = (H @ pts.T).T
    XYw = XYw[:, :2] / XYw[:, 2:3]
    return XYw

# ----------------- Detection backends -----------------
class BackendApriltag:
    def __init__(self):
        try:
            import apriltag  # type: ignore
        except Exception as e:
            raise RuntimeError("Import apriltag failed. Install with: pip install apriltag") from e
        self._apriltag = apriltag
        self._detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
    def detect(self, gray):
        dets = self._detector.detect(gray)
        out = []
        for d in dets:
            corners = np.array(d.corners, dtype=np.float32)  # TL,TR,BR,BL
            out.append((int(d.tag_id), corners))
        return out

class BackendOpenCV:
    def __init__(self):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("Your OpenCV lacks aruco module. Install opencv-contrib-python.")
        ad = cv2.aruco
        if not hasattr(ad, "DICT_APRILTAG_36h11"):
            raise RuntimeError("OpenCV aruco APRILTAG_36h11 dict not found. Need OpenCV >= 4.7.")
        self._dict = ad.getPredefinedDictionary(ad.DICT_APRILTAG_36h11)
        if hasattr(ad, "DetectorParameters") and hasattr(ad, "ArucoDetector"):
            self._detector = ad.ArucoDetector(self._dict, ad.DetectorParameters())
            self._new_api = True
        else:
            self._params = ad.DetectorParameters_create()
            self._new_api = False
    def detect(self, gray):
        if self._new_api:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dict, parameters=self._params)
        out = []
        if ids is not None:
            ids = ids.flatten()
            for i, cid in enumerate(ids):
                out.append((int(cid), corners[i][0].astype(np.float32)))
        return out

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="AprilTag tracker (tag36h11) with homography/timestamp options.")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--out", default="track.csv", help="Output CSV path")
    ap.add_argument("--calib", default=None, help="OpenCV YAML with camera_matrix & distortion_coefficients")
    ap.add_argument("--tag-size", type=float, default=None, help="Tag side length in meters (for METRIC pose)")
    ap.add_argument("--homography", default=None, help="homography.yaml (from one_shot_plane_calib.py) for planar XY in meters")
    ap.add_argument("--id", type=int, default=None, help="Track only this tag ID; default: auto-select")
    ap.add_argument("--backend", choices=["apriltag","opencv"], default="apriltag", help="Detection backend")
    ap.add_argument("--smooth", type=int, default=1, help="Moving average window (odd; 1=no smooth)")
    ap.add_argument("--preview", default=None, help="Optional path to write preview MP4")
    ap.add_argument("--frame-skip", type=int, default=0, help="Process every Nth frame (0 = all)")
    ap.add_argument("--time", choices=["fps","pos_msec"], default="fps", help="Timestamp source")
    ap.add_argument("--write-parquet", action="store_true", help="Also write a Parquet file next to CSV")
    args = ap.parse_args()

    # Load extras
    K, D = load_calib(args.calib)
    H = load_H(args.homography) if args.homography else None
    metric_pose = (K is not None) and (D is not None) and (args.tag_size is not None and args.tag_size > 0)
    has_plane = H is not None

    backend = BackendApriltag() if args.backend == "apriltag" else BackendOpenCV()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}", file=sys.stderr); sys.exit(1)
    fps = get_fps(cap)
    dt_default = 1.0 / fps

    # Preview writer
    writer = None
    if args.preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        writer = cv2.VideoWriter(args.preview, fourcc, fps, (W, Hh))

    rows = []
    traj_pts = []
    target_id = args.id
    frame_idx = 0
    proc_idx = 0
    prev_center = None

    # PnP flags
    PNP_IPPE = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", None)
    PNP_ITER = cv2.SOLVEPNP_ITERATIVE

    while True:
        ok, frame = cap.read()
        if not ok: break
        if args.frame_skip and (proc_idx % (args.frame_skip+1) != 0):
            proc_idx += 1
            frame_idx += 1
            continue
        proc_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = backend.detect(gray)
        t = timestamp(cap, args.time, frame_idx, fps)

        if dets:
            # Auto-select target by area (largest) if not set; then stickiness by nearest center
            if target_id is None:
                areas = []
                for cid, c in dets:
                    area = cv2.contourArea(c.astype(np.int32))
                    areas.append((area, cid))
                areas.sort(reverse=True)
                target_id = areas[0][1]

            # find candidate detections for target_id
            candidates = [(cid, c) for cid, c in dets if cid == target_id]
            if not candidates:
                # stickiness fallback: nearest to prev_center if present
                if prev_center is not None:
                    candidates = dets
                    candidates.sort(key=lambda x: np.linalg.norm(np.mean(x[1], axis=0) - prev_center))
                    cid, corners = candidates[0]
                else:
                    cid, corners = dets[0]
            else:
                cid, corners = candidates[0]

            # Basic pixel center & edge
            cx, cy = float(corners[:,0].mean()), float(corners[:,1].mean())
            dx, dy = (corners[1][0]-corners[0][0], corners[1][1]-corners[0][1])
            yaw_px = math.degrees(math.atan2(dy, dx))
            prev_center = np.array([cx, cy], dtype=np.float32)
            traj_pts.append((int(cx), int(cy)))  # for preview path, both modes

            # Preview outline
            if writer is not None:
                cv2.polylines(frame, [corners.astype(np.int32)], isClosed=True, color=(0,255,0), thickness=2)

            reproj_err = float("nan")
            yaw_deg, pitch_deg, roll_deg = float("nan"), float("nan"), float("nan")
            x_out = y_out = z_out = float("nan")
            Xw = Yw = float("nan")

            # Metric pose via PnP
            if metric_pose:
                s = float(args.tag_size)
                objp = np.array([[-s/2, -s/2, 0],
                                 [ s/2, -s/2, 0],
                                 [ s/2,  s/2, 0],
                                 [-s/2,  s/2, 0]], dtype=np.float64)
                imgp = corners.astype(np.float64)
                rvec = np.zeros((3,1), dtype=np.float64)
                tvec = np.zeros((3,1), dtype=np.float64)
                ok_pnp = False
                if PNP_IPPE is not None:
                    ok_pnp, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=PNP_IPPE)
                if not ok_pnp:
                    ok_pnp, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, rvec, tvec, useExtrinsicGuess=False, flags=PNP_ITER)
                if ok_pnp:
                    R, _ = cv2.Rodrigues(rvec)
                    yaw_deg, pitch_deg, roll_deg = euler_zyx_from_R(R)
                    x_out, y_out, z_out = float(tvec[0]), float(tvec[1]), float(tvec[2])
                    # Reprojection error
                    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
                    proj = proj.reshape(-1,2)
                    reproj_err = float(np.mean(np.linalg.norm(proj - imgp, axis=1)))
                    # Draw axes if preview
                    if writer is not None:
                        try:
                            cv2.drawFrameAxes(frame, K, D, rvec, tvec, s*0.75)
                        except Exception:
                            pass

            # Homography plane mapping (independent of metric pose)
            if has_plane:
                XY = homography_map(H, np.array([[cx, cy]], dtype=np.float64))[0]
                Xw, Yw = float(XY[0]), float(XY[1])

            # Write row
            rows.append((frame_idx, t, cid,
                         x_out, y_out, z_out, yaw_deg, pitch_deg, roll_deg,
                         cx, cy, yaw_px, Xw, Yw, reproj_err))

            # Draw path
            if writer is not None and len(traj_pts) > 1:
                cv2.line(frame, traj_pts[-2], traj_pts[-1], (255,0,0), 2)

        if writer is not None:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if not rows:
        print("No detections written. Check visibility, backend, or tag ID.", file=sys.stderr)
        sys.exit(2)

    # Build DataFrame
    cols = ["frame","t","id",
            "x_m","y_m","z_m","yaw_deg","pitch_deg","roll_deg",
            "x_px","y_px","yaw_px_deg","X_m","Y_m","reproj_px"]
    df = pd.DataFrame(rows, columns=cols).sort_values("t").reset_index(drop=True)

    # Smoothing (apply to available numeric columns separately, NaN safe)
    w = int(args.smooth) if args.smooth else 1
    if w % 2 == 0: w += 1
    for col in ["x_m","y_m","z_m","yaw_deg","pitch_deg","roll_deg","x_px","y_px","yaw_px_deg","X_m","Y_m"]:
        if col in df and df[col].notna().any():
            arr = df[col].to_numpy()
            mask = ~np.isnan(arr)
            arr_s = arr.copy()
            arr_s[mask] = moving_average(arr[mask], w)
            df[col] = arr_s

    # Choose coordinate source for derivatives priority: metric pose > homography > pixels
    if df[["x_m","y_m"]].notna().any(axis=None):
        x_series, y_series = df["x_m"], df["y_m"]
        unit_xy = "m"
    elif df[["X_m","Y_m"]].notna().any(axis=None):
        x_series, y_series = df["X_m"], df["Y_m"]
        unit_xy = "m"
    else:
        x_series, y_series = df["x_px"], df["y_px"]
        unit_xy = "px"

    # Derivatives w.r.t timestamps (handle non-monotonic t with fallback)
    t_arr = df["t"].to_numpy()
    if np.any(np.diff(t_arr) <= 0):
        # fallback to uniform
        t_arr = np.arange(len(df)) * (1.0 / (getattr(cv2, "CAP_PROP_FPS", 30.0) or 30.0))

    def deriv(v):
        return np.gradient(v, t_arr)

    vx = deriv(x_series.to_numpy())
    vy = deriv(y_series.to_numpy())
    vmag = np.hypot(vx, vy)
    ax = deriv(vx)
    ay = deriv(vy)
    amag = np.hypot(ax, ay)

    df[f"vx_{unit_xy}/s"] = vx
    df[f"vy_{unit_xy}/s"] = vy
    df[f"v_{unit_xy}/s"]  = vmag
    df[f"ax_{unit_xy}/s2"] = ax
    df[f"ay_{unit_xy}/s2"] = ay
    df[f"a_{unit_xy}/s2"]  = amag

    # Angular velocity: prefer metric yaw, else pixel yaw
    yaw_src = "yaw_deg" if df["yaw_deg"].notna().any() else "yaw_px_deg"
    yaw_rad = np.deg2rad(df[yaw_src].fillna(method="ffill").fillna(method="bfill").to_numpy())
    yaw_unwrap = np.unwrap(yaw_rad)
    omega = np.gradient(yaw_unwrap, t_arr)
    df["omega_deg/s"] = np.rad2deg(omega)
    df["yaw_src"] = yaw_src

    # Save
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    if args.write_parquet:
        df.to_parquet(os.path.splitext(args.out)[0] + ".parquet", index=False)
    print(f"Wrote {len(df)} rows → {args.out}")
    if args.preview:
        print(f"Preview written → {args.preview}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
