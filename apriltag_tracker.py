#!/usr/bin/env python3
"""
apriltag_tracker.py
-------------------
High-precision AprilTag tracker for a *fixed camera* experiment.
- Detects tag36h11 AprilTags
- Outputs trajectory (x,y[,z]), linear speed/acceleration, and angular velocity
- Works in:
  * METRIC mode: needs --calib (OpenCV YAML) + --tag-size (meters) → pose via solvePnP
  * PIXEL mode: no calibration → centroid/orientation in pixels

Also supports an optional preview video with axes drawn when calibration is provided.

Requirements:
  pip install opencv-python numpy pandas
  (optional & recommended for detection): pip install apriltag
  (or OpenCV >= 4.7 with aruco APRILTAG_36h11 support for --backend opencv)

Usage examples:
  python apriltag_tracker.py --video in.mp4 --out track.csv --calib camera.yaml --tag-size 0.05 --backend apriltag
  python apriltag_tracker.py --video in.mp4 --out track.csv --backend opencv --id 7 --smooth 5 --preview preview.mp4

Calibration YAML must contain "camera_matrix" and "distortion_coefficients".
"""
import argparse, os, sys, math
import numpy as np
import pandas as pd
import cv2

# ----------------- Utils -----------------
def mm(value):  # not used, kept for completeness
    return value

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

def euler_zyx_from_R(R):
    """
    Convert rotation matrix to ZYX euler angles (yaw around Z, pitch around Y, roll around X), in degrees.
    """
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        yaw = math.degrees(math.atan2(R[1,0], R[0,0]))
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        roll = math.degrees(math.atan2(R[2,1], R[2,2]))
    else:
        # Gimbal lock: pitch = +/-90 deg
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
            # corners order: [p0,p1,p2,p3] starting top-left clockwise (apriltag convention)
            corners = np.array(d.corners, dtype=np.float32)  # (4,2)
            out.append((int(d.tag_id), corners))
        return out

class BackendOpenCV:
    def __init__(self):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("Your OpenCV lacks aruco module. Install opencv-contrib-python.")
        ad = cv2.aruco
        if not hasattr(ad, "DICT_APRILTAG_36h11"):
            raise RuntimeError("OpenCV aruco APRILTAG_36h11 dict not found. Need OpenCV >= 4.7 (opencv-contrib-python).")
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
                out.append((int(cid), corners[i][0].astype(np.float32)))  # (4,2)
        return out

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser(description="High-precision AprilTag tracker (tag36h11).")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--out", default="track.csv", help="Output CSV path")
    ap.add_argument("--calib", default=None, help="OpenCV YAML with camera_matrix & distortion_coefficients")
    ap.add_argument("--tag-size", type=float, default=None, help="Tag side length in meters (required for METRIC mode)")
    ap.add_argument("--id", type=int, default=None, help="Track only this tag ID; default: first seen")
    ap.add_argument("--backend", choices=["apriltag","opencv"], default="apriltag", help="Detection backend")
    ap.add_argument("--smooth", type=int, default=1, help="Moving average window (odd; 1=no smooth)")
    ap.add_argument("--preview", default=None, help="Optional path to write preview MP4 (requires calib)")
    ap.add_argument("--frame-skip", type=int, default=0, help="Process every Nth frame (0 = all)")
    args = ap.parse_args()

    # Load calib if provided
    K, D = load_calib(args.calib)
    metric = (K is not None) and (D is not None) and (args.tag_size is not None and args.tag_size > 0)

    # Detector
    backend = BackendApriltag() if args.backend == "apriltag" else BackendOpenCV()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open video: {args.video}", file=sys.stderr); sys.exit(1)
    fps = get_fps(cap)
    dt_default = 1.0 / fps

    # Preview writer
    writer = None
    if args.preview:
        if not metric:
            print("Preview requires calibration + tag-size (for pose). Proceeding without axes.", file=sys.stderr)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        writer = cv2.VideoWriter(args.preview, fourcc, fps, (W, H))

    rows = []
    traj_pts = []
    target_id = args.id
    frame_idx = 0
    proc_idx = 0

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
        t = frame_idx / fps

        if dets:
            if target_id is None:
                target_id = dets[0][0]
            # find target
            det = None
            for cid, corners in dets:
                if cid == target_id:
                    det = (cid, corners); break
            if det is not None:
                cid, corners = det
                # corners order assumed TL,TR,BR,BL
                if metric:
                    # 3D tag corner coordinates in tag frame (Z=0 plane)
                    s = float(args.tag_size)
                    objp = np.array([[-s/2, -s/2, 0],
                                     [ s/2, -s/2, 0],
                                     [ s/2,  s/2, 0],
                                     [-s/2,  s/2, 0]], dtype=np.float64)
                    imgp = corners.astype(np.float64)
                    ok_pnp, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
                    if ok_pnp:
                        R, _ = cv2.Rodrigues(rvec)
                        yaw, pitch, roll = euler_zyx_from_R(R)
                        x, y, z = float(tvec[0]), float(tvec[1]), float(tvec[2])
                        rows.append((frame_idx, t, cid, x, y, z, yaw, pitch, roll))
                        traj_pts.append((int(imgp[:,0].mean()), int(imgp[:,1].mean())))
                        # Preview draw
                        if writer is not None:
                            cv2.polylines(frame, [corners.astype(np.int32)], isClosed=True, color=(0,255,0), thickness=2)
                            try:
                                cv2.drawFrameAxes(frame, K, D, rvec, tvec, s*0.75)  # draw axes of length ~0.75*tag
                            except Exception:
                                pass
                            if len(traj_pts) > 1:
                                for i in range(1, len(traj_pts)):
                                    cv2.line(frame, traj_pts[i-1], traj_pts[i], (255,0,0), 2)
                    else:
                        # fallback to pixel mode if PnP fails
                        cx, cy = float(corners[:,0].mean()), float(corners[:,1].mean())
                        dx, dy = (corners[1][0]-corners[0][0], corners[1][1]-corners[0][1])
                        yaw = math.degrees(math.atan2(dy, dx))
                        rows.append((frame_idx, t, cid, cx, cy, float("nan"), yaw, float("nan"), float("nan")))
                else:
                    cx, cy = float(corners[:,0].mean()), float(corners[:,1].mean())
                    dx, dy = (corners[1][0]-corners[0][0], corners[1][1]-corners[0][1])
                    yaw = math.degrees(math.atan2(dy, dx))
                    rows.append((frame_idx, t, cid, cx, cy, float("nan"), yaw, float("nan"), float("nan")))
                    if writer is not None:
                        cv2.polylines(frame, [corners.astype(np.int32)], isClosed=True, color=(0,255,0), thickness=2)
                        if len(traj_pts) > 1:
                            for i in range(1, len(traj_pts)):
                                cv2.line(frame, traj_pts[i-1], traj_pts[i], (255,0,0), 2)

        if writer is not None:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    if not rows:
        print("No detections written. Check visibility, backend, or tag ID.", file=sys.stderr)
        sys.exit(2)

    df = pd.DataFrame(rows, columns=["frame","t","id","x","y","z","yaw_deg","pitch_deg","roll_deg"]).sort_values("t").reset_index(drop=True)

    # Smoothing
    w = int(args.smooth) if args.smooth else 1
    if w % 2 == 0: w += 1
    for col in ["x","y","z","yaw_deg","pitch_deg","roll_deg"]:
        if col in df and df[col].notna().any():
            arr = df[col].to_numpy()
            mask = ~np.isnan(arr)
            if mask.any():
                arr_s = arr.copy()
                arr_s[mask] = moving_average(arr[mask], w)
                df[col] = arr_s

    # Derivatives with respect to real timestamps (handles VFR)
    t = df["t"].to_numpy()
    if np.any(np.diff(t) <= 0):
        t = np.arange(len(df)) * dt_default  # fallback

    # Position -> velocity/accel
    for axis in ["x","y","z"]:
        arr = df[axis].to_numpy()
        if np.isnan(arr).all(): continue
        v = np.gradient(arr, t)
        a = np.gradient(v, t)
        unit = "m" if metric else "px"
        df[f"v{axis}_{unit}/s"] = v
        df[f"a{axis}_{unit}/s2"] = a
    # Speed magnitude (xy)
    if not np.isnan(df["x"]).all() and not np.isnan(df["y"]).all():
        vx = df["vx_m/s"] if metric and "vx_m/s" in df else df.get("vx_px/s", pd.Series(np.gradient(df["x"], t)))
        vy = df["vy_m/s"] if metric and "vy_m/s" in df else df.get("vy_px/s", pd.Series(np.gradient(df["y"], t)))
        vmag = np.hypot(np.array(vx), np.array(vy))
        df["v_xy_"+("m/s" if metric else "px/s")] = vmag

    # Angular velocity from yaw (unwrap)
    yaw_rad = np.deg2rad(df["yaw_deg"].to_numpy())
    yaw_unwrap = np.unwrap(yaw_rad)
    omega = np.gradient(yaw_unwrap, t)  # rad/s
    df["omega_deg/s"] = np.rad2deg(omega)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows → {args.out}")
    if args.preview:
        print(f"Preview written → {args.preview}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
