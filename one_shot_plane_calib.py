#!/usr/bin/env python3
"""
one_shot_plane_calib.py
-----------------------
Single-image *planar* calibration using one AprilTag (tag36h11) of known size.

Goal:
  Compute a 3x3 homography H that maps image pixel coordinates (u,v,1)
  to world *plane* coordinates (X,Y,1) in meters, assuming the tag lies
  on the same plane where your robot moves and the camera is fixed.

Why this works:
  A single rectangular tag provides 4 non-collinear points with known
  world positions (0,0), (s,0), (s,s), (0,s). This lets us solve a
  homography for the plane without a full camera calibration.
  It's ideal for "top-down-ish" fixed rigs where you only need metric XY
  on the ground plane. (Z and true intrinsics require full calibration.)

Usage:
  python one_shot_plane_calib.py --image frame.jpg --tag-size 0.05 --out homography.yaml
  # If you don't have the 'apriltag' Python lib, use OpenCV backend (needs opencv-contrib-python>=4.7):
  python one_shot_plane_calib.py --image frame.jpg --tag-size 0.05 --backend opencv --out homography.yaml

Output:
  homography.yaml with:
    H (3x3), tag_size_m, image_path, img_width/height, tag_corners_px (TL,TR,BR,BL)

Then apply to tracks (see apply_homography_to_track.py).
"""
import argparse, os, sys, json
import numpy as np
import cv2

def detect_apriltag(gray, backend="apriltag"):
    if backend == "apriltag":
        try:
            import apriltag  # pip install apriltag
        except Exception as e:
            raise RuntimeError("Backend 'apriltag' requires 'pip install apriltag'") from e
        det = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
        results = det.detect(gray)
        if not results:
            return None
        # take first detected tag
        r = results[0]
        corners = np.array(r.corners, dtype=np.float32)  # (4,2) TL,TR,BR,BL
        return corners
    else:
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV aruco not found. Install opencv-contrib-python.")
        ad = cv2.aruco
        if not hasattr(ad, "DICT_APRILTAG_36h11"):
            raise RuntimeError("Your OpenCV lacks APRILTAG_36h11 dictionary (need >=4.7).")
        dct = ad.getPredefinedDictionary(ad.DICT_APRILTAG_36h11)
        if hasattr(ad, "DetectorParameters") and hasattr(ad, "ArucoDetector"):
            det = ad.ArucoDetector(dct, ad.DetectorParameters())
            corners, ids, _ = det.detectMarkers(gray)
        else:
            params = ad.DetectorParameters_create()
            corners, ids, _ = ad.detectMarkers(gray, dct, parameters=params)
        if ids is None or len(ids) == 0:
            return None
        return corners[0][0].astype(np.float32)  # first tag

def save_yaml(path, data: dict):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open for write: {path}")
    # Write basic scalars
    for k in ["img_width","img_height"]:
        if k in data: fs.write(k, int(data[k]))
    # Write floats/matrices
    if "H" in data: fs.write("H", data["H"])
    if "tag_size_m" in data: fs.write("tag_size_m", float(data["tag_size_m"]))
    # Write arrays as strings (for readability)
    if "tag_corners_px" in data:
        arr = data["tag_corners_px"].astype(np.float32)
        fs.write("tag_corners_px", arr)
    if "image_path" in data:
        fs.write("image_path", data["image_path"])
    fs.release()

def main():
    ap = argparse.ArgumentParser(description="Single-image planar calibration from one AprilTag (tag36h11).")
    ap.add_argument("--image", required=True, help="Input image path (frame)")
    ap.add_argument("--tag-size", type=float, required=True, help="Tag black square side (meters)")
    ap.add_argument("--backend", choices=["apriltag","opencv"], default="apriltag", help="Detector backend")
    ap.add_argument("--out", default="homography.yaml", help="Output YAML path")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Cannot read image: {args.image}", file=sys.stderr); sys.exit(1)

    corners = detect_apriltag(img, backend=args.backend)
    if corners is None:
        print("No AprilTag detected. Make sure tag36h11 is visible.", file=sys.stderr); sys.exit(2)

    # image points (TL,TR,BR,BL)
    img_pts = corners.astype(np.float64)  # (4,2)

    s = float(args.tag_size)
    # world points in meters in the plane, following same order TL,TR,BR,BL
    world_pts = np.array([[0, 0],
                          [s, 0],
                          [s, s],
                          [0, s]], dtype=np.float64)

    H, mask = cv2.findHomography(img_pts, world_pts, method=0)
    if H is None:
        print("findHomography failed.", file=sys.stderr); sys.exit(3)

    data = {
        "H": H,
        "tag_size_m": s,
        "image_path": args.image,
        "img_width": img.shape[1],
        "img_height": img.shape[0],
        "tag_corners_px": img_pts.astype(np.float32),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_yaml(args.out, data)
    print(f"Wrote homography → {args.out}")
    print("Now apply it to track CSV with apply_homography_to_track.py")

if __name__ == "__main__":
    main()
