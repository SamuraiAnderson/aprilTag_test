#!/usr/bin/env python3
"""
generate_printables.py
----------------------
Generate printable AprilTag sheets (tag36h11) and calibration boards (ChArUco / Chessboard) to PDF.
Accurate physical sizing via ReportLab.

Requirements:
    pip install opencv-contrib-python reportlab numpy

Examples:
  # AprilTag sheet (A4): IDs 0..7, each 50mm, 10mm white border, 5mm spacing
  python generate_printables.py tags --ids 0 1 2 3 4 5 6 7 --tag-size-mm 50 --border-mm 10 --spacing-mm 5 --outfile tags_a4.pdf

  # ChArUco board (A4): 7x5 squares, square 30mm, marker 24mm, dictionary 4X4_100
  python generate_printables.py charuco --squares-x 7 --squares-y 5 --square-mm 30 --marker-mm 24 --dict 4X4_100 --outfile charuco_a4.pdf

  # Chessboard (A4): 9x6 inner corners, square 30mm
  python generate_printables.py chessboard --cols 9 --rows 6 --square-mm 30 --outfile chessboard_a4.pdf

Notes:
- AprilTag generation relies on OpenCV aruco APRILTAG_36h11 (OpenCV >= 4.7).
- Page defaults to A4 portrait; use --page A4|LETTER or --page-mm W H for custom size.
"""
import argparse, os, io
import numpy as np
import cv2
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm as RLMM
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader

def mm_to_pt(mm): return mm * RLMM  # 1 mm in points

def page_size_from_arg(arg):
    if arg.upper() == "A4": return A4
    if arg.upper() == "LETTER": return LETTER
    # custom "W H" in mm
    parts = arg.split()
    if len(parts) == 2:
        w_mm, h_mm = float(parts[0]), float(parts[1])
        return (mm_to_pt(w_mm), mm_to_pt(h_mm))
    raise ValueError("Unsupported page size. Use A4, LETTER or 'W H' in mm via --page-mm.")

def ensure_aruco_dict(name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV aruco not found. Install opencv-contrib-python")
    ad = cv2.aruco
    mapping = {
        "4X4_50": ad.DICT_4X4_50,
        "4X4_100": ad.DICT_4X4_100,
        "5X5_50": ad.DICT_5X5_50,
        "5X5_100": ad.DICT_5X5_100,
        "6X6_50": ad.DICT_6X6_50,
        "6X6_100": ad.DICT_6X6_100,
        "APRILTAG_36H11": getattr(ad, "DICT_APRILTAG_36h11"),
    }
    if name not in mapping:
        raise RuntimeError(f"Unsupported dict {name}")
    return ad.getPredefinedDictionary(mapping[name])

def draw_apriltag_img(tag_id, dict_name="APRILTAG_36H11", pixels=800):
    ad = cv2.aruco
    dct = ensure_aruco_dict(dict_name)
    # Use the newer API that's available in OpenCV 4.7+
    try:
        # New API (OpenCV 4.7+)
        img = ad.generateImageMarker(dct, tag_id, pixels, 1)
    except AttributeError:
        # Fallback for older versions
        img = np.zeros((pixels, pixels), dtype=np.uint8)
        ad.drawMarker(dct, tag_id, pixels, img, 1)
    return img

def place_image_on_pdf(c, img_np, x_pt, y_pt, w_mm, h_mm):
    # Convert numpy image to ImageReader
    if len(img_np.shape) == 2:
        # grayscale to RGB
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    ok, png = cv2.imencode(".png", img_np)
    if not ok:
        raise RuntimeError("Failed to encode image")
    bio = io.BytesIO(png.tobytes())
    c.drawImage(ImageReader(bio), x_pt, y_pt, width=mm_to_pt(w_mm), height=mm_to_pt(h_mm))

def cmd_tags(args):
    page = A4 if args.page.upper() == "A4" else LETTER if args.page.upper() == "LETTER" else page_size_from_arg(args.page)
    W, H = page
    c = canvas.Canvas(args.outfile, pagesize=page)
    margin = mm_to_pt(args.margin_mm)
    x = margin
    y = H - margin
    tag_mm = args.tag_size_mm
    cell_w = tag_mm + 2*args.border_mm
    cell_h = tag_mm + 2*args.border_mm + args.caption_mm
    col = 0

    for idx, tag_id in enumerate(args.ids):
        if idx == 0:
            # draw guide text
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.black)
            c.drawString(margin, H - margin + 2, f"AprilTag {args.dict}  size={tag_mm}mm  border={args.border_mm}mm  spacing={args.spacing_mm}mm")

        if x + mm_to_pt(cell_w) > W - margin:
            # new row
            x = margin
            y -= mm_to_pt(cell_h + args.spacing_mm)
            col = 0
            if y - mm_to_pt(cell_h) < margin:
                c.showPage()
                y = H - margin

        # cell origin
        img = draw_apriltag_img(tag_id, dict_name=args.dict, pixels= args.pixels)
        # draw white border rectangle
        c.setFillColor(colors.white)
        c.rect(x, y - mm_to_pt(cell_h), mm_to_pt(cell_w), mm_to_pt(cell_h - args.caption_mm), stroke=0, fill=1)
        # place tag centered inside border
        ox = x + mm_to_pt(args.border_mm)
        oy = y - mm_to_pt(args.border_mm + tag_mm + args.caption_mm)
        place_image_on_pdf(c, img, ox, oy, tag_mm, tag_mm)

        # caption
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 10)
        c.drawCentredString(x + mm_to_pt(cell_w*0.5), y - mm_to_pt(cell_h - args.caption_mm*0.7), f"ID {tag_id}")

        x += mm_to_pt(cell_w + args.spacing_mm)
        col += 1

    c.save()
    print(f"Wrote {args.outfile}")

def cmd_charuco(args):
    page = A4 if args.page.upper() == "A4" else LETTER if args.page.upper() == "LETTER" else page_size_from_arg(args.page)
    W, H = page
    dict_ = ensure_aruco_dict(args.dict)
    
    # Handle different OpenCV versions for CharucoBoard creation
    try:
        # New API (OpenCV 4.7+)
        board = cv2.aruco.CharucoBoard(
            (args.squares_x, args.squares_y),
            args.square_mm/1000.0, args.marker_mm/1000.0, dict_
        )
    except (AttributeError, TypeError):
        # Fallback for older versions
        board = cv2.aruco.CharucoBoard_create(
            args.squares_x, args.squares_y,
            args.square_mm/1000.0, args.marker_mm/1000.0, dict_
        )
    
    # Render at high resolution
    px_w = int(args.square_mm * args.squares_x * args.dpi / 25.4)
    px_h = int(args.square_mm * args.squares_y * args.dpi / 25.4)
    
    # Handle different API for image generation
    try:
        # New API style
        img = board.generateImage((px_w, px_h))
    except:
        # Alternative approach
        img = board.draw((px_w, px_h))
    
    c = canvas.Canvas(args.outfile, pagesize=page)
    margin = mm_to_pt(args.margin_mm)
    place_w_mm = args.square_mm*args.squares_x
    place_h_mm = args.square_mm*args.squares_y
    x = (W - mm_to_pt(place_w_mm)) / 2
    y = (H - mm_to_pt(place_h_mm)) / 2
    place_image_on_pdf(c, img, x, y, place_w_mm, place_h_mm)
    c.setFont("Helvetica", 9)
    c.drawString(margin, H - margin + 2, f"ChArUco {args.dict}  {args.squares_x}x{args.squares_y}  square={args.square_mm}mm marker={args.marker_mm}mm")
    c.save()
    print(f"Wrote {args.outfile}")

def cmd_chessboard(args):
    page = A4 if args.page.upper() == "A4" else LETTER if args.page.upper() == "LETTER" else page_size_from_arg(args.page)
    W, H = page
    cols, rows = args.cols, args.rows
    sq = args.square_mm
    board_w_mm = sq * cols
    board_h_mm = sq * rows
    x0 = (W - mm_to_pt(board_w_mm)) / 2
    y0 = (H - mm_to_pt(board_h_mm)) / 2
    c = canvas.Canvas(args.outfile, pagesize=page)
    # Draw squares
    for r in range(rows):
        for col in range(cols):
            if (r + col) % 2 == 0:
                c.setFillColor(colors.black)
            else:
                c.setFillColor(colors.white)
            x = x0 + mm_to_pt(col * sq)
            y = y0 + mm_to_pt((rows-1-r) * sq)
            c.rect(x, y, mm_to_pt(sq), mm_to_pt(sq), stroke=0, fill=1)
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 9)
    margin = mm_to_pt(10)
    c.drawString(margin, H - margin + 2, f"Chessboard {cols}x{rows}  square={sq}mm")
    c.save()
    print(f"Wrote {args.outfile}")

def main():
    parser = argparse.ArgumentParser(description="Generate printable AprilTags and calibration boards to PDF.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tags = sub.add_parser("tags", help="Generate AprilTag sheet (tag36h11)")
    p_tags.add_argument("--ids", nargs="+", type=int, required=True, help="List of tag IDs")
    p_tags.add_argument("--dict", default="APRILTAG_36H11", help="Dictionary name (APRILTAG_36H11)")
    p_tags.add_argument("--tag-size-mm", type=float, required=True, help="Black square size (mm)")
    p_tags.add_argument("--border-mm", type=float, default=10.0, help="White border around tag (mm)")
    p_tags.add_argument("--spacing-mm", type=float, default=5.0, help="Spacing between cells (mm)")
    p_tags.add_argument("--caption-mm", type=float, default=6.0, help="Height reserved for caption (mm)")
    p_tags.add_argument("--pixels", type=int, default=1000, help="Internal render resolution per tag (pixels)")
    p_tags.add_argument("--page", default="A4", help="A4|LETTER or 'W H' in mm (e.g. '300 200')")
    p_tags.add_argument("--margin-mm", type=float, default=10.0, help="Page margin (mm)")
    p_tags.add_argument("--outfile", required=True, help="Output PDF path")
    p_tags.set_defaults(func=cmd_tags)

    p_charuco = sub.add_parser("charuco", help="Generate ChArUco calibration board")
    p_charuco.add_argument("--squares-x", type=int, required=True, help="Number of squares along X")
    p_charuco.add_argument("--squares-y", type=int, required=True, help="Number of squares along Y")
    p_charuco.add_argument("--square-mm", type=float, required=True, help="Square size (mm)")
    p_charuco.add_argument("--marker-mm", type=float, required=True, help="Marker size inside square (mm)")
    p_charuco.add_argument("--dict", default="4X4_100", help="ArUco dictionary for ChArUco markers")
    p_charuco.add_argument("--dpi", type=int, default=600, help="Render DPI for image generation")
    p_charuco.add_argument("--page", default="A4", help="A4|LETTER or 'W H' in mm")
    p_charuco.add_argument("--margin-mm", type=float, default=10.0, help="Page margin (mm)")
    p_charuco.add_argument("--outfile", required=True, help="Output PDF path")
    p_charuco.set_defaults(func=cmd_charuco)

    p_chess = sub.add_parser("chessboard", help="Generate classic chessboard calibration pattern")
    p_chess.add_argument("--cols", type=int, required=True, help="Number of squares along X")
    p_chess.add_argument("--rows", type=int, required=True, help="Number of squares along Y")
    p_chess.add_argument("--square-mm", type=float, required=True, help="Square size (mm)")
    p_chess.add_argument("--page", default="A4", help="A4|LETTER or 'W H' in mm")
    p_chess.add_argument("--outfile", required=True, help="Output PDF path")
    p_chess.set_defaults(func=cmd_chessboard)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
