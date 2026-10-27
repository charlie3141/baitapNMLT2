import os
import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tkinter as tk
from tkinter import colorchooser

# ==================== CAU HINH ====================
CAM_ID = 0
BRUSH_SIZE = 6
SMOOTHING = 0.35


def find_model(name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, name),
        os.path.join(script_dir, "..", name),
        os.path.join(os.getcwd(), name),
    ]
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Khong tim thay {name}. Da tim:\n  "
        + "\n  ".join(os.path.abspath(c) for c in candidates)
    )


HAND_MODEL = find_model("hand_landmarker.task")
FACE_MODEL = find_model("face_landmarker.task")
print("Hand model:", HAND_MODEL)
print("Face model:", FACE_MODEL)

COLORS = [
    ("DO",    (0, 0, 255)),
    ("XANH",  (0, 255, 0)),
    ("LAM",   (255, 0, 0)),
    ("VANG",  (0, 255, 255)),
    ("TRANG", (255, 255, 255)),
    ("DEN",   (0, 0, 0)),
]
current_color = COLORS[0][1]

# ==================== TRANG THAI ====================
mode_face_filter = False
mode_mouse_draw = False
should_quit = False

canvas = None
face_strokes = []        # [(color, thickness, [(u,v), ...])]
face_transform = None    # (cx, cy, angle, scale)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class StrokeState:
    def __init__(self):
        self.prev_screen = None    # diem truoc (screen-space, che do canvas)
        self.smooth_face = None    # diem smooth (face-space)
        self.face_pts = None       # danh sach (u, v) dang ve
        self.color = None
        self.thickness = None

    def reset(self):
        self.prev_screen = None
        self.smooth_face = None
        self.face_pts = None
        self.color = None
        self.thickness = None

    def is_active(self):
        return self.prev_screen is not None or self.face_pts is not None


hand_stroke = StrokeState()
mouse_stroke = StrokeState()

# ==================== LANDMARKERS ====================
hand_landmarker = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )
)
face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

# ==================== HELPERS ====================
def draw_hand(frame, landmarks_xy):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, landmarks_xy[a], landmarks_xy[b], (0, 200, 0), 2)
    for i, pt in enumerate(landmarks_xy):
        r = 5 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, pt, r, (0, 0, 255), -1)


def compute_face_transform(landmarks, w, h):
    """Goc = trung diem 2 mat, goc xoay = duong noi 2 mat, scale = khoang cach 2 mat."""
    try:
        le, re = landmarks[33], landmarks[263]
    except IndexError:
        return None
    lex, ley = le.x * w, le.y * h
    rex, rey = re.x * w, re.y * h
    cx, cy = (lex + rex) * 0.5, (ley + rey) * 0.5
    dx, dy = rex - lex, rey - ley
    angle = math.atan2(dy, dx)
    scale = math.hypot(dx, dy)
    if scale < 5:
        return None
    return (cx, cy, angle, scale)


def screen_to_face(px, py, t):
    cx, cy, angle, scale = t
    dx, dy = px - cx, py - cy
    ca, sa = math.cos(-angle), math.sin(-angle)
    return ((dx * ca - dy * sa) / scale,
            (dx * sa + dy * ca) / scale)


def face_to_screen(u, v, t):
    cx, cy, angle, scale = t
    ca, sa = math.cos(angle), math.sin(angle)
    dx, dy = u * scale, v * scale
    return (int(cx + dx * ca - dy * sa),
            int(cy + dx * sa + dy * ca))


def index_is_extended(lm):
    return lm[8].y < lm[6].y


def add_stroke_point(stroke, x, y):
    if mode_face_filter:
        if face_transform is None:
            return
        u_raw, v_raw = screen_to_face(x, y, face_transform)
        if stroke.smooth_face is None:
            stroke.smooth_face = (u_raw, v_raw)
        else:
            su, sv = stroke.smooth_face
            stroke.smooth_face = (su + SMOOTHING * (u_raw - su),
                                   sv + SMOOTHING * (v_raw - sv))
        u, v = stroke.smooth_face
        if stroke.face_pts is None:
            stroke.face_pts = [(u, v)]
            stroke.color = current_color
            stroke.thickness = BRUSH_SIZE
        else:
            lu, lv = stroke.face_pts[-1]
            if abs(u - lu) > 0.003 or abs(v - lv) > 0.003:
                stroke.face_pts.append((u, v))
    else:
        if canvas is None:
            return
        pt = (x, y)
        if stroke.prev_screen is not None:
            cv2.line(canvas, stroke.prev_screen, pt,
                     current_color, BRUSH_SIZE, cv2.LINE_AA)
        stroke.prev_screen = pt


def finish_stroke(stroke):
    if stroke.face_pts is not None and len(stroke.face_pts) >= 2:
        face_strokes.append((stroke.color, stroke.thickness, stroke.face_pts))
    stroke.reset()


def draw_face_stroke_on_frame(output, pts, color, thickness):
    if face_transform is None:
        return
    screen_pts = [face_to_screen(u, v, face_transform) for u, v in pts]
    if len(screen_pts) == 1:
        cv2.circle(output, screen_pts[0], max(1, thickness // 2), color, -1)
        return
    for i in range(len(screen_pts) - 1):
        cv2.line(output, screen_pts[i], screen_pts[i + 1],
                 color, thickness, cv2.LINE_AA)


def choose_color():
    global current_color
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    color = colorchooser.askcolor(title="Chon mau but ve", parent=root)
    root.destroy()
    if color and color[0]:
        r, g, b = map(int, color[0])
        current_color = (b, g, r)


def toggle_face_filter():
    global mode_face_filter
    mode_face_filter = not mode_face_filter
    finish_stroke(hand_stroke)
    finish_stroke(mouse_stroke)


def toggle_mouse_draw():
    global mode_mouse_draw
    mode_mouse_draw = not mode_mouse_draw
    if not mode_mouse_draw:
        finish_stroke(mouse_stroke)


# ==================== LAYOUT TOOLBAR ====================
ROW1_Y1, ROW1_Y2 = 8, 30
ROW2_Y1, ROW2_Y2 = 35, 67
FACE_BTN = (145, ROW1_Y1, 215, ROW1_Y2)
MOUSE_BTN = (222, ROW1_Y1, 292, ROW1_Y2)


def in_rect(mx, my, x1, y1, x2, y2):
    return x1 <= mx <= x2 and y1 <= my <= y2


# ==================== MOUSE ====================
def on_mouse(event, mx, my, flags, param):
    global should_quit, canvas, face_strokes, current_color
    global mode_face_filter, mode_mouse_draw, mouse_stroke

    if canvas is None:
        return
    w = canvas.shape[1]

    # --- Toolbar ---
    if my < 78:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if ROW1_Y1 <= my <= ROW1_Y2:
            if in_rect(mx, my, w - 190, ROW1_Y1, w - 105, ROW1_Y2):
                canvas[:] = 0
                face_strokes.clear()
                hand_stroke.reset()
                mouse_stroke.reset()
            elif in_rect(mx, my, w - 95, ROW1_Y1, w - 10, ROW1_Y2):
                should_quit = True
            elif in_rect(mx, my, *FACE_BTN):
                toggle_face_filter()
            elif in_rect(mx, my, *MOUSE_BTN):
                toggle_mouse_draw()
        elif ROW2_Y1 <= my <= ROW2_Y2:
            for i, (name, color) in enumerate(COLORS):
                x1 = 12 + i * 78
                if x1 <= mx <= x1 + 62:
                    current_color = color
                    return
            custom_x = 12 + len(COLORS) * 78
            if custom_x <= mx <= custom_x + 92:
                choose_color()
        return

    # --- Drawing area ---
    if not mode_mouse_draw:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if hand_stroke.is_active():
            return
        mouse_stroke.reset()
        add_stroke_point(mouse_stroke, mx, my)
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_stroke.is_active():
            if flags & cv2.EVENT_FLAG_LBUTTON:
                add_stroke_point(mouse_stroke, mx, my)
            else:
                finish_stroke(mouse_stroke)
    elif event == cv2.EVENT_LBUTTONUP:
        if mouse_stroke.is_active():
            finish_stroke(mouse_stroke)


# ==================== KHOI TAO WEBCAM / WINDOW ====================
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("Khong mo duoc webcam!")

cv2.namedWindow("Air Drawing")
cv2.setMouseCallback("Air Drawing", on_mouse)

# ==================== VONG LAP CHINH ====================
while True:
    if should_quit:
        break

    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int(time.monotonic() * 1000)

    hand_result = hand_landmarker.detect_for_video(mp_image, ts)
    face_result = face_landmarker.detect_for_video(mp_image, ts)

    # Cap nhat transform khuon mat (giu lai transform cu neu mat bien mat)
    if face_result.face_landmarks:
        t = compute_face_transform(face_result.face_landmarks[0], w, h)
        if t is not None:
            face_transform = t

    # --- Xu ly ban tay ---
    hand_drawing = False
    cursor = None
    if hand_result.hand_landmarks:
        hand_landmarks = hand_result.hand_landmarks[0]
        pts_xy = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        cursor = pts_xy[8]
        hand_drawing = index_is_extended(hand_landmarks)
        draw_hand(frame, pts_xy)

        if hand_drawing:
            add_stroke_point(hand_stroke, cursor[0], cursor[1])
        else:
            if hand_stroke.is_active():
                finish_stroke(hand_stroke)
    else:
        if hand_stroke.is_active():
            finish_stroke(hand_stroke)

    # --- Render ---
    output = cv2.addWeighted(frame, 0.65, canvas, 1.0, 0)

    if mode_face_filter and face_transform is not None:
        for color, thickness, pts in face_strokes:
            draw_face_stroke_on_frame(output, pts, color, thickness)
        if hand_stroke.face_pts is not None:
            draw_face_stroke_on_frame(
                output, hand_stroke.face_pts,
                hand_stroke.color, hand_stroke.thickness)
        if mouse_stroke.face_pts is not None:
            draw_face_stroke_on_frame(
                output, mouse_stroke.face_pts,
                mouse_stroke.color, mouse_stroke.thickness)

    # --- Toolbar ---
    cv2.rectangle(output, (0, 0), (w, 78), (35, 35, 35), -1)
    cv2.putText(output, "AIR DRAWING", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    face_bg = (0, 180, 0) if mode_face_filter else (70, 70, 70)
    cv2.rectangle(output, FACE_BTN[:2], FACE_BTN[2:], face_bg, -1)
    cv2.putText(output, "FACE", (FACE_BTN[0] + 8, ROW1_Y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    mouse_bg = (0, 180, 0) if mode_mouse_draw else (70, 70, 70)
    cv2.rectangle(output, MOUSE_BTN[:2], MOUSE_BTN[2:], mouse_bg, -1)
    cv2.putText(output, "MOUSE", (MOUSE_BTN[0] + 4, ROW1_Y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(output, (w - 190, ROW1_Y1), (w - 105, ROW1_Y2), (0, 0, 180), -1)
    cv2.putText(output, "CLEAR", (w - 178, ROW1_Y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(output, (w - 95, ROW1_Y1), (w - 10, ROW1_Y2), (70, 70, 70), -1)
    cv2.putText(output, "QUIT", (w - 80, ROW1_Y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, (name, color) in enumerate(COLORS):
        x1 = 12 + i * 78
        cv2.rectangle(output, (x1, ROW2_Y1), (x1 + 62, ROW2_Y2), color, -1)
        if color == current_color:
            cv2.rectangle(output, (x1, ROW2_Y1), (x1 + 62, ROW2_Y2), (255, 255, 255), 3)

    custom_x = 12 + len(COLORS) * 78
    cv2.rectangle(output, (custom_x, ROW2_Y1), (custom_x + 92, ROW2_Y2),
                  (100, 100, 100), -1)
    cv2.putText(output, "CUSTOM", (custom_x + 5, ROW2_Y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # --- Status bar duoi cung ---
    status_parts = ["DRAWING" if hand_drawing else "PEN UP"]
    if mode_mouse_draw:
        status_parts.append("MOUSE ON")
    if mode_face_filter:
        status_parts.append("NO FACE" if face_transform is None else "ON FACE")
    status = " | ".join(status_parts)
    status_color = (0, 255, 0) if hand_drawing else (0, 180, 255)
    cv2.putText(output, status, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

    if cursor is not None:
        c = current_color if hand_drawing else (255, 255, 255)
        cv2.circle(output, cursor, 9, c, 2)

    cv2.imshow("Air Drawing", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('c'):
        choose_color()
    elif key == ord('x'):
        canvas[:] = 0
        face_strokes.clear()
        hand_stroke.reset()
        mouse_stroke.reset()
    elif key == ord('f'):
        toggle_face_filter()
    elif key == ord('m'):
        toggle_mouse_draw()

cap.release()
hand_landmarker.close()
face_landmarker.close()
cv2.destroyAllWindows()