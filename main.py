import argparse
import time

import cv2
import numpy as np

DICT_ID = cv2.aruco.DICT_6X6_250
MARKER_ID = 23
SMOOTH = 0.6          # pose smoothing (0 = none, closer to 1 = smoother but laggier)
HOLD_FRAMES = 8       # keep drawing last pose briefly if the marker is lost


# ----------------------------------------------------------------------------
# 1. Skeleton model, defined in MARKER coordinates.
#    Marker side = 1 unit, origin = marker centre, x right, y up, z towards viewer.
#    The skeleton "stands" on top of the marker.
# ----------------------------------------------------------------------------
OFFSET = np.array([0.0, 0.65, 0.0])


def P(*pts):
    return np.array(pts, dtype=float) + OFFSET


def loop(cx, cy, cz, rx, rz, n=24, tilt=0.0):
    t = np.linspace(0, 2 * np.pi, n)
    z = cz + rz * np.sin(t)
    return np.stack([cx + rx * np.cos(t), cy + tilt * np.sin(t), z], axis=1) + OFFSET


def build_skeleton():
    parts = {}

    # Skull
    skull = [loop(0, 1.90, 0, 0.11, 0.12, tilt=0.0)]
    skull.append(np.array([[0.0, 1.90 + np.cos(t) * 0.11, np.sin(t) * 0.12]
                           for t in np.linspace(0, 2 * np.pi, 24)]) + OFFSET)
    skull.append(P([-0.07, 1.80, 0.10], [0.0, 1.75, 0.12], [0.07, 1.80, 0.10]))  # jaw
    parts["Skull"] = dict(
        color=(235, 235, 235), anchor=P([0.15, 1.95, 0])[0], lines=skull,
        info=["Skull (cranium + face)", "22 bones; 8 form the cranium,", "14 form the face. Protects the brain."])

    # Vertebral column
    ys = np.linspace(1.68, 0.98, 9)
    spine = [np.array([[0.0, y, -0.04 + 0.03 * np.sin((1.68 - y) * 4.5)] for y in ys]) + OFFSET]
    parts["Spine"] = dict(
        color=(80, 200, 255), anchor=P([0.06, 1.35, -0.04])[0], lines=spine,
        info=["Vertebral column", "33 vertebrae at birth (26 bones in adults):", "7 cervical, 12 thoracic, 5 lumbar,",
              "sacrum and coccyx. Protects the spinal cord."])

    # Rib cage
    ribs = []
    for k in range(8):
        y = 1.58 - 0.045 * k
        r = 0.09 + 0.022 * min(k, 4) - 0.012 * max(k - 4, 0)
        ribs.append(loop(0, y, 0.0, r + 0.04, r, n=20, tilt=0.03))
    ribs.append(P([0, 1.60, 0.09], [0, 1.32, 0.10]))   # sternum
    parts["Rib cage"] = dict(
        color=(120, 255, 160), anchor=P([0.20, 1.45, 0])[0], lines=ribs,
        info=["Rib cage (thoracic cage)", "12 pairs of ribs: 7 true, 3 false,", "2 floating, plus the sternum.",
              "Protects the heart and lungs."])

    # Shoulder girdle + upper limbs
    arms = [P([-0.26, 1.62, 0], [0, 1.64, 0.02], [0.26, 1.62, 0])]  # clavicles
    for s in (-1, 1):
        arms.append(P([s * 0.26, 1.62, 0], [s * 0.34, 1.32, 0], [s * 0.38, 1.04, 0.03]))   # humerus, radius/ulna
        arms.append(P([s * 0.38, 1.04, 0.03], [s * 0.40, 0.94, 0.03]))                     # hand
    parts["Upper limbs"] = dict(
        color=(255, 190, 90), anchor=P([-0.55, 1.35, 0])[0], lines=arms,
        info=["Upper limbs + shoulder girdle", "30 bones per arm: humerus, radius,", "ulna, 8 carpals, 5 metacarpals,",
              "14 phalanges. Clavicle + scapula at shoulder."])

    # Pelvis + lower limbs
    legs = [P([-0.17, 1.02, 0], [0, 0.94, 0.03], [0.17, 1.02, 0])]
    for s in (-1, 1):
        legs.append(P([s * 0.17, 1.02, 0], [s * 0.15, 0.55, 0], [s * 0.15, 0.10, 0]))      # femur, tibia/fibula
        legs.append(P([s * 0.15, 0.10, 0], [s * 0.15, 0.04, 0.12]))                         # foot
    parts["Lower limbs"] = dict(
        color=(255, 120, 180), anchor=P([0.30, 0.55, 0])[0], lines=legs,
        info=["Pelvis + lower limbs", "30 bones per leg: femur (longest bone),", "patella, tibia, fibula, 7 tarsals,",
              "5 metatarsals, 14 phalanges."])
    return parts


PARTS = build_skeleton()
NAMES = list(PARTS.keys())

# ----------------------------------------------------------------------------
# 2. Pose estimation helpers
# ----------------------------------------------------------------------------
MARKER_OBJ = np.array([[-0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0]], dtype=np.float32)


def estimate_pose(corners, K, dist):
    ok, rvec, tvec = cv2.solvePnP(MARKER_OBJ, corners.reshape(4, 2), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return (rvec, tvec) if ok else (None, None)


def euler_deg(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = np.hypot(R[0, 0], R[1, 0])
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return pitch, yaw, roll


def project(pts, rvec, tvec, K, dist):
    img, _ = cv2.projectPoints(pts.astype(np.float32), rvec, tvec, K, dist)
    return img.reshape(-1, 2).astype(int)


# ----------------------------------------------------------------------------
# 3. Drawing
# ----------------------------------------------------------------------------
def draw_part(frame, name, rvec, tvec, K, dist, labels):
    part = PARTS[name]
    for line in part["lines"]:
        pts = project(line, rvec, tvec, K, dist)
        cv2.polylines(frame, [pts], False, part["color"], 3, cv2.LINE_AA)
        for p in (pts[0], pts[-1]):
            cv2.circle(frame, tuple(p), 4, part["color"], -1, cv2.LINE_AA)
    if labels:
        a = project(part["anchor"].reshape(1, 3), rvec, tvec, K, dist)[0]
        cv2.putText(frame, name, tuple(a + [6, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, name, tuple(a + [6, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, part["color"], 1, cv2.LINE_AA)


def panel(frame, lines, org=(10, 10)):
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] for t in lines) + 20
    h = 22 * len(lines) + 12
    x, y = org
    ov = frame.copy()
    cv2.rectangle(ov, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    for i, t in enumerate(lines):
        col = (255, 255, 255) if i else (0, 255, 255)
        cv2.putText(frame, t, (x + 10, y + 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# 4. Main loop
# ----------------------------------------------------------------------------
def make_marker():
    d = cv2.aruco.getPredefinedDictionary(DICT_ID)
    img = cv2.aruco.generateImageMarker(d, MARKER_ID, 600)
    img = cv2.copyMakeBorder(img, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)  # white quiet zone
    cv2.imwrite("marker_id23.png", img)
    print("Saved marker_id23.png (print it, or show it full-screen on a phone).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-marker", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    if args.make_marker:
        return make_marker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam. Try --camera 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT_ID), cv2.aruco.DetectorParameters())

    K = None
    dist = np.zeros(5)           # replace K/dist with real calibration values for better accuracy
    mode = -1                    # -1 = all parts, else index into NAMES
    labels = True
    rvec = tvec = None
    lost = 999
    t_prev = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1) if False else frame   # keep un-mirrored so pose stays correct
        h, w = frame.shape[:2]
        if K is None:   # approximate pinhole camera: focal length ~ image width
            K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)

        # --- detection ---
        corners, ids, _ = detector.detectMarkers(frame)
        found = False
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                if i != MARKER_ID:
                    continue
                found = True
                cv2.aruco.drawDetectedMarkers(frame, [c], np.array([[i]]), (0, 255, 0))
                # --- pose estimation ---
                r, t = estimate_pose(c, K, dist)
                if r is not None:
                    if rvec is None or lost > HOLD_FRAMES:
                        rvec, tvec = r, t
                    else:                       # --- tracking: exponential smoothing ---
                        rvec = SMOOTH * rvec + (1 - SMOOTH) * r
                        tvec = SMOOTH * tvec + (1 - SMOOTH) * t
                    lost = 0
                break
        if not found:
            lost += 1

        # --- registration + visualisation ---
        status = "MARKER LOST - show marker ID %d" % MARKER_ID
        if rvec is not None and lost <= HOLD_FRAMES:
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.6, 2)
            names = NAMES if mode < 0 else [NAMES[mode]]
            for n in names:
                draw_part(frame, n, rvec, tvec, K, dist, labels)
            pitch, yaw, roll = euler_deg(rvec)
            status = "TRACKING | dist %.1f | pitch %.0f yaw %.0f roll %.0f" % (
                float(np.linalg.norm(tvec)), pitch, yaw, roll)
            if mode >= 0:
                panel(frame, PARTS[NAMES[mode]]["info"], (10, 44))
            else:
                panel(frame, ["Human skeleton (adult): 206 bones", "Press N / P to study one part at a time"], (10, 44))

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        cv2.rectangle(frame, (0, 0), (w, 34), (20, 20, 20), -1)
        cv2.putText(frame, "%s | FPS %.0f" % (status, fps), (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if lost <= HOLD_FRAMES and rvec is not None else (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "N/P part  A all  L labels  S save  Q quit", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("AR Human Skeleton Learning System", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("n"):
            mode = (mode + 1) % len(NAMES)
        elif k == ord("p"):
            mode = (mode - 1) % len(NAMES)
        elif k == ord("a"):
            mode = -1
        elif k == ord("l"):
            labels = not labels
        elif k == ord("s"):
            fn = "screenshot_%d.png" % int(time.time())
            cv2.imwrite(fn, frame)
            print("saved", fn)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()