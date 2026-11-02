"""Getting frames: from the live screen (via mss) or from a saved screenshot."""

import json

import cv2
import numpy as np


def grab_screen(region=None, monitor=1):
    """Return a BGR frame of the screen.

    region: optional (left, top, width, height) in screen pixels, e.g. the
    emulator / game window. If omitted, the whole monitor is captured.
    """
    import mss  # imported lazily so the rest works without a display

    with mss.mss() as sct:
        if region:
            left, top, width, height = region
            box = {"left": left, "top": top, "width": width, "height": height}
        else:
            box = sct.monitors[monitor]
        shot = np.array(sct.grab(box))  # BGRA
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def scale_roi(roi, scale, offset=(0, 0)):
    """Map a box picked on a (possibly shrunk) preview back to screen pixels.

    roi:    (x, y, w, h) in the displayed preview
    scale:  preview_size / real_size  (<= 1.0)
    offset: the monitor's (left, top) in the virtual screen
    """
    x, y, w, h = roi
    inv = 1.0 / scale
    return (int(round(offset[0] + x * inv)), int(round(offset[1] + y * inv)),
            int(round(w * inv)), int(round(h * inv)))


def pick_region(monitor=1, max_display=1400):
    """Show the screen and let the user drag a box over the game window.

    Returns (left, top, width, height) in screen pixels, or None if cancelled.
    Needs a desktop (OpenCV highgui window).
    """
    import mss

    with mss.mss() as sct:
        mon = sct.monitors[monitor]
        frame = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2BGR)

    h, w = frame.shape[:2]
    scale = min(1.0, max_display / max(h, w))
    disp = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame

    title = "Drag a box over the game, then press ENTER (c to cancel)"
    roi = cv2.selectROI(title, disp, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:  # ENTER with no selection, or cancelled
        return None
    return scale_roi(roi, scale, (mon.get("left", 0), mon.get("top", 0)))


def save_region(path, region):
    with open(path, "w") as f:
        json.dump(list(region), f)


def load_region(path):
    with open(path) as f:
        data = json.load(f)
    if len(data) != 4:
        raise ValueError(f"{path} does not contain a 4-number region")
    return tuple(int(v) for v in data)
