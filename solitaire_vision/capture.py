"""Getting frames: from the live screen (via mss) or from a saved screenshot."""

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
