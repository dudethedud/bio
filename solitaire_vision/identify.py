"""Step 2: read rank and suit from each card's top-left corner.

Method: template matching. The corner crop is split into a rank glyph (top)
and a suit glyph (below it). Each glyph is binarised, cropped to its ink,
resized to a fixed size, and compared with normalised cross-correlation
against labelled templates you collected with `main.py label`.

Templates live in a folder like:

    templates/rank/A_0.png, templates/rank/10_0.png, ...
    templates/suit/H_0.png, templates/suit/S_0.png, ...

Several samples per label are fine (suffix _0, _1, ...); the best one wins.
"""

from pathlib import Path

import cv2
import numpy as np

GLYPH_SIZE = (32, 40)  # (w, h) every glyph is normalised to
PAD = 4  # blank border so solid glyphs still have contrast to correlate on
RED_SUITS = {"H", "D"}


def split_corner(corner):
    """Split a corner crop into (rank_glyph, suit_glyph) binary images."""
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find horizontal bands of ink: rank is the first band, suit the second.
    rows = ink.sum(axis=1) > 0
    bands, start = [], None
    for i, on in enumerate(rows):
        if on and start is None:
            start = i
        elif not on and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(rows)))
    bands = [b for b in bands if b[1] - b[0] >= 3]  # drop specks

    if len(bands) >= 2:
        rank, suit = ink[bands[0][0]:bands[0][1]], ink[bands[1][0]:bands[1][1]]
    else:  # glyphs touch; fall back to a fixed split
        mid = int(ink.shape[0] * 0.55)
        rank, suit = ink[:mid], ink[mid:]
    return normalise(rank), normalise(suit)


def normalise(glyph):
    """Crop to ink bounding box, resize to GLYPH_SIZE, add a blank border."""
    inner = (GLYPH_SIZE[0] - 2 * PAD, GLYPH_SIZE[1] - 2 * PAD)
    pts = cv2.findNonZero(glyph)
    if pts is None:
        return np.zeros(GLYPH_SIZE[::-1], np.uint8)
    x, y, w, h = cv2.boundingRect(pts)
    g = cv2.resize(glyph[y:y + h, x:x + w], inner, interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(g, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=0)


def is_red(corner):
    """Red vs black ink - a cheap sanity check for the suit match."""
    hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90))
    return cv2.countNonZero(red) > cv2.countNonZero(dark)


class TemplateSet:
    def __init__(self, root):
        self.root = Path(root)
        self.ranks = self._load("rank")
        self.suits = self._load("suit")

    def _load(self, kind):
        out = []
        folder = self.root / kind
        if folder.is_dir():
            for p in sorted(folder.glob("*.png")):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is not None:  # saved templates are already normalised
                    out.append((p.stem.rsplit("_", 1)[0], cv2.resize(img, GLYPH_SIZE)))
        return out

    def __bool__(self):
        return bool(self.ranks and self.suits)

    @staticmethod
    def _best(glyph, templates, allowed=None):
        best_label, best_score = None, -1.0
        for label, tmpl in templates:
            if allowed is not None and label not in allowed:
                continue
            score = float(cv2.matchTemplate(glyph, tmpl, cv2.TM_CCOEFF_NORMED)[0, 0])
            if score > best_score:
                best_label, best_score = label, score
        return best_label, best_score

    def identify(self, card, min_score=0.5):
        """Fill in card.rank / card.suit / card.score in place."""
        if not card.face_up or card.corner is None:
            return card
        rank_g, suit_g = split_corner(card.corner)
        red = is_red(card.corner)
        suits_allowed = RED_SUITS if red else {"S", "C"}
        rank, rs = self._best(rank_g, self.ranks)
        suit, ss = self._best(suit_g, self.suits, suits_allowed)
        if suit is None:  # no templates for that colour yet
            suit, ss = self._best(suit_g, self.suits)
        card.score = min(rs, ss)
        if card.score >= min_score:
            card.rank, card.suit = rank, suit
        return card


def save_template(root, kind, label, glyph):
    folder = Path(root) / kind
    folder.mkdir(parents=True, exist_ok=True)
    i = 0
    while (folder / f"{label}_{i}.png").exists():
        i += 1
    path = folder / f"{label}_{i}.png"
    cv2.imwrite(str(path), glyph)
    return path
