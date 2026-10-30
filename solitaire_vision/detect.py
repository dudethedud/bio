"""Step 1: find card-shaped regions in a frame.

Face-up cards are bright, nearly-white rounded rectangles. We threshold on
"bright and unsaturated" in HSV, find external contours, and keep the ones
whose bounding box has a playing-card aspect ratio (~0.70 width/height).

Face-down cards are found the same way but with a colour mask for the card
back, which you configure (Disney Solitaire's back colour depends on the
selected deck, so it's a parameter rather than a constant).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Card:
    x: int
    y: int
    w: int
    h: int
    face_up: bool = True
    free: bool = True  # not covered by another card -> can be played/flipped
    rank: str | None = None
    suit: str | None = None
    score: float = 0.0
    corner: np.ndarray | None = field(default=None, repr=False)

    @property
    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def label(self):
        if not self.face_up:
            return "??"
        return f"{self.rank or '?'}{self.suit or '?'}"


@dataclass
class DetectConfig:
    # Face-up cards: low saturation, high value.
    white_max_sat: int = 45
    white_min_val: int = 200
    # Face-down cards: HSV range of the card back (None disables).
    back_lower: tuple | None = None
    back_upper: tuple | None = None
    # Shape filters, relative to frame area so they work at any resolution.
    min_area_frac: float = 0.002
    max_area_frac: float = 0.08
    aspect_min: float = 0.55
    aspect_max: float = 0.85
    # Fraction of the card (width, height) used as the rank/suit corner.
    corner_w: float = 0.30
    corner_h: float = 0.38
    # Horizontal overlap (fraction of the narrower card) that counts as "covered".
    cover_overlap: float = 0.35


def _mask_white(hsv, cfg):
    lower = np.array([0, 0, cfg.white_min_val], np.uint8)
    upper = np.array([180, cfg.white_max_sat, 255], np.uint8)
    return cv2.inRange(hsv, lower, upper)


def _boxes_from_mask(mask, frame_area, cfg, allow_partial=False):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if not (cfg.min_area_frac * frame_area <= area <= cfg.max_area_frac * frame_area):
            continue
        aspect = w / h
        if cfg.aspect_min <= aspect <= cfg.aspect_max:
            boxes.append((x, y, w, h))
        elif allow_partial and aspect < cfg.aspect_min:
            # Overlapped cards in a peak can be cut short; keep them anyway.
            boxes.append((x, y, w, h))
    return boxes


def find_cards(frame, cfg=None):
    """Return a list of Card objects (unidentified) found in the frame."""
    cfg = cfg or DetectConfig()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    area = frame.shape[0] * frame.shape[1]

    cards = []
    for x, y, w, h in _boxes_from_mask(_mask_white(hsv, cfg), area, cfg):
        cw, ch = max(1, int(w * cfg.corner_w)), max(1, int(h * cfg.corner_h))
        corner = frame[y:y + ch, x:x + cw].copy()
        cards.append(Card(x, y, w, h, face_up=True, corner=corner))

    if cfg.back_lower is not None and cfg.back_upper is not None:
        back = cv2.inRange(hsv, np.array(cfg.back_lower, np.uint8),
                           np.array(cfg.back_upper, np.uint8))
        for x, y, w, h in _boxes_from_mask(back, area, cfg, allow_partial=True):
            cards.append(Card(x, y, w, h, face_up=False))

    # Top-to-bottom, left-to-right: stable ordering for printing / diffing.
    cards.sort(key=lambda c: (c.y // max(1, c.h // 3), c.x))
    mark_coverage(cards, cfg)
    return cards


def mark_coverage(cards, cfg=None):
    """Set card.free for each card.

    In a TriPeaks-style layout, a card that sits lower on screen and overlaps
    the bottom of another card is drawn in front of it, so the upper card is
    covered and cannot be played until the lower one is removed. We flag a
    card as not free when another card starts below its top edge, still within
    its vertical span, and overlaps it horizontally.
    """
    cfg = cfg or DetectConfig()
    for a in cards:
        a.free = True
    for a in cards:
        for b in cards:
            if b is a:
                continue
            # b must start lower than a (drawn in front) but still overlap a.
            if not (a.y < b.y < a.y + a.h):
                continue
            overlap_x = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
            if overlap_x > cfg.cover_overlap * min(a.w, b.w):
                a.free = False
                break
    return cards
