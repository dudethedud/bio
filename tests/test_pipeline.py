"""End-to-end test on a synthetic board (no real game assets needed)."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solitaire_vision import DetectConfig, TemplateSet, find_cards, playable, split_foundation  # noqa: E402
from solitaire_vision.identify import save_template, split_corner  # noqa: E402

CARD_W, CARD_H = 70, 100
GREEN = (40, 110, 30)
BACK = (160, 60, 20)  # blue-ish in BGR


def draw_card(img, x, y, rank, suit):
    cv2.rectangle(img, (x, y), (x + CARD_W, y + CARD_H), (250, 250, 250), -1)
    color = (0, 0, 220) if suit in "HD" else (10, 10, 10)
    cv2.putText(img, rank, (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # Stand-in suit glyphs: distinct shapes per suit.
    cx, cy = x + 11, y + 29
    if suit == "H":
        cv2.circle(img, (cx, cy), 5, color, -1)
    elif suit == "D":
        cv2.fillPoly(img, [np.array([(cx, cy - 6), (cx + 5, cy), (cx, cy + 6), (cx - 5, cy)])], color)
    elif suit == "S":
        cv2.fillPoly(img, [np.array([(cx, cy - 6), (cx + 6, cy + 5), (cx - 6, cy + 5)])], color)
    else:
        cv2.rectangle(img, (cx - 5, cy - 5), (cx + 5, cy + 5), color, -1)


def make_board(layout):
    img = np.full((600, 800, 3), GREEN, np.uint8)
    for x, y, rank, suit in layout:
        draw_card(img, x, y, rank, suit)
    cv2.rectangle(img, (600, 60), (600 + CARD_W, 60 + CARD_H), BACK, -1)  # face-down
    return img


LAYOUT = [(50, 60, "7", "H"), (150, 60, "9", "S"), (250, 60, "Q", "D"),
          (350, 60, "K", "C"), (450, 60, "A", "S"), (360, 470, "8", "D")]


def test_detects_and_identifies(tmp_path):
    board = make_board(LAYOUT)
    cfg = DetectConfig(back_lower=(100, 150, 100), back_upper=(130, 255, 255))
    cards = find_cards(board, cfg)
    assert sum(c.face_up for c in cards) == len(LAYOUT)
    assert sum(not c.face_up for c in cards) == 1

    # "Label" every card from the layout to build templates.
    by_pos = {(x, y): (r, s) for x, y, r, s in LAYOUT}
    for c in cards:
        if c.face_up:
            r, s = by_pos[(c.x, c.y)]
            rg, sg = split_corner(c.corner)
            save_template(tmp_path, "rank", r, rg)
            save_template(tmp_path, "suit", s, sg)

    templates = TemplateSet(tmp_path)
    for c in find_cards(board, cfg):
        templates.identify(c)
        if c.face_up:
            assert (c.rank, c.suit) == by_pos[(c.x, c.y)]

    foundation, tableau = split_foundation(cards, board.shape[0])
    for c in cards:
        templates.identify(c)
    assert foundation.label == "8D"
    assert sorted(c.label for c in playable(foundation, tableau)) == ["7H", "9S"]


def test_wraparound():
    from solitaire_vision import adjacent
    assert adjacent("K", "A") and adjacent("A", "K") and adjacent("10", "J")
    assert not adjacent("5", "5") and not adjacent("2", "4")
