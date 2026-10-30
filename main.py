"""Disney Solitaire screen recognition - command line entry point.

    python main.py detect --image shot.png --out annotated.png
    python main.py detect --screen --region 0 0 1080 1920
    python main.py label  --image shot.png
    python main.py watch  --region 0 0 1080 1920 --interval 1
"""

import argparse
import time

import cv2

from solitaire_vision import DetectConfig, TemplateSet, find_cards, playable, split_foundation
from solitaire_vision.capture import grab_screen, load_image
from solitaire_vision.identify import save_template, split_corner


def get_frame(args):
    if args.image:
        return load_image(args.image)
    return grab_screen(tuple(args.region) if args.region else None)


def make_config(args):
    cfg = DetectConfig()
    if args.back_hsv:
        cfg.back_lower, cfg.back_upper = tuple(args.back_hsv[:3]), tuple(args.back_hsv[3:])
    return cfg


def analyse(frame, cfg, templates):
    cards = find_cards(frame, cfg)
    if templates:
        for c in cards:
            templates.identify(c)
    foundation, tableau = split_foundation(cards, frame.shape[0])
    return cards, foundation, playable(foundation, tableau)


def annotate(frame, cards, foundation, moves):
    out = frame.copy()
    move_ids = {id(c) for c in moves}
    for c in cards:
        if c is foundation:
            color = (255, 128, 0)
        elif id(c) in move_ids:
            color = (0, 200, 0)
        elif c.face_up:
            color = (0, 200, 255)
        else:
            color = (128, 128, 128)
        cv2.rectangle(out, (c.x, c.y), (c.x + c.w, c.y + c.h), color, 2)
        cv2.putText(out, c.label, (c.x + 2, c.y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def report(cards, foundation, moves):
    up = [c for c in cards if c.face_up]
    print(f"cards: {len(cards)} ({len(up)} face up, {len(cards) - len(up)} face down)")
    for c in up:
        print(f"  {c.label:>4} at {c.center}  match={c.score:.2f}")
    print(f"foundation: {foundation.label if foundation else 'not found'}")
    print("playable: " + (", ".join(f"{c.label}@{c.center}" for c in moves) or "none"))


def cmd_detect(args):
    frame = get_frame(args)
    templates = TemplateSet(args.templates)
    if not templates:
        print(f"(no templates in {args.templates}/ - run `label` first; showing boxes only)")
    cards, foundation, moves = analyse(frame, make_config(args), templates)
    report(cards, foundation, moves)
    if args.out:
        cv2.imwrite(args.out, annotate(frame, cards, foundation, moves))
        print(f"annotated image written to {args.out}")


def cmd_label(args):
    """Walk through detected face-up cards and type their rank/suit to build templates."""
    frame = get_frame(args)
    cards = [c for c in find_cards(frame, make_config(args)) if c.face_up]
    print(f"{len(cards)} face-up cards found. For each, type e.g. 'QH', '10S', 'AD'; blank to skip.")
    for i, c in enumerate(cards):
        preview = f"card_{i}.png"
        cv2.imwrite(preview, frame[c.y:c.y + c.h, c.x:c.x + c.w])
        answer = input(f"[{i}] at {c.center} (saved crop to {preview}): ").strip().upper()
        if not answer:
            continue
        rank, suit = answer[:-1], answer[-1]
        rank_g, suit_g = split_corner(c.corner)
        save_template(args.templates, "rank", rank, rank_g)
        save_template(args.templates, "suit", suit, suit_g)
    print(f"templates saved under {args.templates}/")


def cmd_watch(args):
    templates = TemplateSet(args.templates)
    cfg = make_config(args)
    last = None
    while True:
        cards, foundation, moves = analyse(get_frame(args), cfg, templates)
        state = (tuple(c.label for c in cards), foundation.label if foundation else None)
        if state != last:
            print("-" * 40)
            report(cards, foundation, moves)
            last = state
        time.sleep(args.interval)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("detect", cmd_detect), ("label", cmd_label), ("watch", cmd_watch)]:
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        src = s.add_mutually_exclusive_group()
        src.add_argument("--image", help="screenshot file instead of live screen")
        src.add_argument("--screen", action="store_true", help="capture the live screen (default)")
        s.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "W", "H"))
        s.add_argument("--templates", default="templates")
        s.add_argument("--back-hsv", type=int, nargs=6, metavar="N",
                       help="HSV lower+upper bounds of the card back colour, e.g. 100 120 80 130 255 255")
        if name == "detect":
            s.add_argument("--out", help="write an annotated copy of the frame here")
        if name == "watch":
            s.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
