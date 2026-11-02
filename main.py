"""Disney Solitaire screen recognition - command line entry point.

    python main.py pick                            # drag a box over the game, save it
    python main.py detect --image shot.png --out annotated.png
    python main.py detect --screen                 # uses the saved region
    python main.py label  --image shot.png
    python main.py watch  --interval 1             # live, uses the saved region
"""

import argparse
import os
import time

import cv2

from solitaire_vision import (DetectConfig, TemplateSet, best_move, find_cards,
                              playable, split_foundation)
from solitaire_vision.capture import (grab_screen, load_image, load_region,
                                      pick_region, save_region)
from solitaire_vision.identify import save_template, split_corner


def resolve_region(args):
    """Region from --region, else the saved --region-file if it exists, else None."""
    if getattr(args, "region", None):
        return tuple(args.region)
    rf = getattr(args, "region_file", None)
    if rf and os.path.exists(rf):
        return load_region(rf)
    return None


def get_frame(args):
    if getattr(args, "image", None):
        return load_image(args.image)
    return grab_screen(resolve_region(args))


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
    moves = playable(foundation, tableau)
    return cards, foundation, moves, best_move(foundation, tableau)


def annotate(frame, cards, foundation, moves, pick):
    out = frame.copy()
    move_ids = {id(c) for c in moves}
    for c in cards:
        if c is foundation:
            color = (255, 128, 0)        # blue  = foundation / waste
        elif c is pick:
            color = (0, 255, 0)          # bright green = recommended move
        elif id(c) in move_ids:
            color = (0, 170, 90)         # dim green = other legal move
        elif c.face_up:
            color = (0, 200, 255) if c.free else (120, 120, 200)  # yellow free / dim covered
        else:
            color = (128, 128, 128)      # grey = face-down
        thick = 4 if c is pick else 2
        cv2.rectangle(out, (c.x, c.y), (c.x + c.w, c.y + c.h), color, thick)
        cv2.putText(out, c.label, (c.x + 2, c.y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def report(cards, foundation, moves, pick):
    up = [c for c in cards if c.face_up]
    print(f"cards: {len(cards)} ({len(up)} face up, {len(cards) - len(up)} face down)")
    for c in up:
        state = "free" if c.free else "covered"
        print(f"  {c.label:>4} at {c.center}  match={c.score:.2f}  {state}")
    print(f"foundation: {foundation.label if foundation else 'not found'}")
    print("playable: " + (", ".join(f"{c.label}@{c.center}" for c in moves) or "none"))
    if pick:
        print(f"==> best move: play {pick.label} at {pick.center}")
    elif foundation:
        print("==> no legal move - draw from the deck (or use a wild)")


def cmd_detect(args):
    frame = get_frame(args)
    templates = TemplateSet(args.templates)
    if not templates:
        print(f"(no templates in {args.templates}/ - run `label` first; showing boxes only)")
    cards, foundation, moves, pick = analyse(frame, make_config(args), templates)
    report(cards, foundation, moves, pick)
    if args.out:
        cv2.imwrite(args.out, annotate(frame, cards, foundation, moves, pick))
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


def cmd_pick(args):
    region = pick_region(monitor=args.monitor)
    if region is None:
        print("no region selected.")
        return
    save_region(args.region_file, region)
    left, top, w, h = region
    print(f"region saved to {args.region_file}: left={left} top={top} width={w} height={h}")
    print("detect/watch will now use it automatically, or pass it explicitly with:")
    print(f"  --region {left} {top} {w} {h}")


def cmd_watch(args):
    templates = TemplateSet(args.templates)
    cfg = make_config(args)
    last = None
    while True:
        cards, foundation, moves, pick = analyse(get_frame(args), cfg, templates)
        state = (tuple(c.label for c in cards), foundation.label if foundation else None)
        if state != last:
            print("-" * 40)
            report(cards, foundation, moves, pick)
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
        s.add_argument("--region", type=int, nargs=4, metavar=("LEFT", "TOP", "W", "H"),
                       help="screen pixels to capture; overrides --region-file")
        s.add_argument("--region-file", default="region.json",
                       help="saved region from `pick` (used when --region is omitted)")
        s.add_argument("--templates", default="templates")
        s.add_argument("--back-hsv", type=int, nargs=6, metavar="N",
                       help="HSV lower+upper bounds of the card back colour, e.g. 100 120 80 130 255 255")
        if name == "detect":
            s.add_argument("--out", help="write an annotated copy of the frame here")
        if name == "watch":
            s.add_argument("--interval", type=float, default=1.0)

    sp = sub.add_parser("pick", help="drag a box over the game window and save it")
    sp.set_defaults(fn=cmd_pick)
    sp.add_argument("--region-file", default="region.json", help="where to save the region")
    sp.add_argument("--monitor", type=int, default=1, help="monitor number to capture from")

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
