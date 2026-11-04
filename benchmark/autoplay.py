"""Auto-play a TriPeaks deal to completion and print the move sequence.

With a known deal (seed), it solves the board optimally and prints the full
winning line -- an 'auto-complete'. On a real game you'd tap each move yourself,
because the tool can read the board but never sends input to the device.

    python benchmark/autoplay.py --seed 7
"""

import argparse
import random

from simulate import RANKS, new_deal
from solver import solve


def describe(tableau0, stock0, waste0, moves):
    tab = dict(tableau0)
    stock = list(stock0)
    waste = waste0
    out = [f"start: waste={waste}  tableau has {len(tab)} cards"]
    step = 0
    for mv in moves:
        step += 1
        if mv[0] == "play":
            pos = mv[1]
            out.append(f"  {step:2d}. play {tab[pos]:>2} (slot {pos}) onto {waste}")
            waste = tab.pop(pos)
        else:
            waste = stock.pop()
            out.append(f"  {step:2d}. draw -> {waste}")
    out.append(f"result: {'CLEARED' if not tab else str(len(tab)) + ' left'}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tries", type=int, default=200, help="deals to scan for a solvable one")
    args = ap.parse_args()

    for i in range(args.tries):
        tableau, stock, waste = new_deal(random.Random(args.seed + i))
        moves = solve(tableau, stock, waste)
        if moves:
            print(f"deal seed {args.seed + i}: solvable in {len(moves)} moves\n")
            print(describe(tableau, stock, waste, moves))
            return
        print(f"deal seed {args.seed + i}: no winning line exists (unwinnable deal)")
    print("no solvable deal found in the scanned range")


if __name__ == "__main__":
    main()
